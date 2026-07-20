# build-in modules
import math
import time
import datetime
import logging
import psutil
from typing import Tuple, Union

# third-party modules
import torch

# modules from current project
from CRISP.base._utils import qi_utils as utils
from CRISP.base._utils import metrics
from CRISP.base._utils import KNN
from CRISP.base._utils._logging import start_logging, log_without_format, exception_handler
from CRISP.base._utils.data_utils import splitting
from CRISP.base._utils.multithreadding_utils import multithreadded_execution

from CRISP.base import DEVICE_COUNT


class SHLQI2:
    def __init__(self, 
                max_neighbors:int = None,
                distance_samples: int = None,
                computing_method:str = 'k', 
                max_dist: float = 0,
                logarithmic:bool = False,
                dataset_name:str = 'None',
                input_metric:str = 'Euclid', 
                output_metric:str = 'Euclid',
                devices_available:Union[int, str, list] = DEVICE_COUNT,
    ):
        """Class for SHLQI² computation.

        This class can be treated as a standalone end-to-end computation of 
        MLQI², BLQI², HLQI² and SHLQI².

        Parameters
        ----------
        max_neighbors : int, optional
            Maximum amount of neighbors considered in computation. Default is None
        distance_samples : int, optional
            Amount of samples used in distance computation method. Default is None
        computing_method : str, optional
            Method to compute the MLQI² on ('k', 'distance'). Default is 'k'
        max_dist : float, optional
            Maximum distance. Default is 0
        logarithmic : bool, optional
            Logarithmic distance sampling or normal sampling. Default is False.
        dataset_name : str, optional
            Name of the Dataset (only for logging). Default is 'None'.
        input_metric : str, optional
            Name of the metric for input distance computation. Default is 'Euclid'.
        output_metric : str, optional
            Name of the metric for output distance computation. Default is 'Euclid'.
        devices_available : List, optional
            List of indices, indicating available devices for SHLQI² computation.
            Default is list(range(device_count))
            
        Examples
        --------
        >>> _SHQI2 = SHQI2(torch.randn((10,3)), 
                            torch.randn((10,1)))
        >>> _SHQI2.start()
        """
        self.cuda = devices_available != "cpu"
        start_logging("SHLQI²")
        logging.info('Initiating computation of SHLQI² ...')
        logging.info('Parameters:')
        log_without_format('Dataset name:           ' + dataset_name)
        log_without_format('maximum Neighbors:      ' + str(max_neighbors))
        log_without_format('Computational method:   ' + computing_method)
        if computing_method == 'd':
            log_without_format('Distance samples:       ' + self.num_samples)
        log_without_format('GPU accelerated:        ' + str(self.cuda))
        log_without_format('Input metric:           ' + input_metric)
        log_without_format('Output metric:          ' + output_metric)

        self.result:torch.Tensor

        self.minhl = 0
        self.maxhl = 2.5
        self.binsizehl = 0.025
        self.gamma = 0.5
        self.max_neighbors = max_neighbors

        self.num_samples = distance_samples
        self.computing_method = computing_method
        self.max_dist = max_dist
        self.logarithmic = logarithmic
        self.dataset_name = dataset_name

        self.input_metric = input_metric
        self.output_metric = output_metric

        if self.cuda:
            if isinstance(devices_available, str):
                self.devices_available = [int(devices_available.split(":")[-1])]
            elif isinstance(devices_available, list):
                if isinstance(devices_available[0], str):
                    self.devices_available = [int(device.split(":")[-1]) for device in devices_available]
                elif isinstance(devices_available[0], int):
                    self.devices_available = devices_available
            elif isinstance(devices_available, int):
                self.devices_available = list(range(devices_available))
        else:
            self.devices_available = [devices_available]

    def start(self, 
            inputdata:torch.Tensor, 
            outputdata:torch.Tensor) -> None:
        """function for starting the SHLQI² computation.

        This function contains all computational steps for an efficient and multithreadding
        ready computation of BLQI², MLQI², HLQI² and SHLQI². With ´splitting()´ and
        ´multithreadded_execution()´, the computation is split across all devices available. 
        For the computation of BLQI², MLQI² and HLQI² the following four steps are executed 
        in each case.

        - 1st Step:     determine the separation of Data onto the available devices based on 
                        their dedicated available memory (~90% will be used).
        - 2nd Step:     create a thread per available computational device and distribute the 
                        data across those devices regarding the previously computed separation.
        - 3rd Step:     start threads to compute the determined parts
        - 4th Step:     catch the results of the independant threads and put them together wrt 
                        their specified range of computed datapoints
        """     
        st = time.time()
        
        log_without_format('Datapoints:             ' + str(inputdata.shape[0]))
        log_without_format('Input dimensions:       ' + str(inputdata.shape[1]))
        log_without_format('Output dimensions:      ' + str(outputdata.shape[1]))

        # set the maximum neighbors based on dataset length and argument
        if self.max_neighbors is not None:
            self.max_neighbors = min(len(inputdata), self.max_neighbors)
        else: 
            self.max_neighbors = len(inputdata)

        if self.computing_method=='k':
            self.samples = self.max_neighbors     
        else:
            self.samples = self.num_samples

        # compute the k nearest neighbors based on the choosen metric
        input_shape = inputdata.size()
        output_shape = outputdata.size()
        self.inputdata = inputdata
        self.outputdata = outputdata
        
        input_metric:metrics.Metric = metrics.metrics[self.input_metric](input_shape)
        output_metric:metrics.Metric = metrics.metrics[self.output_metric](output_shape)

        metric = KNN(input_metric, 
                    output_metric, 
                    self.max_neighbors,
                    self.devices_available)
        self.knns = metric.get_knns(inputdata)
        del metric
        torch.cuda.empty_cache()

        if self.computing_method == 'd':
            if self.logarithmic:
                self.dist_samples = torch.logspace(0, self.max_dist, self.num_samples)
            else:
                self.dist_samples = torch.linspace(start=0, stop=self.max_dist, 
                                                   steps=self.num_samples)

        #######################################################################################
        # BLQI²

        # compute the possible points per device (ppd) based on the available memory  
        # 2*(x'*k) + 3*x' = floats_on_device
        if self.cuda:
            free_mem = [(torch.cuda.mem_get_info(device)[0] * 0.9) for device in self.devices_available]
        else:
            free_mem = [psutil.virtual_memory().available * 0.9]
        ppd = [math.floor((mem*8/32) / (2*self.knns.size(1) + 3)) for mem in free_mem]

        pts = splitting(torch.tensor(ppd), self.knns.size(0), self.devices_available)

        # initiate multithreadded execution of BLQI² computation
        results = multithreadded_execution(pts, self.compute_blqi2, [], 
                                           [lambda s,e,d : self.knns[s:e].to(d)],
                                           returns=['BLQI2', 'start', 'end', 'device'],
                                           computational_stage='BLQI²')

        # initiating the computed variables
        BLQI2 = torch.zeros((self.inputdata.shape[0], self.max_neighbors), dtype=torch.int8)

        # set back together full Tensor
        for result in results:
            BLQI2[result['start']:result['end']] = result['BLQI2']
        #######################################################################################

        #######################################################################################
        # MLQI²
        # compute the possible points per device (ppd) based on the available memory  
        # 6*(x'*k) + 5*(k²) + (x'*dim_I*k) + (x'*dim_O*k) + x' = floats_on_device 
        if self.cuda:
            free_mem = [(torch.cuda.mem_get_info(device)[0] * 0.9) for device in self.devices_available]
        else:
            free_mem = [psutil.virtual_memory().available * 0.9]
        ppd = torch.tensor([(mem*8/32 - 5*self.max_neighbors**2) / 
                            (6*self.max_neighbors + \
                            self.inputdata.size(1)*self.max_neighbors + \
                            self.outputdata.size(1)*self.max_neighbors + 1) for mem in free_mem])
        pts = splitting(ppd, self.inputdata.size(0), self.devices_available)
        
        # initiate multithreadded execution of MLQI² computation
        results = multithreadded_execution(pts, self.compute_mlqi2, [], 
                                            [lambda s,e,d : self.inputdata[self.knns[s:e]].to(d), 
                                            lambda s,e,d : self.outputdata[self.knns[s:e]].to(d)],
                                            returns=['MLQI2', 'start', 'end', 'device'],
                                            computational_stage='MLQI²')

        # initiating the computed variables
        MLQI2 = torch.zeros((self.inputdata.shape[0], self.samples),
                            device='cpu',
                            dtype=torch.float64)

        # set back together full Tensor
        for result in results:
            MLQI2[result['start']:result['end']] = result['MLQI2']
            
        MLQI2[MLQI2 >= 100] = ((MLQI2[MLQI2 >= 100] - 100) ** 0.5) + 100

        if self.maxhl <= MLQI2.max():
            maxhl = (MLQI2.max()+0.5).item()
        else:
            maxhl = self.maxhl
    
        
        #######################################################################################
            
        #######################################################################################
        # HLQI²
        # compute the possible points per device (ppd) based on the available memory  
        # 5*(x'*k) + (v*k) = floats_on_device
        if self.cuda:
            free_mem = [(torch.cuda.mem_get_info(device)[0] * 0.9) for device in self.devices_available]
        else:
            free_mem = [psutil.virtual_memory().available * 0.9]
        ppd = torch.tensor([(mem*8/32 - ((maxhl-self.minhl)/self.binsizehl)*self.max_neighbors) \
                            / (5*self.max_neighbors) for mem in free_mem])
        pts = splitting(ppd, self.inputdata.size(0), self.devices_available)
        
        # initiate multithreadded execution of HLQI² computation
        results = multithreadded_execution(pts, self.compute_hlqi2, [self.minhl, maxhl], 
                                            [lambda s,e,d: MLQI2[s:e].to(d),
                                            lambda s,e,d: BLQI2[s:e].to(d)], 
                                            returns=['HLQI2', 'assigned_values', 'start', 'end', 'device'],
                                            computational_stage='HLQI²')

        # initiating the computed variables
        self.assigned_values = torch.zeros((self.inputdata.shape[0], self.samples),
                                           device='cpu',
                                           dtype=torch.int16)

        # set back together full Tensor
        hlqi2_max = 0
        for result in results:
            if result['HLQI2'].size(0) >= hlqi2_max:
                hlqi2_max = result['HLQI2'].size(0)
        
        self.HLQI2 = torch.zeros((hlqi2_max,self.samples), dtype=torch.float64, device='cpu')
        for result in results:
            self.HLQI2[0:result['HLQI2'].size(0)] += result['HLQI2'].detach().cpu()
            self.assigned_values[result['start']:result['end']] += result['assigned_values'].detach().cpu()

        #######################################################################################

        self.get_histogram()

        logging.info('\nTime needed for ' + str(self.inputdata.size(0)) + ' Datapoints with ' + str(self.samples) \
            + ' samples per Datapoint: ' + str(datetime.timedelta(seconds=time.time()-st))[:7])

    def compute_blqi2(self, knns:torch.Tensor, #
                      start:int, end:int, device:int) -> Tuple[torch.Tensor, int, int, int]:
        """function for the multithreadded computation of the BLQI² and indices of
        equal areas for different Datapoints. The huge advantage of computing
        `inds` is that areas around different Datapoints containing the exact same 
        Datapoints will have the same MLQI². Therefore we only have to calculate 
        this value once and know where to fill it into the MLQI²
        Args:
            knns (torch.Tensor): matrix of the indices of k-nearest neighbors per points
            start (int): index for the start of the computation in the current thread
            end (int): index for the end of the computation in the current thread
            device (int): the device for the computation

        Returns:
            BLQI2 (torch.Tensor): computed BLQI² for the whole dataset
            start (int): index for the start of the computation in the current thread
            end (int): index for the end of the computation in the current thread
            device (int): the device for the computation
        """
        
        do = f"blqi2_do_{self.computing_method}"
        if hasattr(self, do) and callable(func := getattr(self, do)):
            return func(knns, device), start, end, device
        else:
            raise NameError(f'Name {do} is not defined. You need to choose `computing_method="d"` or `computing_method="k"`')
    

    @exception_handler('error during computation of the BLQI²')
    def blqi2_do_k(self, knns:torch.Tensor, device:int) -> torch.Tensor:
        """Inner function for multithreadded computation.

        Args:
            knns (torch.Tensor): sliced part of the k neighbor indices for the current thread
            device (int): the device for the computation

        Returns:
            BLQI2 (torch.Tensor): computed slice of the BLQI²
        """
        BLQI2 = torch.zeros((knns.size()), dtype=torch.int8, device=device)
        for k in range(knns.size(1)):
            _, ind = torch.unique(torch.sort(knns[:,:k+1])[0], return_inverse=True, dim=0)
            perm = torch.arange(ind.size(0), dtype=ind.dtype, device=ind.device)
            inverse, perm = ind.flip([0]), perm.flip([0])
            perm = torch.zeros_like(ind.unique()).scatter_(0, inverse, perm)
            BLQI2[perm, k] = 1
            if k % math.ceil(90 * (1 - ((k+1) / self.max_neighbors)) + 10) == 0:
                torch.cuda.empty_cache()
        return BLQI2.detach().cpu()


    def compute_mlqi2(self, 
                    inputdata:torch.Tensor, 
                    outputdata:torch.Tensor, 
                    start:int, end:int, device:int,
    ) -> Tuple[torch.Tensor, int, int, int]:
        """function to determine the computational method for the MLQI²
        computational methods are:
        - k-nearest-neighbors: MLQI² regarding a growing neighborhood for every datapoint
        - distance dependant: MLQI² regarding a growing (euclidean) distance around every datapoint

        Args:
            inputdata (torch.Tensor): sliced part of the inputdata with every neighborhood to every point in this slice
            outputdata (torch.Tensor): sliced part of the outputdata with every neighborhood to every point in this slice
            start (int): index for the start of the computation in the current thread
            end (int): index for the end of the computation in the current thread
            device (int): the device for the computation

        Returns:
            MLQI2 (torch.Tensor): computed slice of the MLQI²
            start (int): index for the start of the computation in the current thread
            end (int): index for the end of the computation in the current thread
            device (int): the device for the computation
        """
        do = f"mlqi2_do_{self.computing_method}"
        if hasattr(self, do) and callable(func := getattr(self, do)):
            return func(inputdata, outputdata, device), start, end, device
        else:
            raise NameError(f'Name {do} is not defined. You need to choose `computing_method="d"` or `computing_method="k"`')
    

    @exception_handler('error during computation of the MLQI²')
    def mlqi2_do_k(self, inputdata:torch.Tensor, 
            outputdata:torch.Tensor, device:int
    ) -> torch.Tensor:
        """function to compute the MLQI² with the k-nearest-neighbor dependant method

        The MLQI² describes the complexity of each example p with index i over all neighbourhoods k in a matrix.
        Each entry of the matrix is calculated via
        `mlqi²[i,k](P)= QI²(KNN_{re}(P,p_{i},k))`

        Here `QI²(P')` represents the calculation of the integrated quality indicator of the respective point set P' 
        over the normalised distances in the input and output space of the example pairs x using the formula
        `QI²R(P)=1/(|P²|)·∑_(x∈P²)(d_{NRE}(x)-d_{NRA}(x))²`

        Args:
            inputdata (torch.Tensor): sliced part of the inputdata with every neighborhood to every point in this slice
            outputdata (torch.Tensor): sliced part of the outputdata with every neighborhood to every point in this slice
            device (int): the device for the computation

        Returns:
            MLQI2 (torch.Tensor): computed slice of the MLQI²
        """
        mlqi2 = torch.zeros((inputdata.shape[0], self.max_neighbors), device=device, dtype=torch.float64)
        input_metric:metrics.Metric = metrics.metrics[self.input_metric](self.inputdata.size())
        output_metric:metrics.Metric = metrics.metrics[self.output_metric](self.outputdata.size())
        for i in range(inputdata.shape[0]):
            dRE = input_metric.distance(inputdata[i:i+1])
            dRA = output_metric.distance(outputdata[i:i+1])
            mlqi2[i] = utils.mlqi(mlqi2[i], dRE[0], dRA[0])
            torch.cuda.empty_cache()
        return mlqi2.detach().cpu()


    @exception_handler('error during computation of the MLQI²')
    def mlqi2_do_d(self, inputdata:torch.Tensor, 
            outputdata:torch.Tensor, device:int
    ) -> torch.Tensor:
        """function to compute the MLQI² with the distance dependant method

        The MLQI² describes the complexity of each example p with index i over all neighbourhoods containing datapoints
        within a given distance in a matrix.
        Each entry of the matrix is calculated via
        `mlqi²[i,k](P)= QI²(KNN_{re}(P,p_{i},k))`

        Here `QI²(P')` represents the calculation of the integrated quality indicator of the respective point set P' 
        over the normalised distances in the input and output space of the example pairs x using the formula
        `QI²R(P)=1/(|P²|)·∑_(x∈P²)(d_{NRE}(x)-d_{NRA}(x))²`

        Args:
            inputdata (torch.Tensor): sliced part of the inputdata with every neighborhood to every point in this slice
            outputdata (torch.Tensor): sliced part of the outputdata with every neighborhood to every point in this slice
            device (int): the device for the computation

        Returns:
            MLQI2 (torch.Tensor): computed slice of the MLQI²
        """
        mlqi2 = torch.zeros((inputdata.shape[0], self.max_neighbors), device=device, dtype=torch.float64)
        input_metric:metrics.Metric = metrics.metrics[self.input_metric](self.inputdata.size())
        output_metric:metrics.Metric = metrics.metrics[self.output_metric](self.outputdata.size())
        for i in range(inputdata.shape[0]):
            dRE = input_metric.distance(inputdata[i:i+1])
            dRA = output_metric.distance(outputdata[i:i+1])
            mlqi2[i] = utils.mlqi_dist(mlqi2[i], dRE[0], dRA[0], self.dist_samples)
            torch.cuda.empty_cache()
        return mlqi2.detach().cpu()


    @exception_handler('error during the computation of the HLQI²')
    def compute_hlqi2(self, minhl:float, maxhl:float,
                    MLQI2:torch.Tensor, 
                    BLQI2:torch.Tensor,  
                    start:int, end:int, device:int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int, int]:
        """function to compute a slice of the HLQI²

        The HLQI² serves as a visualisation of the MLQI² as a histogram, which over k assigns all local QI² from the MLQI² to a bin 
        :math:`v=0,1,2,...,[(max_{hi}-min_{hi})/binsize_{hi}]` 

        :math:`hlqi²[v,k](P)= ∑_(i=1)^(|P|)[I³(mlqi²[i,k](P),v)⋅blqi² [i,k](P)]`

        The function I³(h,v) returns a one if :math:`v≤(h-min_{hi})/binsize_{hi}<v+1` is satisfied, and a zero otherwise. 
        Thus, this function is used to check whether the complexity of an example p_{i} lies in a certain bin v. 
        Subsequently, the sum of all examples p_{i} is calculated. This process is repeated for each bin v. Finally, 
        the frequency distribution of all examples p_{i} on individual bins v over growing environments.

        Args:
            MLQI2 (torch.Tensor): slice of the MLQI²
            BLQI2 (torch.Tensor): slice of the BLQI²
            minhl (float): minimum value for a bin `v`
            maxhl (float): minimum value for a bin `v`
            start (int): index for the start of the computation in the current thread
            end (int): index for the end of the computation in the current thread
            device (int): the device for the computation

        Returns:
            HLQI2 (torch.Tensor): HLQI² for the given slice of the MLQI²
            assigned_values (torch.Tensor): assigned bins for each datapoint and every neighborhood that has been computed
            start (int): index for the start of the computation in the current thread
            end (int): index for the end of the computation in the current thread
            device (int): the device for the computation
        """
        MLQI2 = MLQI2.to(device=device)
        BLQI2 = BLQI2.to(device=device)
        hlqi2 = torch.zeros((int((maxhl - minhl) / self.binsizehl), MLQI2.shape[1]), dtype=torch.float64, device=device)
        assigned_values = torch.zeros_like(MLQI2, dtype=torch.int32, device=device)
        for v in range(int((maxhl - minhl) / self.binsizehl)):
            hlqi2[v], assigned_values = utils.hlqi(hlqi2[v], MLQI2, BLQI2, minhl, maxhl, self.binsizehl, assigned_values, v)
            torch.cuda.empty_cache()
        return hlqi2, assigned_values, start, end, device

    def get_histogram(self):
        self.result = (self.HLQI2/self.HLQI2.sum(axis=0, keepdims=True)).detach().cpu() ** self.gamma
