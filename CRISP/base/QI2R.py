# build-in modules
import concurrent.futures
import logging
import math
from typing import List

# third-party modules
import torch

# modules from current project
from CRISP.base._utils import qi_utils as utils
from CRISP.base._utils import metrics
from CRISP.base._utils._logging import start_logging, log_without_format, exception_handler
from CRISP.base._utils.data_utils import distance_splitting, splitting
from CRISP.base._utils.multithreadding_utils import multithreadded_execution

if torch.cuda.is_available():
    cuda = True
    overall_device = 'cuda:0'
    device_count = torch.cuda.device_count()
else:
    cuda = False  
    overall_device = 'cpu'      
    device_count = concurrent.futures.ThreadPoolExecutor()._max_workers

class QI2R_old:
    def __init__(self, 
                requires_grad:bool=False,
                dataset_name:str='None',
                input_metric:str='Euclid',
                output_metric:str='Euclid',
                devices:List = [0]
    ) -> None:
        """QI²R class constructor.

        This class contains every necessary method for computing the QI²R

        Parameters
        ----------
        inputdata : torch.Tensor
            Inputdata for QI²R computation.
        outputdata : torch.Tensor
            Outputdata for QI²R computation.
        requires_grad : bool, optional
            Boolean argument determining wether the inputdata requires gradient 
            computation or not. Default is False
        dataset_name : str, optional
            Name of the dataset (only for logging). Default is 'None'.
        input_metric : str, optional
            Name of the metric for input distance computation. Default is 'Euclid'.
        output_metric : str, optional
            Name of the metric for output distance computation. Default is 'Euclid'.

        Examples
        --------
        >>> _QI2R = QI2R(torch.random((10,3)), torch.random((10,1)))
        >>> _QI2R.start()
        >>> print(_QI2R.QI2R)
        0.523 (may vary)
        """
        self.dataset_name = dataset_name
        self.requires_grad = requires_grad
        self.devices_available = devices

        self.input_metric:metrics.Metric = metrics.metrics[input_metric]
        self.output_metric:metrics.Metric = metrics.metrics[output_metric]

    def norm(self, data:torch.Tensor) -> torch.Tensor:
        data -= data.min(0,keepdims=True)[0]
        data /= (data.max(0,keepdims=True)[0] + 1e-16)
        return data

    def normalization_avg(self, data:torch.Tensor) -> torch.Tensor:
        normalized_data = utils.normalize_avg(data)
        return normalized_data
    
    def distance(self, data:torch.Tensor, start, end, device) -> torch.Tensor:
        return torch.cdist(data, data).float(), start, end, device
    
    @exception_handler('error in over all computation')
    def start(self, 
            inputdata:torch.Tensor, 
            outputdata:torch.Tensor):
        start_logging("QI²R")
        logging.info('starting computation of QI²R ...')
        logging.info('Parameters:')
        log_without_format('Name:                   ' + self.dataset_name)
        log_without_format('Datapoints:             ' + str(inputdata.shape[0]))
        log_without_format('Input dimensions:       ' + str(inputdata.shape[1]))
        log_without_format('Output dimensions:      ' + str(outputdata.shape[1]))
        log_without_format('GPU accelerated:        ' + str(cuda))
        
        self.inputdata = inputdata
        self.outputdata = outputdata

        pts, _, _ = distance_splitting(self.inputdata, self.devices_available)
        #dRE, _,_,_ = self.distance(self.inputdata, 0,0,0)
        results = multithreadded_execution(pts, self._thread_per_device, 
                                            [self, self.inputdata],
                                            returns=['dRE', 'start', 'end', 'device'])
        
        dRE = torch.zeros(self.inputdata.size(0), self.inputdata.size(0), dtype=torch.float32)
        for result in results:
            dRE[result['start']:result['end']] = result['dRE']
        
        pts, _, _ = distance_splitting(self.outputdata, self.devices_available)
        results = multithreadded_execution(pts, metrics._thread_per_device, 
                                            [self.output_metric, self.outputdata],
                                            returns=['dRA', 'start', 'end', 'device'])
        dRA = torch.zeros(self.outputdata.size(0), self.outputdata.size(0), dtype=torch.float32)
        for result in results:
            dRA[result['start']:result['end']] = result['dRA']
        
        dNRE = self.normalization_avg(dRE)
        dNRA = self.normalization_avg(dRA)
        self.result = 1/(self.inputdata.size(0)**2) * ((dNRE-dNRA)**2).sum()
        self.result.requires_grad_(True)




class QI2R:
    def __init__(self, 
                requires_grad:bool=False,
                dataset_name:str='None',
                input_metric:str='Euclid',
                output_metric:str='Euclid',
                devices:List = list(range(device_count))
    ) -> None:
        """QI²R class constructor.

        This class contains every necessary method for computing the QI²R

        Parameters
        ----------
        inputdata : torch.Tensor
            Inputdata for QI²R computation.
        outputdata : torch.Tensor
            Outputdata for QI²R computation.
        requires_grad : bool, optional
            Boolean argument determining wether the inputdata requires gradient 
            computation or not. Default is False
        dataset_name : str, optional
            Name of the dataset (only for logging). Default is 'None'.
        input_metric : str, optional
            Name of the metric for input distance computation. Default is 'Euclid'.
        output_metric : str, optional
            Name of the metric for output distance computation. Default is 'Euclid'.

        Examples
        --------
        >>> _QI2R = QI2R(torch.random((10,3)), torch.random((10,1)))
        >>> _QI2R.start()
        >>> print(_QI2R.QI2R)
        0.523 (may vary)
        """
        self.dataset_name = dataset_name
        self.requires_grad = requires_grad
        self.devices_available = devices

        self.input_metric:metrics.Metric = metrics.metrics[input_metric]
        self.output_metric:metrics.Metric = metrics.metrics[output_metric]

    def norm(self, data:torch.Tensor) -> torch.Tensor:
        data -= data.min(0,keepdims=True)[0]
        data /= (data.max(0,keepdims=True)[0] + 1e-16)
        return data

    def normalization_avg(self, data:torch.Tensor) -> torch.Tensor:
        normalized_data = utils.normalize_avg(data)
        return normalized_data
    
    def distance(self, data:torch.Tensor, secondary_data:torch.Tensor=None) -> torch.Tensor:
        if secondary_data is None:
            return torch.cdist(data, data).float()
        else:
            return torch.cdist(data, secondary_data).float()
        
    def _thread_per_device(self, start:int, end:int, device:int):
        sec_chunk_size = end-start
        sec_start = 0
        
        indistances = torch.zeros((end-start, self.inputdata.size(0)), dtype=torch.float32, device=device)
        outdistances = torch.zeros((end-start, self.outputdata.size(0)), dtype=torch.float32, device=device)
        
        for i in range(math.ceil(self.inputdata.size(0)/sec_chunk_size)):
            indistances[:,sec_start:sec_start+sec_chunk_size] = self.distance(self.inputdata[start:end].to(device), 
                                                    self.inputdata[sec_start:sec_start+sec_chunk_size].to(device))
            sec_start += sec_chunk_size 

        torch.cuda.empty_cache()
        sec_chunk_size = end-start
        sec_start = 0
        for i in range(math.ceil(self.outputdata.size(0)/sec_chunk_size)):
            with torch.no_grad():
                outdistances[:,sec_start:sec_start+sec_chunk_size] = self.distance(self.outputdata[start:end].to(device), 
                                                        self.outputdata[sec_start:sec_start+sec_chunk_size].to(device))
                sec_start += sec_chunk_size 
        
        torch.cuda.empty_cache()
        in_d = indistances.sum().cpu()
        out_d = outdistances.sum().cpu()
        in_s = (indistances**2).sum().cpu()
        out_s = (outdistances**2).sum().cpu()
        #inout = (indistances * outdistances).sum().cpu()
        inout = (indistances * outdistances).sum().cpu()
        return in_d, out_d, in_s, out_s, inout, start, end, device
        return indistances.sum(), outdistances.sum(), \
                (indistances**2).sum(), (outdistances**2).sum(), \
                (indistances * outdistances).sum(), \
                start, end, device
    
    @exception_handler('error in over all computation')
    def start(self, 
            inputdata:torch.Tensor, 
            outputdata:torch.Tensor):
        start_logging("QI²R")
        logging.info('starting computation of QI²R ...')
        logging.info('Parameters:')
        log_without_format('Name:                   ' + self.dataset_name)
        log_without_format('Datapoints:             ' + str(inputdata.shape[0]))
        log_without_format('Input dimensions:       ' + str(inputdata.shape[1]))
        log_without_format('Output dimensions:      ' + str(outputdata.shape[1]))
        log_without_format('GPU accelerated:        ' + str(cuda))
        
        self.inputdata = inputdata / (inputdata.size(0) * inputdata.size(1))
        self.outputdata = outputdata / (inputdata.size(0) * inputdata.size(1))
        if self.requires_grad:
            self.inputdata.requires_grad_(True)
            #self.inputdata.retain_grad()

        import psutil
        cpu_mem = psutil.virtual_memory()
        cpu_mem = [cpu_mem.available / len(self.devices_available) for device in self.devices_available ]
        cpu_ppd = torch.tensor([torch.max(-((torch.prod(torch.tensor(self.inputdata.size())) + torch.prod(torch.tensor(self.outputdata.size())))/2)/2 + \
                                          ((((torch.prod(torch.tensor(self.inputdata.size())) + torch.prod(torch.tensor(self.outputdata.size())))/2)/2)**2 + (mem*32*8)) ** (1/2),
                                          -((torch.prod(torch.tensor(self.inputdata.size())) + torch.prod(torch.tensor(self.outputdata.size())))/2)/2 - \
                                          ((((torch.prod(torch.tensor(self.inputdata.size())) + torch.prod(torch.tensor(self.outputdata.size())))/2)/2)**2 + (mem*32*8)) ** (1/2) 
        ) for mem in cpu_mem])

        free_mem = [(torch.cuda.mem_get_info(device)[0]*0.5) for device in self.devices_available]
        if not self.requires_grad:
            ppd = torch.tensor([torch.max(torch.tensor(-self.inputdata.size(1) + (((self.inputdata.size(1))**2)+(mem*2/32))**(1/2)), 
                            torch.tensor(-self.inputdata.size(1)/2 - (((self.inputdata.size(1))**2)+(mem*2/32))**(1/2))) for mem in free_mem])
        else:
            ppd = torch.tensor([(mem*2/32) / (2*(self.inputdata.size(0) + self.inputdata.size(1) + self.outputdata.size(1))) for mem in free_mem])
        
        ppd = torch.minimum(ppd, cpu_ppd)
        pts = splitting(ppd, self.inputdata.size(0), self.devices_available)

        dRE_sum = 0
        dRA_sum = 0
        dRE_square = 0
        dRA_square = 0
        dREdRA_sum = 0
        results = multithreadded_execution(pts, self._thread_per_device, [],
                                            returns=['dRE_sum', 'dRA_sum', 'dRE_square', 'dRA_square',
                                                      'dREdRA_sum', 'start', 'end', 'device'])
        
        p_square = self.inputdata.size(0) ** 2

        for result in results:
            dRE_sum += result['dRE_sum']
            dRA_sum += result['dRA_sum']

            dRE_square += result['dRE_square']
            dRA_square += result['dRA_square']

            dREdRA_sum += result['dREdRA_sum']

        f1 = dRE_square
        c1 = dRE_sum ** 2

        f2 = dRA_square
        c2 = dRA_sum ** 2

        f3 = dREdRA_sum
        c3 = dRE_sum * dRA_sum

        self.result = p_square * 1/(c1*c2*c3) * (f1*c2*c3 + f2*c1*c3 - 2*f3*c1*c2)
        torch.cuda.empty_cache()


class VQI2R:
    def __init__(self, DQI2R:torch.Tensor):
        """VQI²R class construtor

        Parameters
        ----------
        DQI2R : torch.Tensor
            computed DQI²R
        """
        self.DQI2R = DQI2R

    @exception_handler('error in over all computation')
    def start(self):
        start_logging("VQI²R")
        logging.info('starting computation of VQI²R ...')
        logging.info('Parameters:')
        log_without_format('Length:                 ' + str(len(self.DQI2R)))
        self.result = torch.zeros_like(self.DQI2R, device=overall_device)
        for k in range(len(self.DQI2R)):
            self.result[k] = self.DQI2R[:k+1].sum()

class DQI2R(QI2R):
    def __init__(self, 
                norm:bool=False,
                dataset_name:str='None',
                input_metric:str = 'Euclid', 
                output_metric:str = 'Euclid',
                devices:List = [0],
                per_sample:bool = False):
        """DQI²R class constructor

        Parameters
        ----------
        inputdata : torch.Tensor
            Inputdata for the computation
        outputdata : torch.Tensor
            Outputdata for the computation
        norm : bool, optional
            Whether to norm the data before computation. Default is False.
        dataset_name : str, optional
            Dataset name (only for logging). Default is 'None'.
        input_metric : str, optional
            Name of the metric for input distance computation. Default is 'Euclid'.
        output_metric : str, optional
            Name of the metric for output distance computation. Default is 'Euclid'.
        """
        super().__init__(norm,
                dataset_name,
                input_metric, 
                output_metric,
                devices)
        self.per_sample = per_sample

    def total_distances(self, data:torch.Tensor) -> torch.Tensor:
        if not self.per_sample:
            return data.sum()/data.size(0)**2
        else:
            return data.sum(dim=1)/data.size(0)**2

    @exception_handler('error in over all computation')
    def start(self,
            inputdata:torch.Tensor, 
            outputdata:torch.Tensor):
        start_logging("DQI²R")
        logging.info('starting computation of DQI²R ...')
        logging.info('Parameters:')
        log_without_format('Name:                   ' + self.dataset_name)
        log_without_format('Datapoints:             ' + str(inputdata.shape[0]))
        log_without_format('Input dimensions:       ' + str(inputdata.shape[1]))
        log_without_format('Output dimensions:      ' + str(outputdata.shape[1]))
        log_without_format('GPU accelerated:        ' + str(cuda))

        self.inputdata = inputdata
        self.outputdata = outputdata

        pts, _, _ = distance_splitting(self.inputdata, self.devices_available)
        results = multithreadded_execution(pts, metrics._thread_per_device, 
                                            [self.input_metric, self.inputdata],
                                            returns=['dRE', 'start', 'end', 'device'])
        dRE = results[0]['dRE']
        
        pts, _, _ = distance_splitting(self.outputdata, self.devices_available)
        results = multithreadded_execution(pts, metrics._thread_per_device, 
                                            [self.input_metric, self.inputdata],
                                            returns=['dRA', 'start', 'end', 'device'])
        dRA = results[0]['dRA']

        TOTDE = self.total_distances(dRE)
        TOTDA = self.total_distances(dRA)

        indices = torch.argsort(dRE)

        self.sdRE = torch.zeros_like(dRE, device=dRE.device)
        self.sdRA = torch.zeros_like(dRA, device=dRA.device)
        
        for i in range(len(indices)):
            for j in range(len(indices[i])):
                self.sdRE[i,j] = (dRE[i,indices[i,j]])
                self.sdRA[i,j] = (dRA[i,indices[i,j]])
        del dRE
        del dRA

        if not self.per_sample:
            self.result = torch.zeros(len(self.sdRE), device=self.sdRE.device)
            for k in range(self.sdRE.shape[1]):
                self.result[k] = 1/len(self.sdRE)**2 * ((self.sdRE[:,k]/TOTDE - self.sdRA[:,k]/TOTDA)**2).sum()
        else:
            self.result = torch.zeros_like(self.sdRE, device=self.sdRE.device)
            for k in range(self.sdRE.shape[1]):
                self.result[:,k] = 1/len(self.sdRE)**2 * ((self.sdRE[:,k]/TOTDE - self.sdRA[:,k]/TOTDA)**2)
