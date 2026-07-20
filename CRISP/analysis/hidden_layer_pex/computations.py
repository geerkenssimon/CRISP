# build-in modules
import os
import concurrent.futures
import time
import warnings
from typing import Any, Callable, Dict, List, Tuple, Union

# third-party modules
import cv2
import torch
import torch.nn as nn

# modules from current project
from CRISP.base import SHLQI2
from CRISP.analysis._utils import _LRP
from CRISP.analysis.hidden_layer_pex.conv_utils.data_extraction import _dataset
from CRISP.tmp import save, load

if torch.cuda.is_available():
    cuda = True
    device_count = torch.cuda.device_count()
else:
    cuda = False     
    device_count = concurrent.futures.ThreadPoolExecutor()._max_workers

def _per_layer_SHLQI2(model:nn.Module = None,
                    fn_generate_model:Callable = None,
                    _fn_generate_model_args:Dict = {},
                    _fn_preprocess:Callable = lambda x: torch.tensor(x,dtype=torch.float).permute(2,0,1),
                    _fn_preprocess_args:Dict = {},
                    im_size:Tuple = (3,640,320),
                    im_range:Tuple = (0,256),
                    _fn_read_im:Callable = cv2.imread,
                    _fn_read_im_args:Dict = {},
                    stride_x_mul:Union[List[int], int] = 1,
                    stride_y_mul:Union[List[int], int] = 1,
                    _dataset_generation_fn:Callable = None,
                    data_location:str = 'data',
                    save_location:str = 'save',
                    layer_type:Union[nn.Module, Any] = nn.Conv2d,
                    _not_contain:Union[None, str] = None,
                    model_device:str = 'cuda:0',
                    computation_devices:List = list(range(device_count)),
                    keep_dataset:bool = True,
                    analysis:str = 'normal',
                    max_neighbors:Union[None, int] = None,
                    distance_samples: Union[None, int] = None,
                    computing_method:str = 'k', 
                    max_dist: float = 0,
                    logarithmic:bool = False,
                    input_metric:str = 'Euclid', 
                    output_metric:str = 'Euclid',
                    verbose:bool = False,
                    verbose_lrp:bool = False,
                    padding:bool = False
                    ):
    """function for computing the SHLQI² per layer in a given network.

    Parameters
    ----------
    model : nn.Module, optional
        Model for iteration, by default None
    fn_generate_model : Callable, optional
        Function for generating a model, by default None
    _fn_generate_model_args : Dict, optional
        Arguments given to the previous function, by default {}
    _fn_preprocess : Callable, optional
        Function for preprocessing infered data, by default None
    _fn_preprocess_args : Dict, optional
        Arguments for previous function, by default {}
    im_size : Tuple, optional
        Size of the image. Only needed for maximum and minimum pixel values
        in LRP zB rules, by default (3,640,320)
    im_range : Tuple, optional
        Range of the possible values of the image before preprocessing. 
        Only needed for maximum and minimum pixel values in LRP zB rule,
        by default (0,255)
    _fn_read_im : Callable, optional
        Function for reading the images from a path. The path is given from
        a for loop iterating over every file in a given path, by default cv2.imread
    _fn_read_im_args : Dict, optional
        Arguments for previous function, by default {}
    stride_x_mul, stride_y_mul : int, optional
        Multiplicational factor for striding over layer input. This can be set to
        values > 1 if the featuremap is pretty big to reduce datapoints generated
        for SHLQI². If set to values > 1, this will determine a new stride over 
        input and output of the layer, by default 1
    _dataset_generation_fn : Callable, optional
        Custom function for input and output aggregation from the layers input and 
        output. The normal way of aggregating input and output data for SHLQI² computation
        is unfolding the input by `receptive_field x receptive_field` squares and the 
        output by `1 x 1` squares, depending on padding, stride, and the receptive field.
        The custom function has to return two `torch.Tensor` elements and gets the arguments
            layer : nn.Module, 
            stride : torch.Tensor, 
            padding : torch.Tensor, 
            receptive_field : torch.Tensor, 
            stride_mul : int, 
            save_dest : str, 
            name : str, 
            analysis : str
        by default None
    data_location : str, optional
        Patch to images to be infered and layer-wise computed with SHLQI², by default 'data'
    save_location : str, optional
        Destination to intermediately store the layer-wise datasets, by default 'save'
    layer_type : Union[nn.Module, Any], optional
        Type of the layers to be analyzed. This can be a `nn.Module` from PyTorch or any custom
        module that the Model contains, by default nn.Conv2d
    model_device : str, optional
        Device for Model inference, by default 'cpu'
    computation_devices : List, optional
        Device(s) for SHLQI² computation, by default list(range(device_count))
    keep_dataset : bool, optional
        Wether to keep saved datasets or just the SHLQI² analyses, by default True
    analysis : str, optional
        Type of analysis. 
        Supported options are:
            1. `normal` - considering the normal input and output patch size computation
            2. `simple` - considering a simple input and output patch computation
        by default 'normal'
    max_neighbors : Union[None, int], optional
        Parameter for SHLQI², default is None
    distance_samples : Union[None, int], optional
        Parameter for SHLQI², default is None
    computing_method : str, optional
        Parameter for SHLQI², default is 'k'
    max_dist : float, optional
        Parameter for SHLQI², default is 0
    logarithmic : bool, optional
        Parameter for SHLQI², default is False
    input_metric : str, optional
        Parameter for SHLQI², default is 'Euclid',
    output_metric : str, optional
        Parameter for SHLQI², default is 'Euclid'
    """
    model_device = torch.device(model_device)

    # creating model based on given model or generator function
    if model is None:
        assert fn_generate_model is not None, 'No callable function for model generation and no model given. Aborting ...'
        model:nn.Module = fn_generate_model(**_fn_generate_model_args)
    model = model.to(model_device)

    # checking whether there are layers of the specific type to be computed.
    assert any([type(child) == layer_type for child in model.modules()]), f'No layer or layer group of ' + \
        f'type {layer_type} found. Aborting ...'
    
    # initializing the LRP object with the model and pixel values
    pixel_values = (_fn_preprocess(torch.ones(im_size).numpy() * im_range[0], **_fn_generate_model_args).min().item(), 
                    _fn_preprocess(torch.ones(im_size).numpy() * im_range[1], **_fn_generate_model_args).max().item())
    lrp = _LRP(model, pixel_values)

    if len(computation_devices) > device_count:
        warnings.warn(f'less devices in machine ({device_count}) \
        than specified to use ({len(computation_devices)}). Using all devices')
        computation_devices = list(range(device_count))

    # checking the type of stride multiplicators   
    assert type(stride_x_mul) == type(stride_y_mul), f'Types of stride multiplicates not equal. \
        Type of stride_x_mul: {type(stride_x_mul)}, type of stride_y_mul {type(stride_y_mul)}.'
    
    # checking the type of stride multiplicators
    assert type(stride_x_mul) == int or type(stride_x_mul) == list, f'Types of stride multiplicators not \
        valid. Types ({type(stride_x_mul)},{type(stride_y_mul)}) must be either `int` or `list`.'
    
    # checking the length of stride multiplicators if they are lists. 
    # They must be of the same length and the same length as the number of layers
    if type(stride_x_mul) == list:
        assert len(stride_x_mul) == len(stride_y_mul), f'Length of stride multiplicators does not match. \
            {len(stride_x_mul)}, {len(stride_y_mul)}'
        assert len(stride_x_mul) == len([module for module in lrp.model.modules() if type(module) == layer_type]), \
        f'{len(stride_x_mul)}, {len([module for module in lrp.model.modules() if type(module) == layer_type])}'

    # for every image in the specified path
    #   first all layers data will be stored as datasets ready for the computation of SHLQI²
    #   second every layer-wise SHLQI² is computed
    for root, dirs, files in os.walk(data_location):
        for img_path in files:
            if verbose:
                print(img_path)

            if img_path.split('.')[-1] not in ['jpg', 'jpeg', 'png']:
                continue

            model.to(model_device)
            lrp = _LRP(model, pixel_values)

            # setting up the save location
            destination = os.path.join(save_location, '.'.join(img_path.split('.')[:-1]), analysis)
            os.makedirs(destination, exist_ok=True)
            os.makedirs(os.path.join(destination, 'computed'), exist_ok=True)

            # reading and preprocessing the image
            x = _fn_read_im(os.path.join(data_location, root, img_path), **_fn_read_im_args)
            x = _fn_preprocess(x, **_fn_preprocess_args).to(model_device)
            #x = x.requires_grad_(True)
            
            # for every layer of the specified type the computation of SHLQI² is done
            for i, layer in enumerate([module for module in lrp.model.modules() if type(module) == layer_type]):

                layer_name = lrp.layer_names[lrp.layers.index(layer)]
                if _not_contain is not None and any([_nc in layer_name for _nc in _not_contain]):
                    continue 

                # attribution of the network for getting input and output of the layer
                lrp.forward_with_hooks(x, layer=layer, _restore_model=False, verbose=verbose_lrp)

                if verbose:
                    print(layer.name)#, layer.outputs[model_device][0].size())
                    print(layer)

                # calculating receptive field of the whole module with every child 
                # and get the padding of the first childs computational layer
                #analyzer = ReceptiveFieldAnalyzer(layer)
                #rfs, efs, i_strides, o_strides = analyzer.analyze(layer.activations[model_device])
                #rs, strides, paddings, _, _ = get_attributes(layer, model_device, padding)
                
                # setting the stride multiplicators
                if type(stride_x_mul) == list:
                    _stride_mul = stride_x_mul[i]
                else:
                    _stride_mul = stride_x_mul
                
                # generating the dataset with either a custom function or a predefined
                # one based on the dimension of the layers parameters
                # the custom function needs to be able to receive the variables
                # (layer, s, padding, r,  _stride_mul, destination,  layer_name, analysis, model_device)
                if _dataset_generation_fn is not None:
                    generator_fn = _dataset_generation_fn
                else:
                    generator_fn = _dataset
                
                generator_fn(layer, _stride_mul, destination, 
                            layer_name, analysis, padding)
                lrp._restore_model()
                if verbose_lrp:
                    print('\n' + '-' * 50)
                    print('Press Enter to continue ...')
                    print('\n' + '-' * 50)
                    input('\n')
            if verbose:
                print('\n' + '-' * 50)
                print('Press Enter to continue ...')
                print('\n' + '-' * 50)
                input('\n')

            # memory management
            model.to('cpu')
            del x
            del lrp
            torch.cuda.empty_cache()
            
            for data_path in os.listdir(destination):
                # dont recompute SHLQI² for already computed layers
                if 'SHLQI2_' + data_path in os.listdir(os.path.join(destination, 'computed')) or os.path.isdir(os.path.join(destination, data_path)):
                    continue

                # intializing SHLQI²
                shlqi = SHLQI2(max_neighbors=max_neighbors,
                        dataset_name=data_path,
                        distance_samples=distance_samples,
                        computing_method=computing_method,
                        max_dist=max_dist,
                        logarithmic=logarithmic,
                        input_metric=input_metric,
                        output_metric=output_metric,
                        devices_available=computation_devices)
                layer_name = '_'.join(data_path.split('_')[:-2])
                
                if not layer_name in data_path:
                    continue

                # loading data with inputs and outputs
                data = load(os.path.join(destination, data_path))
                if not keep_dataset:
                    os.remove(os.path.join(destination, data_path))
                inputs = data[layer_name + '_in']
                outputs = data[layer_name + '_out']

                # starting SHLQI²
                shlqi.start(inputs, outputs)
                
                # saving the result
                _destination = os.path.join(destination, 'computed', 'SHLQI2_'+data_path)
                save(shlqi, _destination, sparse=True)

                # memory management
                del shlqi
                torch.cuda.empty_cache()

