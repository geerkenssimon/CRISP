# built-in modules
import concurrent.futures
import datetime
import itertools
import logging
import time
from typing import Any, Callable, Dict, List

# third-party modules
import torch

# modules from current project
from CRISP.base._utils.data_utils import Points
from CRISP.base._utils._logging import log_without_format

if torch.cuda.is_available():
    cuda = True
    device_count = torch.cuda.device_count()
else:
    cuda = False    
    device_count = concurrent.futures.ThreadPoolExecutor()._max_workers


def multithreadded_execution(points:Dict[int, Points], fn:Callable, fn_args:List, fn_lambdas:List=[], returns:List[str]='', 
                              postprocess_fn:Callable=None, post_apply:str='', computational_stage:str='') -> Dict[str, Any]:
    
    try:
        devices = len(torch.unique(torch.Tensor([pt.device for pt in list(points.values())])))
    except:
        devices = 1

    device_uses = [0] * devices
    for device in range(devices):
        for d in points.values():
            if d.device == device:
                device_uses[device] += 1

    log_without_format(f'\n-------------------------------------------------------------------------------------\n' +\
                 f'starting {computational_stage} computation with {devices} device(s). \ndevice use:'+\
                '\n'.join([f'\t device {d}: {device_uses[d]} time(s)' for d in range(devices)]))

    it = iter(list(points.items()))
    results = []
    free_devices = []

    tasks_done = 0
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = {executor.submit(fn, *fn_args, 
                                 *[_fn_lambda(pt.start, pt.end, pt.device) for _fn_lambda in fn_lambdas],
                                 pt.start, pt.end, pt.device)
                                for _, pt in itertools.islice(it, devices)}
        while tasks:
            done, tasks = concurrent.futures.wait(tasks, return_when=concurrent.futures.FIRST_COMPLETED)
            for task in done:
                tasks_done += 1
                result_dict = dict([(key, value) for key, value in zip(returns, task.result())])
                if postprocess_fn is not None:
                    assert post_apply in returns, 'No or no known key to apply postprocessing to. \
                        Please choose a string from returns list to apply the postprocessing-function'
                    result_dict[post_apply] = postprocess_fn(result_dict[post_apply])
                
                results.append(result_dict)
                # last return from called function needs to be the device of the computation
                free_devices.append(task.result()[-1])

                logging.info(f'{computational_stage} computation of part {tasks_done} completed.'+\
                        f'... computational time for ' +\
                        f'{format(tasks_done/len(list(points.items())) * 100,"3.1f")}%' +\
                        f': {str(datetime.timedelta(seconds=time.time()-start_time))[:7]}')

            for _, pt in itertools.islice(it, len(done)):
                if pt.device in free_devices:
                    tasks.add(executor.submit(fn, *fn_args,
                                *[_fn_lambda(pt.start, pt.end, pt.device) for _fn_lambda in fn_lambdas],
                                 pt.start, pt.end, pt.device))
                    free_devices.remove(pt.device)
    torch.cuda.empty_cache()
    return results