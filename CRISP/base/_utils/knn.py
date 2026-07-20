# third party modules
import torch

# modules from current project
from CRISP.base._utils import metrics
from CRISP.base._utils.data_utils import distance_splitting
from CRISP.base._utils.multithreadding_utils import multithreadded_execution



class KNN:
    def __init__(self, m_in, m_out, kneighbors, devices_available):
        self.kneighbors = kneighbors
        self.input_metric:metrics.Metric = m_in
        self.output_metric:metrics.Metric = m_out
        self.devices_available = devices_available

    def get_knns(self, data:torch.Tensor) -> torch.Tensor:        
        pts, _, _ = distance_splitting(data, self.devices_available)

        results = multithreadded_execution(pts, fn=metrics._thread_per_device, fn_args=[self.input_metric, data], returns=['inds', 'start', 'end', 'device'],
                                  postprocess_fn=lambda a: a.topk(self.kneighbors, dim=1, largest=False)[1], post_apply='inds',
                                  computational_stage='KNN Distance')
        
        inds = torch.zeros(data.size(0), self.kneighbors, dtype=torch.int32)
        for result in results:
            inds[result['start']:result['end']] = result['inds']
        return inds.detach().cpu()
