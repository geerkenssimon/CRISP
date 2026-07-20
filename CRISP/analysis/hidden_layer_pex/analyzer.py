# built-in modules
from typing import Union

# third-party modules
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
from torchvision.transforms import Resize, InterpolationMode
import captum.attr._utils.visualization as V

# modules from current project
from CRISP.visualizing import Canvas
from CRISP.analysis._utils import _LRP
from CRISP.analysis.hidden_layer_pex.conv_utils.receptivefield import RFAnalyzer, NormalRFAnalyzer, SimpleRFAnalyzer

fire_red = LinearSegmentedColormap.from_list('mycmap', ['aliceblue', 'cyan', 'dodgerblue', 'blue', 'black', 'firebrick', 'coral', 'yellow', 'lightyellow'])

MODELANALYZER = {
    'Convolutional': {
        'normal': NormalRFAnalyzer,
        'simple': SimpleRFAnalyzer
    }
}

class ModelAnalyzer:
    def __init__(self, 
                 model:nn.Module,
                 model_class:str,
                 input_image:torch.Tensor,
                 layer:str,
                 stride_w:int = 1,
                 stride_h:int = 1,
                 visualization:bool = False,
                 canvas:Union[None, Canvas] = None,
                 fig:Union[None, Figure] = None,
                 axes:Union[None, Axes] = None,
                 gamma_bar:bool = False,
                 gamma:float = .5,
                 _add_callback:bool = False,
                 vis_method:str = 'blended_heat_map',
                 vis_sign:str = 'all',
                 padding:bool = False,
                 analysis:str = 'normal'
                 ) -> None:
        """Model analyzer class for layer based feature importance characterization and
        importance visualization.

        Parameters
        ----------
        model : nn.Module
            Model to be analyzed with feature importance characterization based on 
            interactive SHLQI² as a starting point for importance visualization with
            LRP
        model_class : str
            Class of the model. Options: 
                1. Convolutional
        input_image : torch.Tensor
            Inference image the analysis is based on 
        layer : str
            Layer to be analyzed
        stride_x, stride_< : int, optional
            Multiplicator for x and y direction strides. Has to be the same as in
            dataset generation. This is important for generating the target map inputted
            to the LRP attribution 
        visualization : bool, optional
            Whether to visualize the resulting heatmap from LRP attruibution, by default False
        canvas : Union[None, Canvas], optional
            Canvas object instantiated from `_utils.visualizing.Canvas`. This is only needed 
            for determining the interactively selected points from SHLQI², by default None
        fig : Union[None, Figure], optional
            `Matplotlib.Figure` object to draw the resulting heatmap in, by default None
        axes : Union[None, Axes], optional
            `Matplotlib.Axes` object to draw the resulting heatmap in, by default None
        gamma_bar : bool, optional
            Whether to add a bar for gamma value changes affecting the appearence of the 
            resulting heatmap, by default False
        gamma : float, optional
            Initial gamma value, by default .5
        _add_callback : bool, optional
            Whether to add a callback to the Canvas object. By default the callback is called 
            after computing the points selected interactively in the canvas either by polygon
            or rectangle selection, by default False
        vis_method : str, optional
            Method for heatmap visualization with `captum.attr._utils.visualization`. Options:
                1. `heat_map` - Display heat map of chosen attributions

                2. `blended_heat_map` - Overlay heat map over greyscale version of original 
                    image. Parameter alpha_overlay corresponds to alpha of heat map.

                3. `original_image` - Only display original image.

                4. `masked_image` - Mask image (pixel-wise multiply) by normalized attribution 
                    values.

                5. `alpha_scaling` - Sets alpha channel of each pixel to be equal to normalized 
                    attribution value.
            by default 'blended_heat_map'
        vis_sign : str, optional
            Chosen sign of attributions to visualize with `captum.attr._utils.visualization`. 
            Options:
                1. `positive` - Displays only positive pixel attributions.

                2. `absolute_value` - Displays absolute value of
                    attributions.

                3. `negative` - Displays only negative pixel attributions.

                4. `all` - Displays both positive and negative attribution  values. This is not 
                    supported for `masked_image` or `alpha_scaling` modes, since signed 
                    information cannot be represented in these modes.
            by default 'all'
        """
        assert type(input_image) == torch.Tensor, \
            f'Input image type {(type(input_image))} is not a tensor {type(torch.Tensor)}. Aborting ...'
        
        assert isinstance(model, nn.Module), f'Given model ({type(model)}) does not inheret from nn.Module. \
            The classes it inherits from are {type(model).__mro__}. Aborting ...'
        
        self.device = list(model.parameters())[0].device
        self.input = input_image
        self.canvas = canvas
        self.gamma = gamma
        self.vis_sign = vis_sign
        self.vis_method = vis_method
        self.stride_mul_w = stride_w
        self.stride_mul_h = stride_h

        if _add_callback:
            assert canvas is not None, f'No Canvas to draw in specified, but tried to add a callback to it.'
            self.canvas.callback = self.attribution

        pixel_values = (input_image.min().item(), input_image.max().item())
        self.LRP = _LRP(model, pixel_values)

        self.layer = self.LRP._search_layer(model, layer)

        self.output_shape = self.LRP._intermediate_layer_output_shape(self.layer, self.input)
        
        self.LRP.forward_with_hooks(layer=self.layer, inputs=self.input)

        input_index = int(self.canvas.QI.dataset_name.split('.')[0].split('_')[-1])

        analyzer:RFAnalyzer = MODELANALYZER[model_class][analysis](self.layer, padding)
        inputs = self.layer.activations[self.device]
        xs, ys, rfs, efs, i_strides, o_strides = analyzer.analyze(inputs)

        self.x = xs[input_index]
        self.y = ys[input_index]
        self.rf = rfs[input_index]
        self.ef = efs[input_index]
        self.i_stride = i_strides[input_index]
        self.o_stride = o_strides[input_index]

        self.i_stride *= self.stride_mul_w
        self.o_stride *= self.stride_mul_h

        if visualization:
            if fig is not None:
                self.figure = fig
                self.ax = axes
            else:
                self.figure, self.ax = plt.subplots(1,1, figsize=self.canvas.fig.get_size_inches(), 
                                                    layout=self.canvas.fig.get_layout_engine())

            if gamma_bar:
                self.figure.subplots_adjust(bottom=.1, wspace=.2, hspace=.2)
                self.ax_slider=self.figure.add_axes([0.1, 0., 0.8, 0.03])
                self.slider = Slider(
                    ax=self.ax_slider,
                    label='Gamma',
                    valmin=0.,
                    valmax=1.,
                    valinit=0.5,
                )
                self.slider.on_changed(self._visualize_heatmap)
        self.figure.set_visible(False)

    def _create_target_map(self, indices:Union[None, torch.Tensor]):
        if indices is None:
            indices = self.canvas.points
        _o = torch.zeros(self.y.size())

        _o = _o.unfold(0,self.ef,self.o_stride).unfold(1,self.ef,self.o_stride)
        shape = _o.size()[:2]
        
        inds = torch.unravel_index(indices, shape)

        attentionmap = torch.zeros(shape, dtype=torch.bool).to(self.device)
        attentionmap[inds] = True
        size = self.y.size()
        r = Resize(size[:2], interpolation=InterpolationMode.NEAREST)
        attentionmap:torch.Tensor = r(attentionmap.view(1,1,shape[0],shape[1]))
        
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        attentionmap_ = torch.zeros(self.output_shape, dtype=torch.bool).to(self.device)
        
        start_2 = abs(attentionmap_.size(2) - attentionmap.size(2)) // 2
        start_3 = abs(attentionmap_.size(3) - attentionmap.size(3)) // 2
        
        attentionmap_[:, :, start_2:start_2+attentionmap.size(2), start_3:start_3+attentionmap.size(3)] = attentionmap
                
        return attentionmap_

    def attribution(self, indices:Union[None, torch.Tensor]=None):
        target = self._create_target_map(indices)
        with torch.autograd.set_detect_anomaly(True):
            self.R = self.LRP.attribute(self.input, target=target, layer=self.layer)[0]
        self._visualize_heatmap(gamma=self.gamma)

    def _update_heatmap(self, gamma:float):
        hm = self.R.detach().permute(1,2,0)
        hm /= self.R.abs().max()

        hrp  = (hm-0.00).clip(0,0.25)/0.25
        hgp = (hm-0.25).clip(0,0.25)/0.25
        hbp = (hm-0.50).clip(0,0.50)/0.50

        hbn = -(-hm-0.00).clip(0,0.25)/0.25
        hgn = -(-hm-0.25).clip(0,0.25)/0.25
        hrn = -(-hm-0.50).clip(0,0.50)/0.50

        hm = torch.concatenate([(hrp+hrn)[...,None],(hgp+hgn)[...,None],(hbp+hbn)[...,None]],axis = 2)
        hm_sign = hm.sign()
        hm_abs = hm.abs()
        hm_abs = hm_abs ** gamma

        return hm_abs * hm_sign

    def _visualize_heatmap(self, gamma:Union[None, float]=None):
        if gamma is None:
            gamma = self.gamma

        hm = self._update_heatmap(gamma).detach().cpu().numpy()
        self.figure.set_visible(True)

        image = self.input[0].permute(1,2,0).detach().cpu().numpy()

        V.visualize_image_attr(hm, original_image=image, 
                                title='Heatmap for marked area', 
                                cmap=fire_red, method=self.vis_method, sign=self.vis_sign,
                                plt_fig_axis=(self.figure, self.ax), alpha_overlay=0.75,
                                use_pyplot=False)
        self.figure.canvas.draw()

    

