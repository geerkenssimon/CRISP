from pathlib import Path
import argparse
from argparse import RawTextHelpFormatter

from matplotlib import pyplot as plt
import matplotlib
import cv2, os, sys

sys.path.append(os.getcwd())

from CRISP.visualizing import Canvas
from CRISP.tmp import load
from CRISP.tests.UNet import model_utils
from CRISP.base import SHLQI2
from CRISP.analysis.hidden_layer_pex.analyzer import ModelAnalyzer


gamma = 0.2

factor = 2
textwidth = 7 * factor
golden_ratio = ((5**.5 -1.1)/2)
golden_ratio_heatmap = ((5**.5 -1.1)/2)

fig_w = textwidth
fig_h = fig_w * golden_ratio

heatmap_w = textwidth
heatmap_h = heatmap_w * golden_ratio_heatmap

def _update_rcParams(factor = factor):
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.serif': 'Times Roman',
        'font.size': 10 * factor,
        'axes.labelsize': 10 * factor,
        'axes.titlesize': 10 * factor,
        'text.usetex': True,
        'pgf.rcfonts': True,
    })
_update_rcParams()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='UNet_nn_analysis.py',
                                     usage='python UNet_nn_analysis.py --IMAGE <Image_name> --LAYER <Layer_name> --stride_x <stride_x> --stride_y <stride_y>',
                                     description='feature importance analysis with SHLQI² and LRP of the U-Net.', formatter_class=RawTextHelpFormatter
                                     )
    parser.add_argument('--IMAGE', type=str, default='rs00004.jpg', 
                        help='Name of the image to be analyzed. Example files only available for\n' \
                        '   - OSDAR_017_0924.png\n' \
                        '   - rs00018.jpg\n' \
                        '   - rs00420.jpg\n' \
                        '   - rs00004.jpg\n' \
                        '   - rs000506.jpg')
    parser.add_argument('--LAYER', type=str, default='down1-maxpool_conv-1-conv-Conv2d0',
                        help='Name of the layer to be analyzed. Example files only available for\n' \
                        '   - down1-maxpool_conv-1-conv-Conv2d0\n' \
                        '   - down3-maxpool_conv-1-conv-Conv2d0\n' \
                        '   - down4-maxpool_conv-1-conv-Conv2d0\n' \
                        '   - Up1\n' \
                        '   - up2-conv-conv-Conv2d0\n' \
                        '   - up3-conv-conv-Conv2d1',)
    parser.add_argument('--INPUT', type=int, default=0, help='Index of the input to be analyzed. only important if the module has multiple inputs (e.g. module `Up`)')
    parser.add_argument('--stride_x', type=int, default=2, help='Stride in x direction during computation.')
    parser.add_argument('--stride_y', type=int, default=2, help='Stride in y direction during computation.')
    args = parser.parse_args()

    stride_x = args.stride_x
    stride_y = args.stride_y

    IMAGE = args.IMAGE
    LAYER = args.LAYER
    INPUT = str(args.INPUT)

    shlqi2_path = Path(os.getcwd()) / 'CRISP' / 'tests' / 'UNet' / 'QI2_files' / IMAGE.split('.')[0] / 'simple' / 'computed' / f'SHLQI2_{LAYER}_input_{INPUT}.pkl'
    data_path = Path(os.getcwd()) / 'CRISP' / 'tests' / 'UNet' / 'QI2_files' / IMAGE.split('.')[0] / 'simple' / f'{LAYER}_input_{INPUT}.pkl'
    image_path = Path(os.getcwd()) / 'CRISP' / 'tests' /'UNet' / 'images' / f'{IMAGE}'
    model = model_utils.getModel()


    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(fig_w, fig_h), layout='constrained')

    shlqi2:SHLQI2 = load(shlqi2_path)
    shlqi2.gamma = gamma
    shlqi2.get_histogram()
    
    data = load(data_path)
    
    analysis = str(data_path).split('.')[0].split('\\')[-2]
    
    canvas = Canvas(fig,
                    axs=(ax1,ax2),
                    QI=shlqi2,
                    interactive=True,
                    ellipses=True,
                    fontsize=matplotlib.rcParams['font.size'],
                    labelsize=matplotlib.rcParams['font.size'])
    
    inputdata = list(data.values())[0]
    outputdata = list(data.values())[1]

    #import reducer as reducer
    #inputdata, _ = reducer.reduce_dimensions_umap(torch.hstack((inputdata, outputdata)), 
    #                                                  outputdata, n_neighbors=15, 
    #                                                  n_dimensions=2)
    #inputdata = torch.tensor(inputdata)
    
    canvas.vis_scatter(inputdata[:,0], inputdata[:,1], 
                       title=r'UMAP reduced data', 
                       xlabel=r'$x$', ylabel=r'$y$')
    canvas.vis_imshow(shlqi2.result, title=r'SHLQI$^2$', 
                      xlabel=r'$k$', ylabel=r'$v$')
    
    img = cv2.imread(image_path)
    img_net = model_utils.preprocess(img)

    fig_heatmap, ax_heatmap = plt.subplots(1,1,figsize=(heatmap_w, heatmap_h), layout='constrained')

    analyzer = ModelAnalyzer(model=model, 
                             model_class='Convolutional',
                             input_image=img_net,
                             layer=LAYER,
                             stride_w=stride_x,
                             stride_h=stride_y,
                             visualization=True,
                             canvas=canvas,
                             gamma_bar=True,
                             _add_callback=True,
                             fig=fig_heatmap,
                             axes=ax_heatmap,
                             padding=False,
                             analysis=analysis)
    
    plt.show()
