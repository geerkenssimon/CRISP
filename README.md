# CRISP - Complexity-based Reasoning of Internal Subprocessing

CRISP is a framework that provides tools for analyzing and interpreting the internal workings of neural networks. It leverages complexity-based reasoning to gain insights into subprocessing layers within deep learning models.

## Features

- **Complexity analysis with QI²:** Hidden-Layer application of QI² computations.
- **Layer-wise Relevance Propagation (LRP):** Enhanced LRP methods for detailed analysis of neural networks.
- **Visualization Tools:** Interactive tools for visualizing the relevance and internal processing layers of models.
- **Multi-device Support:** Efficient execution across multiple devices with optimized threading and resource management.
- **Logging and Exception Handling:** Comprehensive logging utilities and robust exception handling mechanisms.

## Installation

To install CRISP, create a virtual environment based on Python 3.11, go to the top level folder of this repo and install the necessary dependencies:

```bash
cd crisp
pip install -r requirements.txt
```

## Usage

1. **Baseline computations**
    Use the function `_per_layer_SHLQI2` in `/analysis/hidden_layer_pex/computations.py` to
    compute the SHLQI² for every layer of a given type in a given network.

    An example can be found in `/scripts/UNet_nn_computations.py`. This file does the 
    computations for the U-Net railway segmentation from the paper with the given parameters.

    ```bash
    python scripts/UNet_nn_computations.py
    ```

    This will trigger the computations and can take a while (depending on the hardware).
    With this, all layers of the given type in the U-Net are extracted and the SHLQI² is
    computed for all images in the folder `/CRIPS/tests/UNet/images`.
    The progress can be seen in the logging file `/log/QUEEN_log_<DATE>.log`

2. **Analyzing a Model:**
    Utilize the `ModelAnalyzer` to assess and interpret model layers. An example is given 
    in `/scripts/UNet_nn_analysis.py`. This file triggers the analysis process depending 
    on the given variables.

    Exemplary, we provide a few example files (Images, Layerdata, SHLQI²) for a few layers 
    to analyze, respective to the images shown in the paper. 


    ### Figure 5
    ```bash
    python scripts/UNet_nn_analysis.py --IMAGE rs00004.jpg --LAYER down1-maxpool_conv-1-conv-Conv2d0 --stride_x 2 --stride_y 2 
    ```

    ### Figure 6a
    ```bash
    python scripts/UNet_nn_analysis.py --IMAGE rs00506.jpg --LAYER down3-maxpool_conv-1-conv-Conv2d0 --stride_x 1 --stride_y 1
    ```

    ### Figure 6b
    ```bash
    python scripts/UNet_nn_analysis.py --IMAGE OSDAR_017_024.png --LAYER down4-maxpool_conv-1-conv-Conv2d1 --stride_x 1 --stride_y 1
    ```

    ### Figure 7a
    ```bash
    python scripts/UNet_nn_analysis.py --IMAGE rs00420.jpg --LAYER up2-conv-conv-Conv2d0 --stride_x 1 --stride_y 1
    ```

    ### Figure 7b
    ```bash
    python scripts/UNet_nn_analysis.py --IMAGE rs00420.jpg --LAYER up2-conv-conv-Conv2d1 --stride_x 1 --stride_y 1
    ```

    ### Figure 8a
    ```bash
    python scripts/UNet_nn_analysis.py --IMAGE rs00506.jpg --LAYER Up1 --stride_x 1 --stride_y 1
    ```
    
    ### Figure 8b
    ```bash
    python scripts/UNet_nn_analysis.py --IMAGE rs00506.jpg --LAYER Up1 --INPUT 1 --stride_x 1 --stride_y 1
    ```
    
    ### Figure 9
    ```bash
    python scripts/UNet_nn_analysis.py --IMAGE rs00018.jpg --LAYER up3-conv-conv-Conv2d0 --stride_x 2 --stride_y 2
    ```


    Variables:
    - `IMAGE` - name of the image stored in `/CRISP/tests/UNet/images` folder. default is `rs00004.jpg`
    - `LAYER` - name of the Layer to be analyzed. Example files only available as presented in the paper. default is `down1-maxpool_conv-1-conv-Conv2d0`. 
    - `INPUT` - Index of the input to be analyzed. only important if the module has multiple inputs (e.g. module `Up1`). default is 0
    - `stride_x`, `stride_y` - strides in x and y direction used in the computation process. defaults are 2.
    
    This file will load and visualize the given data and SHLQI² for the given layer. It opens
    two matplotlib windows. The first one is the interactive visualiation. In the right plot
    one can mark certain areas (square by default, but can be changed to polygon in the top bar).
    These areas are then used as a starting point for backward attribution with LRP to visualize
    the importance within the given input image.