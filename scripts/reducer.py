# build-in modules
import numpy as np

# third-party modules
import umap.umap_ as umap
from sklearn.preprocessing import MinMaxScaler
from tkinter import messagebox


def reduce_dimensions_umap(input_data, original_output, n_neighbors=15, scaled=True, n_dimensions=2, calc_mode=2):
    """ 
    --- INPUTS ---
    input_data:         ndarray; input and output data, output is last column
    original_output:    ndarray; output data (only neccessary for high dim output data)
    input_keys:         list with choosen Inputs; used for computation
    output_keys:        list with choosen Outputs; used for labeling
    n_neighbors:        int; No. of neighbors which are used for dimensionality reduction - small (5~20) lokal structure, big (35~50) global structure
                        for more details see: https://umap-learn.readthedocs.io/en/latest/parameters.html
    scaled:             boolean; if True, csv file data will be scaled cloumnwise
    n_dimensions:       int; desired no. of dimensions after reduction
    calc_mode:          int;    0: supervised - uses input and output keys for computation
                                1: unsupervised, unlabeled - uses inputs for computatation, no labeling
                                2: unsupervised, labeled - uses inputs for computation, outputs for labeling
    --- OUTPUTS ---
    out_data:           Nested list that includes reduced dimension data, labels for visualization
    """
    transformation = True

    if type(input_data) == type(None):
        messagebox.showwarning("WARNING", "Please select a valid csv-file to display UMAP.")
        return None

    else:
        if input_data.shape[1] == 3 and n_dimensions==3:
            transformation = False
        if len(original_output.shape) > 1:
            output_dims = original_output.shape[-1] 
        else:
            output_dims = 1
        if output_dims > 1:
            calc_mode = 1

        # fetch data
        set_data_org = input_data
        # labeling prep
        label_array = set_data_org[:,-output_dims]
        # select in/outputs
        if calc_mode == 0:
            set_data = set_data_org
        elif calc_mode == 1:
            set_data = set_data_org[:,:-output_dims]
            label_array = set_data_org[:,0]*0            
        elif calc_mode == 2:
            set_data = set_data_org[:,:-output_dims]
        else:
            print("Please choose valid supervise mode")

        labels = ['x','y','z'][:n_dimensions]  
        if not transformation:
            embedding = set_data
            return embedding, labels#, transformation

        reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_dimensions)
        # scaling
        if scaled:
            scaler = MinMaxScaler().fit(set_data)
            scaled_set_data = scaler.transform(set_data)
            embedding = reducer.fit_transform(scaled_set_data)
        else:
            embedding = reducer.fit_transform(set_data)
        
        return embedding, labels#, transformation
