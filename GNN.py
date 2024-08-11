import numpy as np

import scipy.io
import matplotlib.pyplot as plt

def load_mat_file(path):
    """Load a .mat file and return the loaded data."""
    data = scipy.io.loadmat(path)
    return data

def visualize_matrix(matrix, title="Matrix Visualization"):
    """Visualize a 2D matrix."""
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

# TODO: create configuration file
# TODO: create docker file


# Load SC and FC data from .mat files
sc_data_path = "/workspaces/thesis/brain_net/SC4alignment.mat"  # Update this path
fc_data_path = "/workspaces/thesis/brain_net/FC4alignment.mat"  # Update this path

sc_data = load_mat_file(sc_data_path)
fc_data = load_mat_file(fc_data_path)


# Print the type of sc_data and fc_data 
"""
print("Type of sc_data:", type(sc_data)) # <class 'dict'>
print("Type of fc_data:", type(fc_data)) # <class 'dict'>

# Print the keys in sc_data and fc_data
if isinstance(sc_data, dict) and isinstance(fc_data, dict):
    print("Keys in sc_data:", sc_data.keys())
    print("Keys in fc_data:", fc_data.keys())

    # Print the overview of the keys
    for key in sc_data.keys():
        print(f"Overview of sc_data[{key}]: Type: {type(sc_data[key])}, Shape/Length: {np.shape(sc_data[key]) if hasattr(sc_data[key], 'shape') else len(sc_data[key])}")
    for key in fc_data.keys():
        print(f"Overview of fc_data[{key}]: Type: {type(fc_data[key])}, Shape/Length: {np.shape(fc_data[key]) if hasattr(fc_data[key], 'shape') else len(fc_data[key])}")
""" 

# Extract the content of the 'sc' and 'fc' keys
sc_content = sc_data['sc'][0,0] 
fc_content = fc_data['fc'][0,0]

# Print the type of sc_content and fc_content
""""
print("Type of sc_content:", type(sc_content)) # <class 'numpy.void'>
print("Type of fc_content:", type(fc_content)) # <class 'numpy.void'>

# Print the fields in sc_content and fc_content
print("sc_content fields:", sc_content.dtype.names) # ('young', 'adult', 'old')
print("fc_content fields:", fc_content.dtype.names) # ('young', 'adult', 'old')
"""

# Extract the 'young', 'adult', and 'old' matrices from sc_content and fc_content
sc_young_matrix = sc_content['young']
sc_adult_matrix = sc_content['adult']
sc_old_matrix = sc_content['old']

fc_young_matrix = fc_content['young'] 
fc_adult_matrix = fc_content['adult']
fc_old_matrix = fc_content['old']

# Print the type and shape of the matrices
"""
print("sc_young_matrix:", type(sc_young_matrix), sc_young_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
print("sc_adult_matrix:", type(sc_adult_matrix), sc_adult_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
print("sc_old_matrix:", type(sc_old_matrix), sc_old_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)

print("fc_young_matrix:", type(fc_young_matrix), fc_young_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
print("fc_adult_matrix:", type(fc_adult_matrix), fc_adult_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
print("fc_old_matrix:", type(fc_old_matrix), fc_old_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
"""

# Visualize the matrices
"""
visualize_matrix(sc_young_matrix[:,:,0], title="Young SC Matrix")
visualize_matrix(sc_adult_matrix[:,:,0], title="Adult SC Matrix")
visualize_matrix(sc_old_matrix[:,:,0], title="Old SC Matrix")

visualize_matrix(fc_young_matrix[:,:,0], title="Young FC Matrix")
visualize_matrix(fc_adult_matrix[:,:,0], title="Adult FC Matrix")
visualize_matrix(fc_old_matrix[:,:,0], title="Old FC Matrix")
"""


#Given this dataset, can I build a Graph Neural Network (GNN) to perform network alignment for a comparative analysis of the modular structures in Structural Connectivity (SC) and Functional Connectivity (FC) networks across the three age groups?

# GNN
import torch
import torch.nn.functional as F
 
print (torch.__version__) # 2.4.0+cu118
