import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

def load_mat_file(path):
    """Load a .mat file and return the loaded data."""
    data = scipy.io.loadmat(path)
    return data

# Load connectivity data from .mat file
data_path = r"C:\Users\barbo\Desktop\thesis repo clone\thesis\data\scfc_schaefer100_ya_oa\scfc_schaefer100_ya_oa.mat"

data = load_mat_file(data_path)


# Print data type
print("Type of data:", type(data)) # <class 'dict'>

# Print the keys in data
if isinstance(data, dict):
    print("Keys in data:", data.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'data'])

    # Print the overview of the keys
    for key in data.keys():
        print(f"Overview of data[{key}]: Type: {type(data[key])}, Shape/Length: {np.shape(data[key]) if hasattr(data[key], 'shape') else len(data[key])}") # data[data]: Type: <class 'numpy.ndarray'>, Shape/Length: (1, 1) 

# Extract the content of the 'data' key
data_content = data['data'][0,0] 

# Print the type of data content
print("Type of data_content:", type(data_content)) # <class 'numpy.void'>

# Print the fields in sc_content and fc_content
print("data_content fields:", data_content.dtype.names) # ('sc_ya', 'fc_ya', 'sc_oa', 'fc_oa', 'info')

# Extract the connectivity matrices and store them in a dictionary
matrices = {}

matrices['sc_ya'] = np.array(data_content['sc_ya'])
matrices['fc_ya'] = np.array(data_content['fc_ya'])
matrices['sc_oa'] = np.array(data_content['sc_oa'])
matrices['fc_oa'] = np.array(data_content['fc_oa'])

# Print the type and shape of the matrices 
for key, matrix in matrices.items():
    print(f"{key}: Type: {type(matrix)}, Shape: {matrix.shape}") 
    # sc_ya: Type: <class 'numpy.ndarray'>, Shape: (100, 100, 101)
    # fc_ya: Type: <class 'numpy.ndarray'>, Shape: (100, 100, 101)
    # sc_oa: Type: <class 'numpy.ndarray'>, Shape: (100, 100, 78)
    # fc_oa: Type: <class 'numpy.ndarray'>, Shape: (100, 100, 78)




