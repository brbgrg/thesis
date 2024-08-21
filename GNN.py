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
sc_data_path = r"C:\Users\barbo\Desktop\thesis repo clone\thesis\brain_net\SC4alignment.mat"  # Update this path
fc_data_path = r"C:\Users\barbo\Desktop\thesis repo clone\thesis\brain_net\FC4alignment.mat"  # Update this path

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
fig, axs = plt.subplots(3, 5, figsize=(15, 9))

for i in range(5):
    axs[0, i].imshow(sc_young_matrix[:, :, i], cmap='viridis')
    axs[0, i].set_title(f"Young SC Matrix {i+1}")
    axs[1, i].imshow(sc_adult_matrix[:, :, i], cmap='viridis')
    axs[1, i].set_title(f"Adult SC Matrix {i+1}")
    axs[2, i].imshow(sc_old_matrix[:, :, i], cmap='viridis')
    axs[2, i].set_title(f"Old SC Matrix {i+1}")

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(3, 5, figsize=(15, 9))

for i in range(5):
    axs[0, i].imshow(fc_young_matrix[:, :, i], cmap='viridis')
    axs[0, i].set_title(f"Young FC Matrix {i+1}")
    axs[1, i].imshow(fc_adult_matrix[:, :, i], cmap='viridis')
    axs[1, i].set_title(f"Adult FC Matrix {i+1}")
    axs[2, i].imshow(fc_old_matrix[:, :, i], cmap='viridis')
    axs[2, i].set_title(f"Old FC Matrix {i+1}")

plt.tight_layout()
plt.show()
"""

# Community Detection
import networkx as nx
import community as community_louvain

# Convert matrices to graphs
def matrix_to_graph(matrix):
    """Convert a matrix to a graph."""
    graph = nx.from_numpy_array(matrix)
    return graph

sc_young_graph = [matrix_to_graph(sc_young_matrix[:, :, i]) for i in range(5)]
sc_adult_graph = [matrix_to_graph(sc_adult_matrix[:, :, i]) for i in range(5)]
sc_old_graph = [matrix_to_graph(sc_old_matrix[:, :, i]) for i in range(5)]

fc_young_graph = [matrix_to_graph(fc_young_matrix[:, :, i]) for i in range(5)]
fc_adult_graph = [matrix_to_graph(fc_adult_matrix[:, :, i]) for i in range(5)]
fc_old_graph = [matrix_to_graph(fc_old_matrix[:, :, i]) for i in range(5)]
                 

# Visualize the graphs
"""
def plot_graph_on_axis(graph, ax, title):
    ax.set_title(title)
    nx.draw(graph, ax = ax, with_labels=False, node_color='skyblue', node_size=1, edge_color='gray')

fig, axs = plt.subplots(3, 5, figsize=(15, 9))

for i in range(5):
    plot_graph_on_axis(sc_young_graph[i], axs[0, i], f"Young SC Graph {i+1}")
    plot_graph_on_axis(sc_adult_graph[i], axs[1, i], f"Adult SC Graph {i+1}")
    plot_graph_on_axis(sc_old_graph[i], axs[2, i], f"Old SC Graph {i+1}")

plt.tight_layout()
plt.show()
"""

# Community detection using Louvain method
def community_detection(graph):
    """Detect communities in a graph using Louvain method."""
    partition = community_louvain.best_partition(graph)
    return partition

sc_young_partition = [community_detection(graph) for graph in sc_young_graph]
sc_adult_partition = [community_detection(graph) for graph in sc_adult_graph]
sc_old_partition = [community_detection(graph) for graph in sc_old_graph]


# BAD NODE DEGREE


"""
fc_young_partition = [community_detection(graph) for graph in fc_young_graph]
fc_adult_partition = [community_detection(graph) for graph in fc_adult_graph]
fc_old_partition = [community_detection(graph) for graph in fc_old_graph]

"""

# Visualize the communities

def plot_communities_on_axis(graph, partition, ax, title):
    ax.set_title(title)
    pos = nx.spring_layout(graph)
    cmap = plt.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw(graph, pos, ax=ax, with_labels=False, node_color=list(partition.values()), node_size=1, cmap=cmap, edge_color='gray')




