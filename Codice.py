import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os

def load_mat_file(path):
    """Load a .mat file and return the loaded data."""
    data = scipy.io.loadmat(path)
    return data

# Function to save figures
def save_figure(fig, filename, title=None):
    directory = r"C:\Users\barbo\Desktop\thesis repo clone\thesis\Thesis Draft\figures"
    report_file = r'C:\Users\barbo\Desktop\thesis repo clone\thesis\Thesis Draft\reports\report1.tex'

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    fig_path = os.path.join(directory, filename)
    fig.savefig(fig_path)

    if title is None:
        # Remove the file extension
        base_name = os.path.splitext(filename)[0]
        # Replace underscores with spaces and capitalize each word
        title = base_name.replace('_', ' ').title()

    # Check if the figure path is already in the file
    fig_include_str = fig_path.replace('\\', '/')
    with open(report_file, 'r') as f:
        content = f.read()
        if fig_include_str in content:
            return

    # Add the figure to the summary file
    with open(report_file, 'a') as f:
        f.write("\\begin{figure}[h]\n")
        f.write("\\centering\n")
        f.write("\\includegraphics[width=0.8\\textwidth]{{{}}}\n".format(fig_path.replace('\\', '/')))
        f.write("\\caption{{{}}}\n".format(title))
        f.write("\\end{figure}\n\n")

    plt.close(fig)

# TODO: create configuration file
# TODO: create docker file


# Load SC and FC data from .mat files
sc_data_path = r"C:\Users\barbo\Desktop\thesis repo clone\thesis\brain_net\SC4alignment.mat"  # Update this path
fc_data_path = r"C:\Users\barbo\Desktop\thesis repo clone\thesis\brain_net\FC4alignment.mat"  # Update this path

sc_data = load_mat_file(sc_data_path)
fc_data = load_mat_file(fc_data_path)


# TODO: create a function that taken in input sc_data, fc_data extracts the content


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
sc_young_matrix = np.array(sc_content['young'])
sc_adult_matrix = np.array(sc_content['adult'])
sc_old_matrix = np.array(sc_content['old'])

fc_young_matrix = np.array(fc_content['young'])
fc_adult_matrix = np.array(fc_content['adult'])
fc_old_matrix = np.array(fc_content['old'])



# Print the type and shape of the matrices
"""
print("sc_young_matrix:", type(sc_young_matrix), sc_young_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
print("sc_adult_matrix:", type(sc_adult_matrix), sc_adult_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
print("sc_old_matrix:", type(sc_old_matrix), sc_old_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)

print("fc_young_matrix:", type(fc_young_matrix), fc_young_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
print("fc_adult_matrix:", type(fc_adult_matrix), fc_adult_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
print("fc_old_matrix:", type(fc_old_matrix), fc_old_matrix.shape) # <class 'numpy.ndarray'> (200, 200, 5)
"""


# Visualize matrices as a heatmap

def plot_and_save_heatmap(young_matrix, adult_matrix, old_matrix, title_prefix, filename=None, preprocessed = False):
    """
    Visualize the matrices as heatmaps.
    
    Parameters:
    - title_prefix: 'SC' or 'FC'
    - filename: name.png
    """
    status = "Preprocessed" if preprocessed else "Original"
    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    title = f'Heatmap of {status} {title_prefix} Matrices'
    fig.suptitle(title, fontsize=16)

    for i in range(5):
        im = axs[0, i].imshow(young_matrix[:, :, i], cmap='viridis')
        axs[0, i].set_title(f"Young {title_prefix} Matrix {i+1}")
        fig.colorbar(im, ax=axs[0, i])
        
        im = axs[1, i].imshow(adult_matrix[:, :, i], cmap='viridis')
        axs[1, i].set_title(f"Adult {title_prefix} Matrix {i+1}")
        fig.colorbar(im, ax=axs[1, i])
        
        im = axs[2, i].imshow(old_matrix[:, :, i], cmap='viridis')
        axs[2, i].set_title(f"Old {title_prefix} Matrix {i+1}")
        fig.colorbar(im, ax=axs[2, i])

    plt.tight_layout()
    #plt.show()

    if filename:
        save_figure(fig, filename, title)
    
    plt.close(fig)



# Plot the histogram 

import matplotlib.pyplot as plt
import numpy as np

def plot_and_save_histogram(young_matrix, adult_matrix, old_matrix, title_prefix, filename=None, preprocessed = False):
    """
    Plot and save the histogram of the upper triangular part (excluding the diagonal) of a matrix.
    
    Parameters:
    - young_matrix: 3D numpy array of shape (n, n, num_subjects)
    - adult_matrix: 3D numpy array of shape (n, n, num_subjects)
    - old_matrix: 3D numpy array of shape (n, n, num_subjects)
    - title_prefix: 'FC' or 'SC'
    - filename: name.png
    """
    status = "Preprocessed" if preprocessed else "Original"
    title = f'Histogram of {status} {title_prefix} Matrices'
    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(title, fontsize=16)

    for i in range(5):
        # Young group
        upper_tri_matrix = np.triu(young_matrix[:, :, i], k=1)
        flattened_matrix = upper_tri_matrix.flatten()
        axs[0, i].hist(flattened_matrix, bins=100, edgecolor='k')
        axs[0, i].set_title(f"Young {title_prefix} Matrix {i+1}")
        axs[0, i].set_xlabel('Value')
        axs[0, i].set_ylabel('Frequency (log scale)')
        axs[0, i].set_yscale('log')

        # Adult group
        upper_tri_matrix = np.triu(adult_matrix[:, :, i], k=1)
        flattened_matrix = upper_tri_matrix.flatten()
        axs[1, i].hist(flattened_matrix, bins=100, edgecolor='k')
        axs[1, i].set_title(f"Adult {title_prefix} Matrix {i+1}")
        axs[1, i].set_xlabel('Value')
        axs[1, i].set_ylabel('Frequency (log scale)')
        axs[1, i].set_yscale('log')

        # Old group
        upper_tri_matrix = np.triu(old_matrix[:, :, i], k=1)
        flattened_matrix = upper_tri_matrix.flatten()
        axs[2, i].hist(flattened_matrix, bins=100, edgecolor='k')
        axs[2, i].set_title(f"Old {title_prefix} Matrix {i+1}")
        axs[2, i].set_xlabel('Value')
        axs[2, i].set_ylabel('Frequency (log scale)')
        axs[2, i].set_yscale('log')

    plt.tight_layout()
    #plt.show()
    
    if filename:
        save_figure(fig, filename, title)
    
    plt.close(fig)

 

# Check feasibility of the matrices

def check_symetric(matrix):
    """Check if a matrix is symmetric."""
    return np.allclose(matrix, matrix.T)

def check_zero_diagonal(matrix):
    """Check if the diagonal of a matrix is zero."""
    return np.allclose(np.diag(matrix), 0)

def check_dimensions(matrix, expected_shape = (200, 200)):
    """Check if the matrix has the expected shape."""
    return matrix.shape == expected_shape


# Check the properties of the SC matrices

def check_properties(matrix, age_group, matrix_type):
    symmetric = [check_symetric(matrix[:, :, i]) for i in range(5)]
    zero_diagonal = [check_zero_diagonal(matrix[:, :, i]) for i in range(5)]
    correct_shape = [check_dimensions(matrix[:, :, i], (200, 200)) for i in range(5)]
    
    print(f"{matrix_type} {age_group} Symmetric: {symmetric}")
    print(f"{matrix_type} {age_group} Zero Diagonal: {zero_diagonal}")
    print(f"{matrix_type} {age_group} Correct Shape: {correct_shape}")
    
    return symmetric, zero_diagonal, correct_shape

# Check properties of SC matrices
"""
sc_young_symmetric, sc_young_zero_diagonal, sc_young_correct_shape = check_properties(sc_young_matrix, "Young", "SC")
sc_adult_symmetric, sc_adult_zero_diagonal, sc_adult_correct_shape = check_properties(sc_adult_matrix, "Adult", "SC")
sc_old_symmetric, sc_old_zero_diagonal, sc_old_correct_shape = check_properties(sc_old_matrix, "Old", "SC")
"""

# Check properties of FC matrices
"""
fc_young_symmetric, fc_young_zero_diagonal, fc_young_correct_shape = check_properties(fc_young_matrix, "Young", "FC")
fc_adult_symmetric, fc_adult_zero_diagonal, fc_adult_correct_shape = check_properties(fc_adult_matrix, "Adult", "FC")
fc_old_symmetric, fc_old_zero_diagonal, fc_old_correct_shape = check_properties(fc_old_matrix, "Old", "FC")
"""


# TODO: Check if the matrices are normalized


# TODO: Graph metrics


#------------------------------------#

# Community Detection

import networkx as nx
import community as community_louvain
from scipy.stats import zscore



# Preprocessing the matrices 


# TODO: denoising and sparsification?

def preprocess_fc_matrix(matrix, threshold=0.5, method='zscore'):
    """Preprocess the FC matrix to set negative weights to zero, apply a threshold, normalize and scale"""
    # Make a copy of the matrix to avoid in-place modification
    matrix_copy = matrix.copy()

    # Set negative weights to zero
    matrix_copy[matrix_copy < 0] = 0

    # Apply thresholding
    matrix_copy[matrix_copy < threshold] = 0
    normalized_matrix = matrix_copy

    """
    # Normalize the matrix based on the selected method
    if method == 'zscore':
        # Flatten the matrix to apply z-score normalization
        flat_matrix = matrix_copy.flatten()
        # Apply z-score normalization
        normalized_flat_matrix = zscore(flat_matrix)
        # Reshape back to the original matrix shape
        normalized_matrix = normalized_flat_matrix.reshape(matrix_copy.shape)
    elif method == 'global_mean':
        # Normalize the matrix by dividing by the global mean
        global_mean = np.mean(matrix_copy)
        normalized_matrix = matrix_copy / global_mean
    else:
        raise ValueError("Unsupported normalization method")
    """
    # Apply min-max scaling to ensure values are in the range [0, 1]
    min_val = np.min(normalized_matrix)
    max_val = np.max(normalized_matrix)
    rescaled_matrix = (normalized_matrix - min_val) / (max_val - min_val)
    
    return rescaled_matrix


# Initialize empty arrays to store the preprocessed FC matrices
fc_young_matrix_preprocessed = np.empty_like(fc_young_matrix)
fc_adult_matrix_preprocessed = np.empty_like(fc_adult_matrix)
fc_old_matrix_preprocessed = np.empty_like(fc_old_matrix)

# Apply the preprocessing function

for i in range(fc_young_matrix.shape[2]):
    fc_young_matrix_preprocessed[:, :, i] = preprocess_fc_matrix(fc_young_matrix[:, :, i], threshold= 0.5, method='zscore')
    fc_adult_matrix_preprocessed[:, :, i] = preprocess_fc_matrix(fc_adult_matrix[:, :, i], threshold= 0.5, method='zscore')
    fc_old_matrix_preprocessed[:, :, i] = preprocess_fc_matrix(fc_old_matrix[:, :, i], threshold= 0.5, method='zscore')



# Normalization of the SC matrices 
def preprocess_sc_matrix(matrix, method='zscore'):
    """Preprocess the SC matrix to apply normalization and min-max scaling."""
    # Make a copy of the matrix to avoid in-place modification
    matrix_copy = matrix.copy()
    normalized_matrix = matrix_copy

    """
    
    if method == 'zscore':
        # Flatten the matrix to apply z-score normalization
        flat_matrix = matrix_copy.flatten()
        # Apply z-score normalization
        normalized_flat_matrix = zscore(flat_matrix)
        # Reshape back to the original matrix shape
        normalized_matrix = normalized_flat_matrix.reshape(matrix_copy.shape)
    elif method == 'global_mean':
        # Normalize the matrix by dividing by the global mean
        global_mean = np.mean(matrix_copy)
        normalized_matrix = matrix_copy / global_mean
    else:
        raise ValueError("Unsupported normalization method")
    """
    
    # Apply min-max scaling to ensure values are in the range [0, 1]
    min_val = np.min(normalized_matrix)
    max_val = np.max(normalized_matrix)
    rescaled_matrix = (normalized_matrix - min_val) / (max_val - min_val)
    
    return rescaled_matrix

# Initialize empty arrays to store the preprocessed SC matrices
sc_young_matrix_preprocessed = np.empty_like(sc_young_matrix)
sc_adult_matrix_preprocessed = np.empty_like(sc_adult_matrix)
sc_old_matrix_preprocessed = np.empty_like(sc_old_matrix)

# Apply the preprocessing function
for i in range(sc_young_matrix.shape[2]):
    sc_young_matrix_preprocessed[:, :, i] = preprocess_sc_matrix(sc_young_matrix[:, :, i], method='zscore')
    sc_adult_matrix_preprocessed[:, :, i] = preprocess_sc_matrix(sc_adult_matrix[:, :, i], method='zscore')
    sc_old_matrix_preprocessed[:, :, i] = preprocess_sc_matrix(sc_old_matrix[:, :, i], method='zscore')


"""
# Visualize the original SC matrices as a heatmap
plot_and_save_heatmap(sc_young_matrix, sc_adult_matrix, sc_old_matrix, 'SC', 'SC_matrices_heatmap.png')

# Visualize the original FC matrices as a heatmap
plot_and_save_heatmap(fc_young_matrix, fc_adult_matrix, fc_old_matrix, 'FC', 'FC_matrices_heatmap.png')

# Visualize the preprocessed FC matrices as a heatmap
plot_and_save_heatmap(fc_young_matrix_preprocessed, fc_adult_matrix_preprocessed, fc_old_matrix_preprocessed, 'FC', 'FC_matrices_heatmap_preprocessed.png', preprocessed=True)

# Visualize the preprocessed SC matrices as a heatmap
plot_and_save_heatmap(sc_young_matrix_preprocessed, sc_adult_matrix_preprocessed, sc_old_matrix_preprocessed, 'SC', 'SC_matrices_heatmap_preprocessed.png', preprocessed=True)
"""

"""
# Plot the histogram of the original FC matrix values
plot_and_save_histogram(fc_young_matrix, fc_adult_matrix, fc_old_matrix, 'FC', 'FC_matrices_histogram.png')

# Plot the histogram of the original SC matrix values
plot_and_save_histogram(sc_young_matrix, sc_adult_matrix, sc_old_matrix, 'SC', 'SC_matrices_histogram.png')

# Plot the histogram of the preprocessed FC matrix values
plot_and_save_histogram(fc_young_matrix_preprocessed, fc_adult_matrix_preprocessed, fc_old_matrix_preprocessed, 'FC', 'FC_matrices_histogram_preprocessed.png', preprocessed=True)

# Plot the histogram of the preprocessed SC matrix values
plot_and_save_histogram(sc_young_matrix_preprocessed, sc_adult_matrix_preprocessed, sc_old_matrix_preprocessed, 'SC', 'SC_matrices_histogram_preprocessed.png', preprocessed=True)
"""
    

# Convert matrices to graphs
def matrix_to_graph(matrix):
    """Convert a matrix to a graph."""
    # matrix = np.array(matrix)
    graph = nx.from_numpy_array(matrix) 
    return graph


# Convert the original matrices to graphs
sc_young_graph = [matrix_to_graph(sc_young_matrix[:, :, i]) for i in range(5)]
sc_adult_graph = [matrix_to_graph(sc_adult_matrix[:, :, i]) for i in range(5)]
sc_old_graph = [matrix_to_graph(sc_old_matrix[:, :, i]) for i in range(5)]

fc_young_graph = [matrix_to_graph(fc_young_matrix[:,:,i]) for i in range(5)]
fc_adult_graph = [matrix_to_graph(fc_adult_matrix[:, :, i]) for i in range(5)]
fc_old_graph = [matrix_to_graph(fc_old_matrix[:, :, i]) for i in range(5)]

# Convert the preprocessed matrices to graphs
sc_young_graph_preprocessed = [matrix_to_graph(sc_young_matrix_preprocessed[:, :, i]) for i in range(5)]
sc_adult_graph_preprocessed = [matrix_to_graph(sc_adult_matrix_preprocessed[:, :, i]) for i in range(5)]
sc_old_graph_preprocessed = [matrix_to_graph(sc_old_matrix_preprocessed[:, :, i]) for i in range(5)]

fc_young_graph_preprocessed = [matrix_to_graph(fc_young_matrix_preprocessed[:,:,i]) for i in range(5)]
fc_adult_graph_preprocessed = [matrix_to_graph(fc_adult_matrix_preprocessed[:, :, i]) for i in range(5)]
fc_old_graph_preprocessed = [matrix_to_graph(fc_old_matrix_preprocessed[:, :, i]) for i in range(5)]



# Visualize the graphs

def plot_graph_on_axis(graph, ax, title, pos = None, partition = None):
    """Plot a graph on a given axis with node sizes proportional to the degree and edge widths proportional to the edge weights. 
        If partition is provided, nodes are colored by their community."""
    ax.set_title(title)

    # Extract edge weights
    edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    # Apply min-max normalization
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    if min_weight != max_weight:  # Avoid division by zero
        edge_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
    else:
        edge_weights = [1 for _ in edge_weights]  # If all weights are the same, set them to 1

    # Calculate the degree of each node
    degrees = dict(graph.degree())
    # Normalize the degrees to the range 0-1
    max_degree = max(degrees.values())
    min_degree = min(degrees.values())
    normalized_degrees = {node: (degree - min_degree) / (max_degree - min_degree) for node, degree in degrees.items()}
    # Set node sizes based on normalized degrees
    node_sizes = [(normalized_degrees[node] + 0.1) * 5 for node in graph.nodes()]  

    if partition:
        # If partition is provided, color nodes by their community
        cmap = plt.get_cmap('viridis', max(partition.values()) + 1)
        node_color = list(partition.values())
    else:
        node_color = 'steelblue'
    
    # Draw the graph
    if pos is None:
        pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, ax=ax, node_size=node_sizes, with_labels=False, node_color=node_color, cmap=cmap if partition else None, edge_color='gray', width=edge_weights)

    return pos


def plot_and_save_graph(young_graphs, adult_graphs, old_graphs, title_prefix, filename=None, positions = None, partitions = None, status = "Original", with_communities = False):
    """
    Plot and save graphs for different age groups on a grid of subplots.
    
    Parameters:
    - young_graphs: List of 5 NetworkX graphs for young subjects
    - adult_graphs: List of 5 NetworkX graphs for adult subjects
    - old_graphs: List of 5 NetworkX graphs for old subjects
    - title_prefix: 'FC' or 'SC'
    - filename: name.png
    - positions: List of positions for the graphs
    - partitions: List of 3 lists, each containing partitions for the graphs of each age group
    - status: 'Original', 'Preprocessed', or 'Louvain Preprocessed'
    """
    community_text = " with Louvain Communities" if with_communities else ""
    title = f'{status} {title_prefix} Graphs{community_text}'
    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(title, fontsize=16)

    if positions is None:
        positions = [] 
    
    if partitions is None:
        partitions = [None] * 3
        
    # add a funciton to validate the partitions and positions lengths = 5?

    for i in range(5):
        if len(positions) > i:
            pos = positions[i]
            plot_graph_on_axis(young_graphs[i], axs[0, i], f"Young {title_prefix} Graph {i+1}", pos, partitions[0][i] if partitions[0] else None)
        else:
            pos = plot_graph_on_axis(young_graphs[i], axs[0, i], f"Young {title_prefix} Graph {i+1}", partition=partitions[0][i] if partitions[0] else None)
            positions.append(pos)
        plot_graph_on_axis(adult_graphs[i], axs[1, i], f"Adult {title_prefix} Graph {i+1}", pos, partitions[1][i] if partitions[1] else None)
        plot_graph_on_axis(old_graphs[i], axs[2, i], f"Old {title_prefix} Graph {i+1}", pos, partitions[2][i] if partitions[2] else None)

    plt.tight_layout()
    #plt.show()
    
    if filename:
        save_figure(fig, filename, title)
    
    plt.close(fig)

    return positions


"""
# Visualize the original SC graphs
positions_sc = plot_and_save_graph(sc_young_graph, sc_adult_graph, sc_old_graph, 'SC', 'SC_graphs.png')

# Visualize the preprocessed SC graphs
plot_and_save_graph(sc_young_graph_preprocessed, sc_adult_graph_preprocessed, sc_old_graph_preprocessed, 'SC', 'SC_graphs_preprocessed.png', positions_sc, status="Preprocessed")

# Visualize the original FC graphs
positions_fc = plot_and_save_graph(fc_young_graph, fc_adult_graph, fc_old_graph, 'FC', 'FC_graphs.png', positions_sc)

# Visualize the preprocessed FC graphs
plot_and_save_graph(fc_young_graph_preprocessed, fc_adult_graph_preprocessed, fc_old_graph_preprocessed, 'FC', 'FC_graphs_preprocessed.png', positions_fc, status="Preprocessed")
"""



# Louvain Preprocessing

# TODO: try Louvain with only Louvain preprocessing

def louvain_preprocessing(graph):
    # Create a copy of the graph to avoid in-place modifications
    graph_copy = graph.copy()

    # Remove nodes with zero degree
    graph_copy.remove_nodes_from(list(nx.isolates(graph_copy)))

    # Remove self-loops
    graph_copy.remove_edges_from(nx.selfloop_edges(graph_copy))

    # Remove edges with weight 0
    zero_edges = [(u, v) for u, v, d in graph_copy.edges(data=True) if d.get('weight', 1) == 0]
    graph_copy.remove_edges_from(zero_edges)

    # Ensure the graph is connected (remove small disconnected components)
    if not nx.is_connected(graph_copy):
        # Get the largest connected component
        largest_cc = max(nx.connected_components(graph_copy), key=len)
        graph_copy = graph_copy.subgraph(largest_cc).copy()
    
    return graph_copy

# Apply the Louvain preprocessing function to the preprocessed FC graphs

fc_young_graph_preprocessed_louvain = [louvain_preprocessing(graph) for graph in fc_young_graph_preprocessed]
fc_adult_graph_preprocessed_louvain = [louvain_preprocessing(graph) for graph in fc_adult_graph_preprocessed]
fc_old_graph_preprocessed_louvain = [louvain_preprocessing(graph) for graph in fc_old_graph_preprocessed]

# Apply the Louvain preprocessing function to the preprocessed SC graphs
sc_young_graph_preprocessed_louvain = [louvain_preprocessing(graph) for graph in sc_young_graph_preprocessed]
sc_adult_graph_preprocessed_louvain = [louvain_preprocessing(graph) for graph in sc_adult_graph_preprocessed]
sc_old_graph_preprocessed_louvain = [louvain_preprocessing(graph) for graph in sc_old_graph_preprocessed]



# Visualize Louvain preprocessed graphs

""" I don't need to filter out removed nodes's positions from positions"""

"""
# Visualize the Louvain preprocessed SC graphs
plot_and_save_graph(sc_young_graph_preprocessed_louvain, sc_adult_graph_preprocessed_louvain, sc_old_graph_preprocessed_louvain, 'SC', 'SC_graphs_preprocessed_louvain.png', positions_sc, status="Louvain Preprocessed")

# Visualize the Louvain preprocessed FC graphs
plot_and_save_graph(fc_young_graph_preprocessed_louvain, fc_adult_graph_preprocessed_louvain, fc_old_graph_preprocessed_louvain, 'FC', 'FC_graphs_preprocessed_louvain.png', positions_fc, status="Louvain Preprocessed")
"""


# Community detection using Louvain method

#TODO: Louvain resolution parameters

def community_detection(graph):
    """Detect communities in a graph using Louvain method."""
    partition = community_louvain.best_partition(graph)
    return partition

"""
sc_young_partition = [community_detection(graph) for graph in sc_young_graph_preprocessed_louvain]
sc_adult_partition = [community_detection(graph) for graph in sc_adult_graph_preprocessed_louvain]
sc_old_partition = [community_detection(graph) for graph in sc_old_graph_preprocessed_louvain]

fc_young_partition = [community_detection(graph) for graph in fc_young_graph_preprocessed_louvain]
fc_adult_partition = [community_detection(graph) for graph in fc_adult_graph_preprocessed_louvain]
fc_old_partition = [community_detection(graph) for graph in fc_old_graph_preprocessed_louvain]
"""


#TODO: maybe it's not the best to use the same positions for visualizing communities 
"""
# Visualize SC graphs with communities
plot_and_save_graph(sc_young_graph_preprocessed_louvain, sc_adult_graph_preprocessed_louvain, sc_old_graph_preprocessed_louvain, 'SC', 'SC_graphs_preprocessed_louvain_communities.png', positions_sc, [sc_young_partition, sc_adult_partition, sc_old_partition], status="Louvain Preprocessed", with_communities=True)

# Visualize FC preprocessed Louvain graphs with communities 
plot_and_save_graph(fc_young_graph_preprocessed_louvain, fc_adult_graph_preprocessed_louvain, fc_old_graph_preprocessed_louvain, 'FC', 'FC_graphs_preprocessed_louvain_communities.png', positions_fc, [fc_young_partition, fc_adult_partition, fc_old_partition], status="Louvain Preprocessed", with_communities=True)
"""

# TODO: Community detection evaluation 
# TODO: Multi-modal and multi-subject modularity optimization?

# TODO: visualization with BrainNet Viewer




# GAT for aging biomarker identification

# combine structural and functional graphs in a single graph

import torch
import scipy.stats
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader


def calculate_node_features(matrix):
    """ matrix is a 2D torch tensor of shape (num_nodes, num_nodes) representing the connectivity matrix of a single subject.
    Anatomical features are Number of Vertices (NumVert), Surface
    Area (SurfArea), Gray Matter Volume (GrayVol), Average Thickness 
    (ThickAvg), Thickness Standard Deviation (ThickStd), Integrated 
    Rectified Mean Curvature (MeanCurv) and Integrated Rectified Gaussian 
    Curvature (GausCurv) [6],
    Functional features are from connectivity statistics: mean, standard deviation, kurtosis and
    skewness of the node s connectivity vector to all the other nodes (Yang 2019)
    """
    node_features = []

    for i in range(matrix.shape[0]):
        # Functional features from connectivity statistics
        connections = matrix[i, :]
        mean = connections.mean().item()
        std = connections.std().item()
        skew = scipy.stats.skew(connections.numpy())
        kurtosis = scipy.stats.kurtosis(connections.numpy())
        node_features.append([mean, std, skew, kurtosis])
    
    return torch.tensor(node_features, dtype=torch.float32)

def combined_graph(sc_matrix, fc_matrix):
    """Combine structural and functional connectivity matrices
     into a single graph representation where graphs are constructed 
     from the functional connectivity matrices and the node features 
     consist of both anatomical features and statistics of the nodal connectivity. (Yang 2019)"""
    
    # turn sc_matrix and fc_matrix into torch tensors
    sc_tensor = torch.tensor(sc_matrix, dtype=torch.float)
    fc_tensor = torch.tensor(fc_matrix, dtype=torch.float)

    # Initialize empty tensors to store node features for each subject
    num_subjects = sc_tensor.shape[2]
    num_nodes = sc_tensor.shape[0]
    num_features = 4  # mean, std, skew, kurtosis

    sc_node_features = torch.empty((num_nodes, num_features, num_subjects), dtype=torch.float32)
    fc_node_features = torch.empty((num_nodes, num_features, num_subjects), dtype=torch.float32)
    
    # Iterate over the third dimension of sc and fc tensor to fill them with node features for each subject
    for i in range(num_subjects):
        sc_node_features[:, :, i] = calculate_node_features(sc_tensor[:, :, i])
        fc_node_features[:, :, i] = calculate_node_features(fc_tensor[:, :, i])

    # Combine the node features
    node_features = torch.cat([sc_node_features, fc_node_features], dim=1)

    # Set edges for the graph
    # TODO: redo all section
    edges = []
    edge_weights = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            # Remove null edges?  
            # Add both directions since undirected?
            edges.append([i, j])
            edges.append([j, i])
            edge_weights.append(weight)
            edge_weights.append(weight)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() # (2, num_edges)
    # edge_weights = 

    # Create graph data object
    data = Data(x=node_features, edge_index = edge_index, edge_attr = edge_weights)
    
    return data


combined_graph(sc_young_matrix, fc_young_matrix)