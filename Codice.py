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

# Load connectivity data from .mat file
data_path = r"C:\Users\barbo\Desktop\thesis repo clone\thesis\scfc_schaefer100_ya_oa\scfc_schaefer100_ya_oa.mat"

data = load_mat_file(data_path)

"""
# Print data type
print("Type of data:", type(data)) # <class 'dict'>

# Print the keys in data
if isinstance(data, dict):
    print("Keys in data:", data.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'data'])

    # Print the overview of the keys
    for key in data.keys():
        print(f"Overview of data[{key}]: Type: {type(data[key])}, Shape/Length: {np.shape(data[key]) if hasattr(data[key], 'shape') else len(data[key])}") # data[data]: Type: <class 'numpy.ndarray'>, Shape/Length: (1, 1) 
"""

# Extract the content of the 'data' key
data_content = data['data'][0,0] 

"""
# Print the type of data content
print("Type of data_content:", type(data_content)) # <class 'numpy.void'>

# Print the fields in sc_content and fc_content
print("data_content fields:", data_content.dtype.names) # ('sc_ya', 'fc_ya', 'sc_oa', 'fc_oa', 'info')
"""

# Extract the connectivity matrices and store them in a dictionary
matrices = {}

matrices['sc_ya'] = np.array(data_content['sc_ya'])
matrices['fc_ya'] = np.array(data_content['fc_ya'])
matrices['sc_oa'] = np.array(data_content['sc_oa'])
matrices['fc_oa'] = np.array(data_content['fc_oa'])

"""
# Print the type and shape of the matrices 
for key, matrix in matrices.items():
    print(f"{key}: Type: {type(matrix)}, Shape: {matrix.shape}") 
    # sc_ya: Type: <class 'numpy.ndarray'>, Shape: (100, 100, 101)
    # fc_ya: Type: <class 'numpy.ndarray'>, Shape: (100, 100, 101)
    # sc_oa: Type: <class 'numpy.ndarray'>, Shape: (100, 100, 78)
    # fc_oa: Type: <class 'numpy.ndarray'>, Shape: (100, 100, 78)
"""

# Check feasibility of the matrices

def check_symetric(matrix):
    """Check if a matrix is symmetric."""
    return np.allclose(matrix, matrix.T)

def check_zero_diagonal(matrix):
    """Check if the diagonal of a matrix is zero."""
    return np.allclose(np.diag(matrix), 0)

def check_dimensions(matrix, expected_shape=(100, 100)):
    """Check if the matrix has the expected shape."""
    return matrix.shape == expected_shape

# Check the properties of the matrices
"""
def check_properties(matrices):
    for key, matrix in matrices.items():
        age_group, matrix_type = key.split('_')
        num_subjects = matrix.shape[2]

        for i in range(num_subjects):
            symmetric = check_symetric(matrix[:, :, i])
            zero_diagonal = check_zero_diagonal(matrix[:, :, i])
            correct_shape = check_dimensions(matrix[:, :, i], (100, 100))
            
            if not symmetric or not zero_diagonal or not correct_shape:
                print(f"{matrix_type.upper()} {age_group.capitalize()} Subject {i+1}:")
                if not symmetric:
                    print(" Not Symmetric ")
                if not zero_diagonal:
                    print(" Diagonal is not Zero ")
                if not correct_shape:
                    print(" Incorrect Shape ")

# Check properties of matrices

check_properties(matrices)
"""



import networkx as nx

# preprocessing?


# Convert matrices to graphs
def matrix_to_graph(matrix):
    """Convert a matrix to a graph."""
    # matrix = np.array(matrix)
    graph = nx.from_numpy_array(matrix) 
    return graph


graphs = {}

for key, matrix in matrices.items():
    num_subjects = matrix.shape[2]
    graphs[key] = [matrix_to_graph(matrix[:, :, i]) for i in range(num_subjects)] #list of graphs



# Visualize the graphs

def plot_graph_on_axis(graph, ax, title, pos=None, partition=None):
    """Plot a graph on a given axis with node sizes proportional to the degree and edge widths proportional to the edge weights. 
        If partition is provided, nodes are colored by their community."""
    ax.set_title(title)

    # Extract edge weights
    edge_weights = [graph[u][v]['weight'] for u, v in graph.edges()]
    # Apply min-max normalization for edge thickness
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)
    if min_weight != max_weight:  # Avoid division by zero
        edge_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
    else:
        edge_weights = [1 for _ in edge_weights]  # If all weights are the same, set them to 1

    # Calculate the degree of each node for node size
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


def plot_and_save_graph(graphs, filename=None, positions=None, partitions=None, status="Original", with_communities=False, num_subjects=1):
    """
    Plot and save graphs for different age groups on a grid of subplots.
    
    Parameters:
    - graphs: Dictionary containing NetworkX graphs for each age group and matrix type
    - filename: name.png
    - positions: List of positions for the graphs
    - partitions: Dictionary containing partitions for the graphs of each age group and matrix type
    - status: 'Original', 'Preprocessed', or 'Louvain Preprocessed'
    - num_subjects: Number of subjects to plot (default is 1)
    """
    community_text = " with Louvain Communities" if with_communities else ""
    title = f'{status} Graphs{community_text}'
    fig, axs = plt.subplots(4, num_subjects, figsize=(15, 9))
    fig.suptitle(title, fontsize=16)

    if positions is None:
        positions = {}
    
    if partitions is None:
        partitions = {key: [None] * num_subjects for key in graphs.keys()}

    # TODO
    """for j, (key, graph_list) in enumerate(graphs.items()):
        age_group, matrix_type = key.split('_')
        title_prefix = matrix_type.upper()
        for i in range(min(num_subjects, len(graph_list))):
            if len(positions.get(key, [])) > i:
                pos = positions[key][i]
                plot_graph_on_axis(graph_list[i], axs[j, i], f"{key.upper()} Graph {i+1}", pos, partitions[key][i] if partitions[key] else None)
            else:
                pos = plot_graph_on_axis(graph_list[i], axs[j, i], f"{key.upper()} Graph {i+1}", partition=partitions[key][i] if partitions[key] else None)
                if key not in positions:
                    positions[key] = []
                positions[key].append(pos)"""

    plt.tight_layout()
    #plt.show()
    
    if filename:
        save_figure(fig, filename, title)
    
    plt.close(fig)

    return positions



# plot_and_save_graph(graphs, filename='graphs.png', status=' ', num_subjects=1)






# GAT for aging biomarker identification

# combine structural and functional graphs in a single graph

import torch
import scipy.stats
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch_geometric.nn import GATConv, TopKPooling, global_mean_pool
from torch_geometric.data import Data, DataLoader


def calculate_node_features(sc_matrix):
    """ sc_matrix is a 2D torch tensor of shape (num_nodes, num_nodes) representing the structural connectivity matrix of a single subject.
    features are from connectivity statistics: mean, standard deviation, kurtosis and
    skewness of the node's connectivity vector to all the other nodes (Yang 2019)
    """
    node_features = []

    for i in range(sc_matrix.shape[0]):
        # Functional features from connectivity statistics
        connections = sc_matrix[i, :]
        mean = connections.mean().item()
        std = connections.std().item()
        skew = scipy.stats.skew(connections.numpy())
        kurtosis = scipy.stats.kurtosis(connections.numpy())
        node_features.append([mean, std, skew, kurtosis])
    
    return torch.tensor(node_features, dtype=torch.float32)


def combined_graph(fc_matrix, sc_matrix):
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
    
    # does not make sense
    #fc_node_features = torch.empty((num_nodes, num_features, num_subjects), dtype=torch.float32)
    
    # Initialize empty list for storing graph data objects
    graph_list = []

    # Iterate over the third dimension of sc and fc tensor to fill them with node features for each subject
    for i in range(num_subjects):
        sc_node_features[:, :, i] = calculate_node_features(sc_tensor[:, :, i])
        #fc_node_features[:, :, i] = calculate_node_features(fc_tensor[:, :, i])

        # Set edges for the graph as in fc_matrix
        edges = []
        edge_weights = []
        for j in range(num_nodes):
            for k in range(j+1, num_nodes):
                weight = fc_matrix[j, k, i]
                # Remove null edges? (Yang doesn't)  
                # Add both directions since undirected?
                if weight != 0:  # Remove null edges
                    edges.append([j, k])
                    edges.append([k, j])
                    edge_weights.append(weight)
                    edge_weights.append(weight)

        # Convert graph edges in form of torch tensor of size (2, num_edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() 

        # Convert graph edge weights in form of torch tensor 
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

        # Create graph data object
        data = Data(x=sc_node_features[:,:,i], edge_index = edge_index, edge_attr = edge_weights)
        
        # Append the graph data object to the list
        graph_list.append(data)

    return graph_list


# Create a single graph from the combined matrices 
# Each element is a list of 5 graph data objects
combined_graphs = {
    'ya': combined_graph(matrices['fc_ya'], matrices['sc_ya']), #list of graphs
    'oa': combined_graph(matrices['fc_oa'], matrices['sc_oa'])
}

# check number of nodes and non-zero edges?

# Create labeled graph set for GAT training (structural, functional and combined)


combined_graphs_labeled = [] #len = 179

for label, graph_list in combined_graphs.items():
    for i in range(len(graph_list)):
        combined_graphs_labeled.append((graph_list[i], label)) 

structural_graphs_labeled = [] # len = 179
functional_graphs_labeled = [] # len = 179

for key, graph_list in graphs.items():
    type, label = key.split('_')
    if type == 'sc':
        for i in range(len(graph_list)):
            structural_graphs_labeled.append((graph_list[i], label)) 
    else:
        for i in range(len(graph_list)):
            functional_graphs_labeled.append((graph_list[i], label))



# Split data in training and test sets (70-30)

from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(combined_graphs_labeled, test_size=0.3, random_state=42)


class EGATNet(nn.Module):
    def __init__(self):
        super(EGATNet, self).__init__()
        self.gat1 = GATConv(in_channels=4, out_channels=8, heads=5, concat=True)
        self.pool1 = TopKPooling(in_channels=8*5, ratio=32/129)
        self.gat2 = GATConv(in_channels=8*5, out_channels=16, heads=1, concat=True)
        self.pool2 = TopKPooling(in_channels=16, ratio=4/32)
        self.fc1 = nn.Linear(16*4, 64)
        self.fc2 = nn.Linear(64, 2)  # Assuming binary classification


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.gat1(x, edge_index)
        x, edge_index, edge_attr, batch, _ = self.pool1(x, edge_index, edge_attr, batch=batch)
        x = self.gat2(x, edge_index)
        x, edge_index, edge_attr, batch, _ = self.pool2(x, edge_index, edge_attr, batch=batch)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# Initialize the model, loss function, and optimizer
model = EGATNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train():
    model.train()
    for data, label in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    return correct / total

# Train the model
for epoch in range(50):  # Number of epochs
    train()
    train_acc = evaluate(train_loader)
    test_acc = evaluate(test_loader)
    print(f'Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
