At the individual level, we define a graph as G = (N, E, A, F) representing a brain connectome, where A in R n×n is a connectivity matrix capturing the pairwise relationships between n ROIs and F in R n×f is a feature matrix where f is the dimension of node’s feature. It is initialized using an identity matrix since in a brain graph there are no features naturally assigned to brain regions. 

At the ==population level==, a graph can encode the ==relationship between a set of connectomes==, where A denotes an affinity matrix capturing the similarity between n brain graphs and F in R n×f is a feature matrix stacking the feature vector of size f of each node (i.e, a single subject) in the population graph (Figure 5). Therefore, N is a set of nodes (i.e, ROIs or subjects), E is a set of edges denoting either the biological connectivity between the brain regions or the similarity between brain graphs. In case the nodes do not have features, F can be represented by an identity matrix.

(Bessadok15)

