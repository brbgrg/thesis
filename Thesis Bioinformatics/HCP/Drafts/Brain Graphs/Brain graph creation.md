To create a brain graph, one could consider the neurons and their synapses as the basic building blocks of graph (i.e., nodes and edges). However, this was demonstrated to be a computationally expensive task as it requires an intensive data acquisition and processing steps [14]. Hence, the scalable method of constructing a brain graph is to consider a set of neurons as a single node in the graph. This is achieved via several anatomical parcellation schemes applied to a particular imaging modality such as MRI (Figure 3). Conventionally, a connectome is an undirected graph which means that there is no inferences about possible directions of information flow between brain regions [14]. Therefore, a distinction is often drawn between three types of brain graphs: morphological brain graph, functional brain graph and structural brain graph [7], [8], [11].
(Bessadok15)

To generate a brain graph, one needs to pre-process and register the MRI data to a specific atlas space for automatic labelling of brain ROIs. This will result in a parcellation of the brain into n anatomical brain regions defining the resolution of the graph [68], [69]. In other word, low-resolution and super-resolution brain graphs are generally derived from different MRI atlases which are the products of error-prone and time-consuming image processing pipelines [69].
(Bessadok15)






