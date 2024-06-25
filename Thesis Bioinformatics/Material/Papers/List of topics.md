Application fields: 
Disease prediction
Drug discovery
Biomedical imaging

Papers:
Protein Structure and Function Prediction 
(most diseases are closely related to protein dysfunction, GNNs to integrate the feature of protein relationship networks to structural characteristics)
3. protein function prediction based on protein structure (Ioannidis et al., 2019) (multi-relation diagram method based on PPI network modeling with semi-supervised learning) 
4. protein function prediction based on PPI networks (Gligorijevic et al., 2019)

Medical Imaging
Brain Connectivity
1. Ktena, S. I., Parisot, S., Ferrante, E., Rajchl, M., Lee, M., Glocker, B., et al. (2018). Metric learning with spectral graph convolutions on brain connectivity networks. NeuroImage 169, 431–442. doi: 10.1016/j.neuroimage.2017.12.052 
2. Ktena, S. I., Parisot, S., Ferrante, E., Rajchl, M., Lee, M., Glocker, B., et al. (2017). “Distance metric learning using graph convolutional networks: Application to functional brain networks,” in International Conference on Medical Image
3. Li, X., Dvornek, N. C., Zhou, Y., Zhuang, J., Ventola, P., and Duncan, J. S. (2019a). “Graph neural network for interpreting task-fmri biomarkers,” in Proceedings of the International Conference on Medical Image Computing and ComputerAssisted Intervention, (Cham: New Orleans, Springer), 485–493. doi: 10.1007/ 978-3-030-32254-0_54
4. Mirakhorli, J., and Mirakhorli, M. (2019). Graph-based method for anomaly detection in functional brain network using variational autoencoder. bioRxiv [Preprint]. 616367.
5. Grigis, A., Tasserie, J., Frouin, V., Jarraya, B., and Uhrig, L. (2020). Predicting cortical signatures of consciousness using dynamic functional connectivity graph-convolutional neural networks. bioRxiv [Preprint]. doi: 10.1101/2020.05. 11.078535
6. Li, X., Zhou, Y., Gao, S., Dvornek, N., Zhang, M., Zhuang, J., et al. (2020). Braingnn: Interpretable brain graph neural network for fmri analysis. bioRxiv [Preprint]. doi: 10.1101/2020.05.16.100057
7. Zhang, Y., and Pierre, B. (2020). Transferability of brain decoding using graph convolutional networks. bioRxiv [Preprint]. doi: 10.1101/2020.06.21.163964 doi: 10.1016/j.compmedimag.2020.101748
8. Zhang, Y., Tetrel, L., Thirion, B., and Bellec, P. (2021). Functional annotation of human cognitive states using deep graph convolution. NeuroImage 231:117847. doi: 10.1016/j.neuroimage.2021.117847
1. Tian et al. (2020a) utilized spectral convolution for realizing cortical surface parcellation (for processing surface data (such as MRI), this method was able to directly learn the surface features of the cortex)


10. PPI networks with a small amount of label information that were encoded to predict the relationship between drugs and diseases (Bajaj et al., 2017) [[link prediction]]
11. disease similarity networks, and microRNA (miRNA) similarity networks were built to indicate the association between miRNA and disease by VGAE (Ding et al., 2020) [[link prediction]]
12. Graph residual NN for protein function prediction in brain cell dataset (Ioannidis et al 2019)
13. Network-based disease prediction methods (Barabasi et al., 2011)
14. Population disease prediction, Parisot et al. (2017)
15.  bio-inspired graph learning NicheCompass (Helmholz center Munich, Wellcome Trust Sanger Institute)
16. AlphaFold meets flow matching for generating protein ensembles

Graph analysis tasks:
Node Level:
21. population disease prediction 
28. prediction of unlabeled proteins through labeled proteins in the protein–protein interaction network 
22. inferring the association between entities in the biological network 
23. node classification in biological networks 
24. graph representation learning 
25. graph embedding methods
Edge Level:
1. ==link prediction== (has captured the attention of different research fields due to its broad applicability)
	2. predicting the interaction between biological entities (plays an important role in the research of bioinformatics and has become increasingly important and more challenging)
Graph Level:
10. de novo molecular design 
12. generation of synthetic molecules through actual molecular graph learning (graph-level GNN)
13. drug design
14. protein structure design
15. flow matching 
16. feature representations (GCN)

17. graph alignment
18. drug re-purposing 
19. gene-disease association prediction 
20. network alignment
21. molecular property prediction
22. network reconstruction

Architectures:
1. GNN
2. GCN
	1. The convolution operation on graph encodes the structure and features of protein into the graph embedding representation and aggregates information along the edges of the network nodes for association scores, which solves the spatial limitations of conventional convolution methods
3. GRNN (graph residual NN)
4. GAT (graph attention)
5. VAE (variational autoencoder)
6. MT-DNN (multi task deep NN) 
	1. makes neural networks more powerful in drug discovery
7. GCN + MT-DNN
	1. further improve prediction accuracy

Models:
1. Zhang and Chen (2018) proposed the SEAL (learning from subgraphs, embeddings, and attributes for link prediction) model
2. Neil et al. (2018) proposed a model that could adapt to noise data and reduced the influence of noisy data on the overall prediction effect of the model by assigning low weights to unreliable edges


Keywords:
1. statistical inference 
19. network analysis 
2. graph embedding model
3. disease similarity network
4. microRNA similarity network
5. computational biology 
6. gene co-expression networks 
7. gene regulatory networks 
8. metabolic networks 
9. protein-protein interactions
10. signaling networks 
11. brain networks 
12. symptom networks 
13. network medicine 
14. networks in neuroscience 
15. networks in psychology 
16. multi-omic 
17. multi-features 
18. integrative networks 
19. preventative and predictive medicine
20. stratification of patients
21. personalized medicine
22. patient-specific disease network
23. knowledge graph



