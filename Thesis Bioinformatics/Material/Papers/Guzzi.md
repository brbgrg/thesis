The paper "Survey of local and global biological network alignment: the need to reconcile the two sides of the same coin" by Pietro Hiram Guzzi and Tijana Milenković explores the methods and challenges associated with biological network alignment, comparing local and global approaches. 

### Abstract and Introduction

- Biological network alignment is analogous to genomic sequence alignment, aiming to transfer biological knowledge across species.
- The alignment can redefine sequence-based homology to network-based homology.
- There are two primary types of network alignment: local network alignment (LNA) and global network alignment (GNA).
- The paper surveys prominent computational approaches for each type, discusses their advantages and disadvantages, and the need to reconcile the two as they capture different aspects of cellular functioning.

### Local Network Alignment (LNA)

- LNA focuses on finding highly similar network regions, often resulting in small mapped subnetworks.
- LNA methods aim to identify conserved functional structures and building blocks of cellular machinery.
- LNA approaches can be categorized into mine-and-merge and merge-and-mine methods:
    - **Mine-and-merge**: Identify modules in each network and then align these modules.
    - **Merge-and-mine**: Merge networks into a single graph and identify conserved modules within this merged graph.
- Examples of LNA methods include MOPHY, MODULA, and NetworkBlast, each utilizing different strategies for identifying and aligning modules.

### Global Network Alignment (GNA)

- GNA seeks to find the best superimposition of entire input networks, aiming to maximize a cost function that evaluates the alignment.
- GNA methods typically produce larger but less functionally conserved mappings.
- Examples of GNA methods include IsoRank, GRAAL, and MAGNA, each using different approaches for node similarity and alignment strategies.
- Recent advances in GNA incorporate both node and edge conservation to improve alignment quality.

### Complementarity of LNA and GNA

- LNA and GNA are complementary; LNA provides high functional quality while GNA offers high topological quality.
- There is a growing recognition of the need to integrate these approaches to capture both functional and topological aspects of biological networks.
- IGLOO is a recent approach that aims to reconcile LNA and GNA by combining high functional quality seeds from LNA with global expansion strategies from GNA.

### Open Research Problems

- Developing hybrid methods that balance the strengths of LNA and GNA.
- Improving computational efficiency and scalability of alignment algorithms, especially for large networks.
- Establishing standardized metrics and frameworks for comparing network alignment methods.
- Expanding the applicability of biological network alignment methods to other domains, such as social networks or brain connectivity networks.

### Conclusion

- The paper highlights the importance of biological network alignment in advancing our understanding of cellular functioning.
- It calls for the development of new integrative methods that can simultaneously achieve high functional and topological alignment quality.
- Future research should focus on refining these methods, improving their scalability, and exploring their applications in other fields.

The paper provides a comprehensive overview of the current state of biological network alignment, emphasizing the need for integrative approaches to fully leverage the complementary strengths of local and global alignment methods.