# ComGA
The code of paper: ComGA:Community-Aware Attributed Graph Anomaly Detection

##ComGA:Community-Aware Attributed Graph Anomaly Detection.
This a Tensorflow implementation of the ComGA algorithm, which designs a tailored deep graph convolutional network (tGCN) to capture local, global and community structure anomalies for anomaly detection on attributed graphs. 
### ABSTRACT:
Anomaly detection on attributed graphs aims to distinguish nodes whose patterns or behaviors deviate significantly from the majority of reference nodes. Recently, some methods utilize reconstruction errors of nodes between learned nodes representations and original attributed graphs to spot anomalous nodes, which achieves state-of-the-art performance. However, two major problems exist: (1) they cannot learn more distinguishable anomalous nodes representations from the majority nodes within the community structure, and brings difficulties to improving the performance of anomaly detection; and (2) it is worth considering that real-world anomalies are various, including local, global and community structure anomalies. To this end, in this paper, we propose a novel community-aware attributed graph anomaly detection framework. We design a tailored deep graph convolutional network (tGCN) which propagates community-specific representation into its corresponding layers of graph convolutional network via multiple gateways. Thus, the tGCN can respect the community structure of graph and effectively learn more distinguishable and anomaly-aware nodes representations for multiple anomalies by deeper tGCN layer. Furthermore, we introduce an autoencoder module to encode and decode modularity matrix of the graph, which can learn high-quality community-specific representation. Then, the joint reconstruction errors consisted of structure reconstruction and attributes reconstruction can be used to analyze anomalous nodes. Extensive experiments on various attributed graphs demonstrate the efficiency of the proposed approach.
### MOTIVATION:![](https://img-blog.csdnimg.cn/20210808231655136.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1b3h1ZXhpb25n,size_16,color_FFFFFF,t_70)

The toy example is a Twitter graph with different types of anomalies users including two different communities users. When only considering attributes a3, user 6 is local anomaly as its attribute (a3) value deviates relatively from the other users within community subgraph (users 7, 8 and 9). Meanwhile, user 1 is global anomaly since its attribute (a2) value is significantly higher than other users within the whole graph. When we focus on community structure information, user 5 is community structure anomaly. Although user 5 attributes are normal within community users 1, 2, 3, and 4, it has strong link with another community users 6, 7, 8 and 9.

### The framework of ComGA
![](https://img-blog.csdnimg.cn/20210808231645495.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1b3h1ZXhpb25n,size_16,color_FFFFFF,t_70)

The framework of ComGA consists of three major modules. For the autoencoder module, we utilize the autoencoder to encode and decode the modularity matrix of graph to obtain the community information of each node. For the tGCN module, we use the topology structure and nodal attributes as the input of the GCN model to capture local and global anomalies information. Simultaneously, we introduce the gateway that propagates the community feature representation of each node in the autoencoder into the feature representations of its corresponding nodes in GCN, which can fuse community structure anomaly feature of each node. For the anomaly detection module, we design the structure decoder and attribute decoder to reconstruct the topology structure and nodal attributes respectively, and compute the ranking of anomalous nodes according to the corresponding anomaly score from the joint reconstruction error.

### Requirement:
    python 3.6.4
    Tensorflow 1.0
    Networkx
    Numpy
    Scipy
    Keras
    Pandas
    The develop tool is PyCharm Community Edition 2019.3.1
 ### Datasets:
     BlogCatalog
     ACM
     Flickr
     notice:these datasets contain anomaly information
### Including:
    For BlogCatalog dataset:
    BlogCatalog.edgelist
    BlogCatalog.mat (BlogCatalog.features)
### Basic Usage:
#### input data
##### BlogCatalog.edgelist: each line contains two connected nodes

    node_1 node_2
    node_2 node_2
      ...
##### BlogCatalog.feature: this file has n lines.

     the n lines are as follows:(each node per line ordered by node id)
    (for node_1) feature_1 feature_2...feature_n
    (for node_1) feature_1 feature_2...feature_n
    ...
#### output: the AUC for anomaly detection
#### Run
     python run.py
#### Anomaly detection result on AUC values
![](https://img-blog.csdnimg.cn/2021080823163462.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1b3h1ZXhpb25n,size_16,color_FFFFFF,t_70)

