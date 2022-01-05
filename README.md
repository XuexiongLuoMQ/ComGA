# ComGA
The code of paper: ComGA:Community-Aware Attributed Graph Anomaly Detection (Accepted by WSDM 2022)

##ComGA:Community-Aware Attributed Graph Anomaly Detection.
This a Tensorflow implementation of the ComGA algorithm, which designs a tailored deep graph convolutional network (tGCN) to capture local, global and structure anomalies for anomaly detection on attributed graphs. 
### ABSTRACT:
Graph anomaly detection, here, aims to find rare patterns that are significantly different from other nodes. Attributed graphs containing complex structure and attribute information are ubiquitous in our life scenarios such as bank account transaction graph and paper citation graph. Anomalous nodes on attributed graphs show great difference from others in the perspectives of structure and attributes, and give rise to various types of graph anomalies. In this paper, we investigate three types of graph anomalies: local, global, and structure anomalies. And, graph neural networks (GNNs) based anomaly detection methods attract considerable research interests due to the power of modeling attributed graphs. However, the convolution operation of GNNs aggregates neighbors information to represent nodes, which makes node representations more similar and cannot effectively distinguish between normal and anomalous nodes, thus result in sub-optimal results. To improve the performance of anomaly detection, we propose a novel community-aware attributed graph anomaly detection framework (ComGA). We design a tailored deep graph convolutional network (tGCN) to anomaly detection on attributed graphs. Extensive experiments on eight real-life graph datasets demonstrate the effectiveness of ComGA.
### MOTIVATION:![](https://img-blog.csdnimg.cn/20210808231655136.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1b3h1ZXhpb25n,size_16,color_FFFFFF,t_70)

The toy example is a Twitter graph with different types of graph anomalies, where the dense links of the graph form two different communities C1 and C2. When considering attribute information in the whole graph, user 1 is global anomaly since its attribute (a2) value is significantly higher than others. When considering attribute information within one community (e.g., C2), user 6 is local anomaly as its attribute (a3) value relatively deviates from other users within C2. When considering structure information across different communities, users 5 and 7 are structure anomaly because they have link with other communities while other users in their community do not have cross-community links.

### The framework of ComGA
![](https://img-blog.csdnimg.cn/20210808231645495.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1b3h1ZXhpb25n,size_16,color_FFFFFF,t_70)

The framework of ComGA consists of three major modules. For the community detection module, we utilize autoencoder to encode and decode modularity matrix of the graph to obtain community-specific representation of each node. For the tGCN module, we use the topology structure and nodal attributes as the input of GCN model, and aggregate neighbors information to capture local and global anomalies by GCN layer. Simultaneously, we introduce the gateway that propagates community-specific representation of each node in the autoencoder into the feature representations of its corresponding nodes in GCN model, which can fuse structure anomaly feature of each node and learn anomaly-aware node representations. For the anomaly detection module, we design structure decoder and attribute decoder to reconstruct the topology structure and nodal attributes from anomaly-aware node representations at the output of tGCN, respectively, and rank these anomalous nodes according to the corresponding anomaly score from the joint reconstruction errors.

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

