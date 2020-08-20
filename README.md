# CEAD
The code of paper: Community enhanced anomaly detection on attributed network

## CEAD:Community Enhanced Anomaly Detection on Attributed Networks
![](https://img-blog.csdnimg.cn/20200820155625522.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1b3h1ZXhpb25n,size_16,color_FFFFFF,t_70)

This a Tensorflow implementation of the CEAD algorithm, which utilizes the captured community structure information of network to alleviate the over-smoothing node representation for anomaly detection. 
### ABSTRACT:
Anomaly detection on attributed networks aims to distinguish nodes whose patterns or behaviors deviate significantly from the majority of reference nodes. Recently, anomaly detection has attracted a surge of research attention due to its broad real-world applications such as suspicious account detection in social media, abuse monitoring in healthcare systems and financial fraud monitoring. Anomaly detection methods with graph neural networks, as the mainstream techniques, achieve state-of-the-art performance. However, when working on attributed networks, they are easily encountered with over-smoothing problem in respect of node representation, which makes the anomalous nodes less distinguishable from the majority nodes within the community, and brings difficulties to improving the performance of anomaly detection. Besides, how to extract community structure information of the network to
spot anomalous nodes is another tough challenge. 
To this end, in this paper, we propose a novel Community Enhanced Anomaly Detection framework on attributed networks(CEAD). We utilize the captured community structure information of network to alleviate the problem of over-smoothing of node representation for anomaly detection. Specifically, to make the anomalous nodes more distinguishable and anomaly-aware node representation easier to be learned, we design a tailored graph convolutional network (tGCN) module which propagates community-specific representation into its corresponding layers of tGCN via multiple gateways. Thus, the tGCN module can respect the community structure of network and effectively alleviate over-smoothing of node representation. To learn high-quality community-specific
representation, we introduce an autoencoder module to encode and decode the modularity matrix of network. Then, the joint reconstruction errors consisted of the structure reconstruction and attributes reconstruction can be analyzed anomalous nodes. Extensive experiments on various attributed networks demonstrate the efficiency of the proposed approach.
### MOTIVATION:![](https://img-blog.csdnimg.cn/20200820155600527.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1b3h1ZXhpb25n,size_16,color_FFFFFF,t_70)
Node 4 belonging to community 2 connects nodes belonging to community 1, and its attributes are different from other nodes’ attributes in community 2. Thus, node 4 is the abnormal node. Conventional methods based on graph neural networks are bothered by the over-smoothing of node representation, and the representations of nodes are not distinguishable. Thus, when utilizing the joint reconstruction errors on structure reconstruction and attributes
reconstruction to detect anomalous nodes, the node 2 with complex structure relations will be higher reconstruction errors than other nodes and mistaken as an abnormal node. In comparison, our method utilizes the community structure information of network to alleviate over-smoothing of node representation, highlights the anomaly-aware node representation, and thus can detect the abnormal node 4.
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
![](https://img-blog.csdnimg.cn/20200820155658321.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2x1b3h1ZXhpb25n,size_16,color_FFFFFF,t_70)

