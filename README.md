# ComGA
The code of paper: ComGA:Community-Aware Attributed Graph Anomaly Detection (Accepted by WSDM 2022)

## ComGA:Community-Aware Attributed Graph Anomaly Detection.
This a Tensorflow implementation of the ComGA algorithm, which designs a tailored deep graph convolutional network (tGCN) to capture local, global and structure anomalies for anomaly detection on attributed graphs. 
### ABSTRACT:
Graph anomaly detection, here, aims to find rare patterns that are significantly different from other nodes. Attributed graphs containing complex structure and attribute information are ubiquitous in our life scenarios such as bank account transaction graph and paper citation graph. Anomalous nodes on attributed graphs show great difference from others in the perspectives of structure and attributes, and give rise to various types of graph anomalies. In this paper, we investigate three types of graph anomalies: local, global, and structure anomalies. And, graph neural networks (GNNs) based anomaly detection methods attract considerable research interests due to the power of modeling attributed graphs. However, the convolution operation of GNNs aggregates neighbors information to represent nodes, which makes node representations more similar and cannot effectively distinguish between normal and anomalous nodes, thus result in sub-optimal results. To improve the performance of anomaly detection, we propose a novel community-aware attributed graph anomaly detection framework (ComGA). We design a tailored deep graph convolutional network (tGCN) to anomaly detection on attributed graphs. Extensive experiments on eight real-life graph datasets demonstrate the effectiveness of ComGA.
### MOTIVATION:![](https://img-blog.csdnimg.cn/308797cead554dabb9eca794cc27e168.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5aSn56yo54aK44CC44CC44CC,size_17,color_FFFFFF,t_70,g_se,x_16)

The toy example is a Twitter graph with different types of graph anomalies, where the dense links of the graph form two different communities C1 and C2. When considering attribute information in the whole graph, user 1 is global anomaly since its attribute (a2) value is significantly higher than others. When considering attribute information within one community (e.g., C2), user 6 is local anomaly as its attribute (a3) value relatively deviates from other users within C2. When considering structure information across different communities, users 5 and 7 are structure anomaly because they have link with other communities while other users in their community do not have cross-community links.

### The framework of ComGA
![](https://img-blog.csdnimg.cn/d993ddb3045c44eb9272eee8581a464b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5aSn56yo54aK44CC44CC44CC,size_20,color_FFFFFF,t_70,g_se,x_16)

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
#### The complete construction process of injected anomaly graphs
#### Structural anomalies injection: 
The structural anomalies are acquired by perturbing the topological structure of graphs. Concretely, some small cliques composed of originally unrelated nodes are generated as anomalies. The intuition is that in a small clique, a small set of nodes are much more closely linked to each other than average, which can be regarded as a typical structural anomalous situation in real-world graphs. To make the cliques, we first specify the clique size 𝑝 and the number of cliques 𝑞. When generating a clique, we randomly choose 𝑝 nodes from the set of nodes V and make them fully connected. As such, the selected 𝑝 nodes are all marked as structural anomaly nodes. To generate 𝑞 cliques, we repeat the above process for 𝑞 times. Finally, a total of 𝑝 × 𝑞 structural anomalies were injected. According to the size of datasets, we control the number of injected anomalies. We fix 𝑝 = 15 and set 𝑞 to 10, 15, 20, 5, 5, 20 for BlogCatalog, Flickr, ACM, Cora, Citeseer, and Pubmed, respectively.
#### Attribute anomalies injection: 
To guarantee an equal number of anomalies from structural perspective and attribute perspective will be injected into the attributed graph, we first randomly select another m × n nodes as the attribute perturbation candidates. For each selected node i, we randomly pick another k nodes from the data and select the node j whose attributes deviate the most from node i among the k nodes by maximizing the Euclidean
distance ||xi − xj ||2. Afterwards, we then change the attributes xi of node i to xj . In our experiments, we set the value of k to 50.
##### This process can follow these two works from: "Deep Anomaly Detection on Attributed Networks" and "Anomaly Detection on Attributed Networks via Contrastive Self-Supervised Learning"
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
#### If you find that this code is useful for your research, please cite our paper:
        @inproceedings{luo2022comga,
        title={ComGA: Community-Aware Attributed Graph Anomaly Detection},
        author={Luo, Xuexiong and Wu, Jia and Beheshti, Amin and Yang, Jian and Zhang, Xiankun and Wang, Yuan and Xue, Shan},
        booktitle={Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining},
        pages={657--665},
        year={2022}
        }
 



