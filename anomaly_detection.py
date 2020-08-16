from __future__ import division
from __future__ import print_function
import os
import numpy as np
import networkx as nx
from Graph import Graph
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

from constructor import get_placeholder, update
from input_data import format_data
from sklearn.metrics import roc_auc_score
from model import *
from optimizer import *
from layers import *
# Settings
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

def precision_AT_K(actual, predicted, k, num_anomaly):
    act_set = np.array(actual[:k])
    pred_set = np.array(predicted[:k])
    ll = act_set & pred_set
    tt = np.where(ll == 1)[0]
    prec = len(tt) / float(k)
    rec = len(tt) / float(num_anomaly)
    return round(prec, 4), round(rec, 4)

class AnomalyDetectionRunner():
    def __init__(self, settings):
        self.data_name = settings['data_name']
        self.iteration = settings['iterations']
        self.model = settings['model']
        self.decoder_act = settings['decoder_act']
        self.detection_method = settings['detection_method']
        self.at=settings['baln']

    def erun(self, writer):
        model_str = self.model
        # load data
        feas = format_data(self.data_name)

        print("feature number: {}".format(feas['num_features']))
        # Define placeholders
        placeholders = get_placeholder()

        num_features = feas['num_features']
        features_nonzero = feas['features_nonzero']
        num_nodes = feas['num_nodes']
        adj = feas['adj']
        m = 171743#239738 flikr #71980 acm#171743 blog
        my_graph = Graph()
        my_graph.load_graph_from_weighted_edgelist('./data/BlogCatalog/BlogCatalog.edgelist')
        if self.detection_method == 'infomap':
            my_graph.infomap()
        elif self.detection_method == 'lpa':
            my_graph.lpa()
        x_1, y_1 = my_graph.output_data()
        print(x_1.shape,'!!!!@@@@@@@@@@@@@@@@@@@@')
        k1 = np.sum(x_1, axis=1)
        k2 = k1.reshape(k1.shape[0], 1)
        k1k2 = k1 * k2
        Eij = k1k2 / (2 * m)
        B =np.array(x_1 - Eij)
        #print(B.shape,'NNNNNNNNNNNNNNNNNNNNNNN')
        #-----------------------------------------------------read_graph(folder + 'citeseer.edgelist')
        if model_str == 'Dominant':
            model = GCNModelAE(placeholders,num_features,num_nodes,adj,features_nonzero,2000,500,128,256,128,self.at)
            opt = OptimizerAE(preds_community=model.community_reconstructions,
                              labels_community=model.B,
                              z_mean=model.z,
                              z_arg=model.z_a,
                            preds_attribute=model.attribute_reconstructions,
                              labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                              preds_structure=model.structure_reconstructions,
                              labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']),
                              alpha=FLAGS.alpha,
                              eta=FLAGS.eta, theta=FLAGS.theta,num_nodes=num_nodes)

        elif model_str == 'AnomalyDAE':
            model = AnomalyDAE(placeholders, num_features, num_nodes, features_nonzero, self.decoder_act)
            opt = OptimizerDAE(preds_attribute=model.attribute_reconstructions,
                               labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                               preds_structure=model.structure_reconstructions,
                               labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=FLAGS.alpha,
                               eta=FLAGS.eta, theta=FLAGS.theta)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        tf.reset_default_graph()
        AVER_auc=0
        # Train model
        for epoch in range(1, self.iteration+1):

            train_loss, re_loss,kl_loss,loss_stru, loss_attr, rec_error = update(model, opt, sess,
                                                                  feas['adj_norm'],
                                                                  feas['adj_label'],
                                                                  feas['features'],
                                                                  placeholders, feas['adj'],B )

            if epoch % 1 == 0:
                y_true = [label[0] for label in feas['labels']]
                #print(y_true,'111111SSSSSSSSSSSSS')
                auc=0
                try:
                    scores = np.array(rec_error)
                    scores = (scores - np.min(scores)) / (
                            np.max(scores) - np.min(scores))
                    #print(scores,'2222222222##########')
                    auc = roc_auc_score(y_true, scores)
                    AVER_auc = AVER_auc + auc
                except Exception:
                    print("[ERROR] for auc calculation!!!")

                print("Epoch:", '%04d' % (epoch),
                      "AUC={:.5f}".format(round(auc,4)),
                      #"train_loss={:.5f}".format(train_loss),
                      #"re_loss={:.5f}".format(re_loss),
                      "kl_loss={:.5f}".format(kl_loss),
                      "loss_struc={:.5f}".format(loss_stru),
                      "loss_attr={:.5f}".format(loss_attr))

                #writer.add_scalar('loss_total', train_loss, epoch)
                #writer.add_scalar('loss_re', re_loss, epoch)
                writer.add_scalar('loss_kl', kl_loss, epoch)
                writer.add_scalar('loss_struc', loss_stru, epoch)
                writer.add_scalar('loss_attr', loss_attr, epoch)
                writer.add_scalar('auc', auc, epoch)
        Aver = AVER_auc / self.iteration
        print(Aver, 'XXXXXXXXXXXXXXXXXXXXXXX')



