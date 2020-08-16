import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from model import *
from optimizer import *
from preprocessing import construct_feed_dict
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

def get_placeholder():
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        #'matrix': tf.compat.v1.sparse_placeholder(tf.float32,shape=(16484,16484)),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    return placeholders


def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero):
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str=='AnomalyDAE':
        model = AnomalyDAE(placeholders, num_features, num_nodes, features_nonzero)
    else:
        print("[ERROR] no such model name: {}".format(model_str))

    return model


def get_optimizer(model_str, model, placeholders, num_nodes, alpha, eta, theta):
    print("alpha:", alpha)

    opt=None
    if model_str == 'gcn_ae'or model_str=='gcn_can':
        opt = OptimizerAE(preds_attribute=model.attribute_reconstructions,
                          labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                          preds_structure=model.structure_reconstructions,
                          labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=alpha)
    elif model_str=='AnomalyDAE':
        opt = OptimizerDAE(preds_attribute=model.attribute_reconstructions,
                           labels_attribute=tf.sparse_tensor_to_dense(placeholders['features']),
                           preds_structure=model.structure_reconstructions,
                           labels_structure=tf.sparse_tensor_to_dense(placeholders['adj_orig']), alpha=alpha,
                           eta=eta, theta=theta)
    else:
        print("[ERROR] no such model name: {}".format(model_str))

    return opt

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj,B):
    # Construct feed dictionary
    #feed_dict={placeholders['features']: features,model.adj: adj, model.inputs:features,placeholders['adj_orig']: adj}
    feed_dict = construct_feed_dict(model.B,adj_norm, adj_label, features, placeholders,B)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    #print('????????????????')
    _, train_loss, re_loss,kl_loss, loss_stru, loss_attr, rec_error = sess.run([opt.opt_op,
                                         opt.cost,
                                         opt.re_loss,
                                         opt.kl_loss,
                                         opt.structure_cost,
                                         opt.attribute_cost,
                                         opt.reconstruction_errors], feed_dict=feed_dict)

    return train_loss, re_loss,kl_loss,loss_stru, loss_attr, rec_error