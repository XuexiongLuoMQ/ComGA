import tensorflow as tf
import torch.nn.functional as F
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
import scipy.stats
from keras import metrics as me

class OptimizerAE(object):
    def __init__(self, preds_community,labels_community,z_mean,z_arg,preds_attribute, labels_attribute,preds_structure,labels_structure,alpha,eta,theta,num_nodes):
        #self.re_loss = tf.reduce_sum(tf.pow((preds_community - labels_community), 2))
        #self.re_los=F.mse_loss(preds_community, labels_community)
        self.re_loss = num_nodes * me.binary_crossentropy(labels_community,preds_community)
        #-------------------------
        B_attr = labels_attribute * (eta - 1) +1
        diff_attribute = tf.square(tf.subtract(preds_attribute, labels_attribute) * B_attr)
        self.attribute_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_attribute, 1))
        self.attribute_cost = tf.reduce_mean(self.attribute_reconstruction_errors)
        #attribute reconstruction loss
        #self.kl_loss = F.kl_div(z_mean.log(), z_arg, reduction='batchmean')
        #---------------------------
        #diff_attribute = tf.square(preds_attribute - labels_attribute)
        #print(diff_attribute,'ccccccccccccccccsssssssssss')
        #self.attribute_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_attribute, 1))
        #self.reconstruction_errors =  tf.losses.mean_squared_error(labels= labels, predictions=preds)
        #self.attribute_cost = tf.reduce_mean(self.attribute_reconstruction_errors)
        #print(self.attribute_cost,'%%%%%%%%%%%%%%%%%')

        #structure reconstruction loss
        #diff_structure = tf.square(preds_structure - labels_structure)
        B_struc = labels_structure * (theta - 1) + 1
        diff_structure = tf.square(tf.subtract(preds_structure, labels_structure) * B_struc)
        #self.structure_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_structure, 1))
        #self.structure_cost = tf.reduce_mean(self.structure_reconstruction_errors)
        self.structure_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_structure, 1))
        self.structure_cost = tf.reduce_mean(self.structure_reconstruction_errors)
        #------------------
        self.kl_loss =-((0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * z_arg - tf.square(z_mean) -
                                                                    tf.square(tf.exp(z_arg)), 1)))
        #self.kl_loss = scipy.stats.entropy(z_mean, z_arg)
        #self.kl_loss=F.kl_div(z_mean.log(), z_arg, reduction='batchmean')
        self.reconstruction_errors = tf.multiply(alpha, self.attribute_reconstruction_errors)+ tf.multiply(1-alpha, self.structure_reconstruction_errors)
        #print(self.reconstruction_errors,'{{{{{{{{{{{{{{{{{')
        self.cost = self.re_loss +0.1*self.kl_loss + alpha * self.attribute_cost + (1-alpha) * self.structure_cost
        #print(self.cost,'|||||||||||||||||||||')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        #print(self.cost,'ZZZZZZZZZZZZZZZZZZZZ')
        #print('zzzzzzzzzzzzzzzzzzzzz')
        # self.grads_vars = self.optimizer.compute_gradients(self.cost)


class OptimizerDAE(object):
    def __init__(self, preds_attribute, labels_attribute, preds_structure, labels_structure,
                 alpha, eta, theta):

        self.preds_attribute = preds_attribute
        self.labels_attribute = labels_attribute
        #attr_2nd_loss = tf.reduce_sum(tf.pow((self.attr_hidden[-1] - self.S) * attr_B, 2))
        # attribute reconstruction loss
        B_attr = labels_attribute * (eta - 1) + 1
        diff_attribute = tf.square(tf.subtract(preds_attribute, labels_attribute)*B_attr)
        self.attribute_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_attribute, 1))
        self.attribute_cost = tf.reduce_mean(self.attribute_reconstruction_errors)

        # structure reconstruction loss
        B_struc = labels_structure * (theta - 1) + 1
        diff_structure = tf.square(tf.subtract(preds_structure, labels_structure)*B_struc)
        self.structure_reconstruction_errors = tf.sqrt(tf.reduce_sum(diff_structure, 1))
        self.structure_cost = tf.reduce_mean(self.structure_reconstruction_errors)

        self.reconstruction_errors = tf.multiply(alpha, self.attribute_reconstruction_errors) \
                                     + tf.multiply(1-alpha, self.structure_reconstruction_errors)

        self.cost = alpha * self.attribute_cost + (1-alpha) * self.structure_cost

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        # self.grads_vars = self.optimizer.compute_gradients(self.cost)
