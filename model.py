from layers import *
import tensorflow as tf
# from torch.nn import Linear
# import torch.nn.functional as F
from keras.layers import Input, Dropout
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
from keras.layers import Dense as de


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders,num_features,num_nodes,adj,features_nonzero, n_enc_1, n_enc_2, n_enc_3, n_enc_4, n_e,at, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)
        self.B = Input(shape=(num_nodes,), dtype='float32', name='matrix')
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.n_samples=num_nodes
        self.ad=adj
        self.n_enc_1 = n_enc_1#5000
        self.n_enc_2 = n_enc_2#2000
        self.n_enc_3 = n_enc_3#500#512
        self.n_enc_4 = n_enc_4#256
        self.n_e = n_e#128
        self.at=at

        self.build()

    def _build(self):
        self.hidden1 =Dense(input_dim=self.n_samples,
              output_dim=self.n_enc_1,
              act=tf.nn.relu,
              sparse_inputs=False,
              dropout=self.dropout)(self.B)
        self.hidden2 = Dense(input_dim=self.n_enc_1,
                             output_dim=self.n_enc_2,
                              act=tf.nn.relu,
                              sparse_inputs=False,
                              dropout=self.dropout)(self.hidden1)
        self.z_a= Dense(input_dim=self.n_enc_2,
                             output_dim=self.n_enc_3,
                             act=tf.nn.relu,
                             sparse_inputs=False,
                             dropout=self.dropout)(self.hidden2)
        # self.z_a = Dense(input_dim=self.n_enc_3,
        #                       output_dim=self.n_e,
        #                       act=tf.nn.relu,
        #                       sparse_inputs=False,
        #                       dropout=self.dropout)(self.hidden3)
#-----------------------------------------------------------------------------------------
        self.enc1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                            output_dim=self.n_enc_1,
                                            adj=self.adj,
                                            features_nonzero=self.features_nonzero,
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging)(self.inputs)

        self.enc2 = GraphConvolution(input_dim=self.n_enc_1,
                                           output_dim=self.n_enc_2,
                                           adj=self.adj,
                                           act=tf.nn.relu,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.enc1+self.hidden1)
        #print(self.enc2,'BBBBBBBBBBBBBBBBBBb')
        self.enc3= GraphConvolution(input_dim=self.n_enc_2,
                                     output_dim=self.n_enc_3,
                                     adj=self.adj,
                                     act=tf.nn.relu,
                                     dropout=self.dropout,
                                     logging=self.logging)(self.enc2 + self.hidden2)
        self.z = GraphConvolution(input_dim=self.n_enc_3,
                                     output_dim=self.n_enc_3,
                                     adj=self.adj,
                                     act=tf.nn.relu,
                                     dropout=self.dropout,
                                     logging=self.logging)(self.enc3 + self.z_a)
        # self.z= GraphConvolution(input_dim=self.n_e,
        #                              output_dim=self.n_e,
        #                              adj=self.adj,
        #                              act=tf.nn.relu,
        #                              dropout=self.dropout,
        #                              logging=self.logging)(self.enc4 + self.z_a)
        # self.z_s_log = GraphConvolution(input_dim=self.n_z,
        #                             output_dim=self.n_e,
        #                             adj=self.adj,
        #                             act=tf.nn.relu,
        #                             dropout=self.dropout,
        #                             logging=self.logging)(self.enc3 + z_a)
        # z = self.z_s_mean + tf.random_normal([self.n_samples, self.n_e]) * tf.exp(self.z_s_log)

        #------------------------_decoder1------------------------------
        #社区------------------------------------------
        self.se1 = Dense(input_dim=self.n_enc_3,
                         output_dim=self.n_enc_2,
                         act=tf.nn.relu,
                         sparse_inputs=False,
                         dropout=self.dropout)(self.z_a)
        print(self.se1, '!!!!!!!!!!!!!')
        self.se2 = Dense(input_dim=self.n_enc_2,
                         output_dim=self.n_enc_1,
                         act=tf.nn.relu,
                         sparse_inputs=False,
                         dropout=self.dropout)(self.se1)
        print(self.se2, 'CCCDDDDDDDFFFFFFFFF')
        # self.se3 = Dense(input_dim=self.n_enc_2,
        #                  output_dim=self.n_enc_1,
        #                  act=tf.nn.relu,
        #                  sparse_inputs=False,
        #                  dropout=self.dropout)(self.se2)

        self.seu = Dense(input_dim=self.n_enc_1,
                       output_dim=self.n_samples,
                       act=tf.nn.sigmoid,
                       sparse_inputs=False,
                       dropout=self.dropout)(self.se2)
        self.community_reconstructions = self.seu
        #-------------------------------
        self.att_decoder_layer1 = GraphConvolution(input_dim=self.n_enc_3,
                                                         output_dim=self.n_enc_3,
                                                         adj=self.adj,
                                                         act=tf.nn.relu,
                                                         dropout=self.dropout,
                                                         logging=self.logging)(self.z)
        self.att_decoder_layer2 = GraphConvolution(input_dim=self.n_enc_3,
                                                         output_dim=self.n_enc_2,
                                                         adj=self.adj,
                                                         act=tf.nn.relu,
                                                         dropout=self.dropout,
                                                         logging=self.logging)(self.att_decoder_layer1)
        self.att_decoder_layer3 = GraphConvolution(input_dim=self.n_enc_2,
                                                         output_dim=self.n_enc_1,
                                                         adj=self.adj,
                                                         act=tf.nn.relu,
                                                         dropout=self.dropout,
                                                         logging=self.logging)(self.att_decoder_layer2)
        # self.att_decoder_layer4 = GraphConvolution(input_dim=self.n_enc_2,
        #                                                  output_dim=self.n_enc_1,
        #                                                  adj=self.adj,
        #                                                  act=tf.nn.relu,
        #                                                  dropout=self.dropout,
        #                                                  logging=self.logging)(self.att_decoder_layer3)
        self.att_decoder_layer5 = GraphConvolution(input_dim=self.n_enc_1,
                                                   output_dim=self.input_dim,
                                                   adj=self.adj,
                                                   act=tf.nn.relu,
                                                   dropout=self.dropout,
                                                   logging=self.logging)(self.att_decoder_layer3)
        self.attribute_reconstructions = self.att_decoder_layer5
    #----------------------structure reconstrcution-------------
        self.z_st = InnerProductDecoder(input_dim=self.n_enc_1,
                                                     act=tf.nn.sigmoid,
                                                     logging=self.logging)(self.att_decoder_layer3)
        self.structure_reconstructions=self.z_st

class AnomalyDAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero,
                 decoder_act=[tf.nn.sigmoid, tf.nn.sigmoid], **kwargs):
        super(AnomalyDAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.decoder_act = decoder_act
        self.build()

    def _build(self):
        self.hidden1 = Dense(input_dim=self.input_dim,
                             output_dim=FLAGS.hidden1,
                             act=tf.nn.relu,
                             sparse_inputs=True,
                             dropout=self.dropout)(self.inputs)

        self.hidden1 = tf.expand_dims(self.hidden1, 1)
        attns = []
        k=1
        for _ in range(k):
            attns.append(NodeAttention(bias_mat=self.adj, nb_nodes=self.n_samples,
                                       # act=tf.nn.relu,
                                       act=lambda x: x,
                                       out_sz=FLAGS.hidden2//k)(self.hidden1))

        self.embeddings_s = tf.concat(attns, axis=-1)[0]
        print(self.embeddings_s,'aaaaaaaaaaaaa')

        self.hidden2 = Dense(input_dim=self.n_samples,
                             output_dim=FLAGS.hidden1,
                             act=tf.nn.relu,
                             sparse_inputs=True,
                             dropout=self.dropout)(tf.sparse_transpose(self.inputs))

        self.embeddings_a = Dense(input_dim=FLAGS.hidden1,
                              output_dim=FLAGS.hidden2,
                              act=lambda x: x,
                              # act=tf.nn.relu,
                              dropout=self.dropout)(self.hidden2)
        print("FLAGS.hidden2,",FLAGS.hidden2)
        print(self.embeddings_a,'ssssssssssssssss')

        self.structure_reconstructions, self.attribute_reconstructions\
            = InnerDecoder(input_dim=FLAGS.hidden2,
                                            act=self.decoder_act,
                                            logging=self.logging)((self.embeddings_s, self.embeddings_a))
        print(self.structure_reconstructions,'dddddddddddd')
        print(self.attribute_reconstructions,'eeeeeeeeeeeeeee')

