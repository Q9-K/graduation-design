import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers
from keras import models
from keras.layers import Layer
from keras.layers.core import Dropout, Lambda
from tensorflow.python.framework import ops
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation, GlobalAveragePooling2D, Softmax
from keras.layers import Flatten, Reshape, TimeDistributed, BatchNormalization, Permute

'''
Model code of MSTGCN.
--------
Model input:  (*, T, V, F)
    T: num_of_timesteps
    V: num_of_vertices
    F: num_of_features
Model output: (*, 5)
'''


################################################################################################
################################################################################################
# Attention Layers

class TemporalAttention(Layer):
    '''
    compute temporal attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''

    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.U_1 = self.add_weight(name='U_1',
                                   shape=(num_of_vertices, 1),
                                   initializer='uniform',
                                   trainable=True)
        self.U_2 = self.add_weight(name='U_2',
                                   shape=(num_of_features, num_of_vertices),
                                   initializer='uniform',
                                   trainable=True)
        self.U_3 = self.add_weight(name='U_3',
                                   shape=(num_of_features,),
                                   initializer='uniform',
                                   trainable=True)
        self.b_e = self.add_weight(name='b_e',
                                   shape=(1, num_of_timesteps, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        self.V_e = self.add_weight(name='V_e',
                                   shape=(num_of_timesteps, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        _, T, V, F = x.shape

        # shape of lhs is (batch_size, V, T)
        lhs = K.dot(tf.transpose(x, perm=[0, 1, 3, 2]), self.U_1)
        lhs = tf.reshape(lhs, [tf.shape(x)[0], T, F])
        lhs = K.dot(lhs, self.U_2)

        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.U_3, tf.transpose(x, perm=[2, 0, 3, 1]))
        rhs = tf.transpose(rhs, perm=[1, 0, 2])

        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)

        S = tf.transpose(K.dot(self.V_e, tf.transpose(K.sigmoid(product + self.b_e), perm=[1, 2, 0])), perm=[2, 0, 1])

        # normalization
        S = S - K.max(S, axis=1, keepdims=True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis=1, keepdims=True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[1])


class SpatialAttention(Layer):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''

    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.W_1 = self.add_weight(name='W_1',
                                   shape=(num_of_timesteps, 1),
                                   initializer='uniform',
                                   trainable=True)
        self.W_2 = self.add_weight(name='W_2',
                                   shape=(num_of_features, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        self.W_3 = self.add_weight(name='W_3',
                                   shape=(num_of_features,),
                                   initializer='uniform',
                                   trainable=True)
        self.b_s = self.add_weight(name='b_s',
                                   shape=(1, num_of_vertices, num_of_vertices),
                                   initializer='uniform',
                                   trainable=True)
        self.V_s = self.add_weight(name='V_s',
                                   shape=(num_of_vertices, num_of_vertices),
                                   initializer='uniform',
                                   trainable=True)
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        _, T, V, F = x.shape

        # shape of lhs is (batch_size, V, T)
        lhs = K.dot(tf.transpose(x, perm=[0, 2, 3, 1]), self.W_1)
        lhs = tf.reshape(lhs, [tf.shape(x)[0], V, F])
        lhs = K.dot(lhs, self.W_2)

        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.W_3, tf.transpose(x, perm=[1, 0, 3, 2]))
        rhs = tf.transpose(rhs, perm=[1, 0, 2])

        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)

        S = tf.transpose(K.dot(self.V_s, tf.transpose(K.sigmoid(product + self.b_s), perm=[1, 2, 0])), perm=[2, 0, 1])

        # normalization
        S = S - K.max(S, axis=1, keepdims=True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis=1, keepdims=True)
        #S_normalized = K.ones_like(S_normalized)/tf.convert_to_tensor(int(V), tf.float32)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[2])


class FeatureAttention(Layer):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_features, num_of_features)
    '''

    def __init__(self, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.W_1 = self.add_weight(name='W_1',
                                   shape=(num_of_timesteps, 1),
                                   initializer='uniform',
                                   trainable=True)
        self.W_2 = self.add_weight(name='W_2',
                                   shape=(num_of_vertices, num_of_timesteps),
                                   initializer='uniform',
                                   trainable=True)
        self.W_3 = self.add_weight(name='W_3',
                                   shape=(num_of_vertices,),
                                   initializer='uniform',
                                   trainable=True)
        self.b_s = self.add_weight(name='b_s',
                                   shape=(1, num_of_features, num_of_features),
                                   initializer='uniform',
                                   trainable=True)
        self.V_s = self.add_weight(name='V_s',
                                   shape=(num_of_features, num_of_features),
                                   initializer='uniform',
                                   trainable=True)
        super(FeatureAttention, self).build(input_shape)

    def call(self, x):
        _, T, V, F = x.shape

        # shape of lhs is (batch_size, F, T)
        lhs = K.dot(tf.transpose(x, perm=[0, 3, 2, 1]), self.W_1)
        lhs = tf.reshape(lhs, [tf.shape(x)[0], F, V])
        lhs = K.dot(lhs, self.W_2)

        # shape of rhs is (batch_size, T, F)
        rhs = K.dot(self.W_3, tf.transpose(x, perm=[1, 0, 2, 3]))
        rhs = tf.transpose(rhs, perm=[1, 0, 2])

        # shape of product is (batch_size, F, F)
        product = K.batch_dot(lhs, rhs)

        S = tf.transpose(K.dot(self.V_s, tf.transpose(K.sigmoid(product + self.b_s), perm=[1, 2, 0])), perm=[2, 0, 1])

        # normalization
        S = S - K.max(S, axis=1, keepdims=True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis=1, keepdims=True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2], input_shape[2])


################################################################################################
################################################################################################
# Adaptive Graph Learning Layer

def diff_loss(diff, S):
    '''
    compute the 1st loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:
        # batch input
        return K.mean(K.sum(K.sum(diff ** 2, axis=3) * S, axis=(1, 2)))
    else:
        return K.sum(K.sum(diff ** 2, axis=2) * S)


def F_norm_loss(S, Falpha):
    '''
    compute the 2nd loss of L_{graph_learning}
    '''
    if len(S.shape) == 4:
        # batch input
        return Falpha * K.sum(K.mean(S ** 2, axis=0))
    else:
        return Falpha * K.sum(S ** 2)


class Graph_Learn(Layer):
    '''
    Graph structure learning (based on the middle time slice)
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''

    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S = tf.convert_to_tensor([[[0.0]]])  # similar to placeholder
        self.diff = tf.convert_to_tensor([[[[0.0]]]])  # similar to placeholder
        super(Graph_Learn, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a = self.add_weight(name='a',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        # add loss L_{graph_learning} in the layer
        self.add_loss(F_norm_loss(self.S, self.alpha))
        self.add_loss(diff_loss(self.diff, self.S))
        super(Graph_Learn, self).build(input_shape)

    def call(self, x):
        _, T, V, F = x.shape
        N = tf.shape(x)[0]

        outputs = []
        diff_tmp = 0
        for time_step in range(T):
            # shape: (N,V,F) use the current slice
            xt = x[:, time_step, :, :]
            # shape: (N,V,V)
            diff = tf.transpose(tf.broadcast_to(xt, [V, N, V, F]), perm=[2, 1, 0, 3]) - xt
            # shape: (N,V,V)
            tmpS = K.exp(K.reshape(K.dot(tf.transpose(K.abs(diff), perm=[1, 0, 2, 3]), self.a), [N, V, V]))
            # normalization
            S = tmpS / tf.transpose(tf.broadcast_to(K.sum(tmpS, axis=1), [V, N, V]), perm=[1, 2, 0])

            diff_tmp += K.abs(diff)
            outputs.append(S)

        outputs = tf.transpose(outputs, perm=[1, 0, 2, 3])
        self.S = K.mean(outputs, axis=0)
        self.diff = K.mean(diff_tmp, axis=0) / tf.convert_to_tensor(int(T), tf.float32)
        return outputs

    def compute_output_shape(self, input_shape):
        # shape: (n, num_of_vertices,num_of_vertices, num_of_vertices)
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[2])


################################################################################################
################################################################################################
# GCN layers

class cheb_conv_with_Att_GL(Layer):
    '''
    K-order chebyshev graph convolution with attention after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             Att (batch_size, num_of_vertices, num_of_vertices),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''

    def __init__(self, num_of_filters, k, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        super(cheb_conv_with_Att_GL, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape, Att_shape, S_shape = input_shape
        _, T, V, F = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, F, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_Att_GL, self).build(input_shape)

    def call(self, x):
        # Input:  [x, Att, S]
        assert isinstance(x, list)
        assert len(x) == 3, 'Cheb_gcn input error'
        x, Att, S = x
        _, T, V, F = x.shape

        S = K.minimum(S, tf.transpose(S, perm=[0, 1, 3, 2]))  # Ensure symmetry

        # GCN
        outputs = []
        for time_step in range(T):
            # shape of x is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            output = K.zeros(shape=(tf.shape(x)[0], V, self.num_of_filters))

            A = S[:, time_step, :, :]
            # Calculating Chebyshev polynomials (let lambda_max=2)
            D = tf.matrix_diag(K.sum(A, axis=1))
            L = D - A
            L_t = L - [tf.eye(int(V))]
            cheb_polynomials = [tf.eye(int(V)), L_t]
            for i in range(2, self.k):
                cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

            for kk in range(self.k):
                T_k = cheb_polynomials[kk]  # shape of T_k is (V, V)
                T_k_with_at = T_k * Att  # shape of T_k_with_at is (batch_size, V, V)
                theta_k = self.Theta[kk]  # shape of theta_k is (F, num_of_filters)

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at, perm=[0, 2, 1]), graph_signal)
                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output, -1))

        return tf.transpose(K.relu(K.concatenate(outputs, axis=-1)), perm=[0, 3, 1, 2])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.num_of_filters)


class cheb_conv_with_Att_static(Layer):
    '''
    K-order chebyshev graph convolution with static graph structure
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             Att (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''

    def __init__(self, num_of_filters, k, cheb_polynomials, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        self.cheb_polynomials = tf.to_float(cheb_polynomials)
        super(cheb_conv_with_Att_static, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape, Att_shape = input_shape
        _, T, V, F = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, F, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_Att_static, self).build(input_shape)

    def call(self, x):
        # Input:  [x, Att]
        assert isinstance(x, list)
        assert len(x) == 2, 'cheb_gcn error'
        x, Att = x
        _, T, V, F = x.shape

        outputs = []
        for time_step in range(T):
            # shape is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            output = K.zeros(shape=(tf.shape(x)[0], V, self.num_of_filters))

            for kk in range(self.k):
                T_k = self.cheb_polynomials[kk]  # shape of T_k is (V, V)
                T_k_with_at = K.dropout(T_k * Att, 0.6)  # shape of T_k_with_at is (batch_size, V, V)
                theta_k = self.Theta[kk]  # shape of theta_k is (F, num_of_filters)

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at, perm=[0, 2, 1]), graph_signal)
                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output, -1))

        return tf.transpose(K.relu(K.concatenate(outputs, axis=-1)), perm=[0, 3, 1, 2])

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.num_of_filters)


################################################################################################
################################################################################################
# Some operations

def reshape_dot(x):
    # Input:  [x,TAtt]
    x, TAtt = x
    return tf.reshape(
        K.batch_dot(
            tf.reshape(tf.transpose(x, perm=[0, 2, 3, 1]),
                       (tf.shape(x)[0], -1, tf.shape(x)[1])), TAtt),
        [-1, x.shape[1], x.shape[2], x.shape[3]]
    )


def reshape_dot_f(x):
    # Input:  [x,FAtt]
    x, FAtt = x
    return tf.reshape(
        K.batch_dot(
            tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]),
                       (tf.shape(x)[0], -1, tf.shape(x)[3])), FAtt),
        [-1, x.shape[1], x.shape[2], x.shape[3]]
    )

'MMD functions'
def compute_kernel(x, y):
    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
    tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
    return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

def mcc_obtain(x, alpha):
    x = x / 4.0
    x = Softmax(axis=1)(x)
    print('x shape_x', x.shape)
    # 相关矩阵 类别x类别 3x3
    # x = tf.reshape(x, (-1, 5, 1)) # !fix DM
    x = tf.reshape(x, (-1, 10, 1))

    print('x reshape_x', x.shape)
    #print('x loss_x', loss_x.shape)
    X_corr = K.batch_dot(x, tf.transpose(x, perm=[0, 2, 1]))
    print('x shape_x_corr0', X_corr.shape)
    #X_corr = X_corr / K.sum(X_corr, axis=1)
    print('x shape_x_corr', X_corr.shape)
    ktrace = tf.linalg.trace(X_corr)
    print('tr ', ktrace.shape)
    ksum = K.sum(X_corr, axis=(1, 2))
    print('sum ', ksum.shape)
    loss1 = (ksum - ktrace) / 10 * alpha # !fix
    print('loss1', loss1.shape)
    loss_x = K.sum(loss1)
    print('loss_x', loss_x.shape)
    #print('x trace shape_x_corr', tf.linalg.trace(X_corr).shape)
    #X_corr = X_corr / K.sum(X_corr)
    #print('x norm shape_x_corr', X_corr.shape)
    return loss_x


class MCCLoss(Layer):
    '''
    MCC loss compute
    --------
    Input:  (batch_size, num_of_classes)
    Output: loss
    '''
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S = tf.convert_to_tensor([[0.0]])
        super(MCCLoss, self).__init__()

    def build(self, input_shape):
        _, num_of_classes = input_shape

        # add loss L_{graph_learning} in the layer
        #self.add_loss(mcc_obtain(self.S, self.alpha))
        super(MCCLoss, self).build(input_shape)

    def call(self, x):
        print('x_shape mcc2', x.shape)
        self.S = x
        return mcc_obtain(self.S, self.alpha)

def mycrossentropy(y_true, y_pred, e=0.1):
    loss1 = K.categorical_crossentropy(y_true, y_pred)
    loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/10, y_pred)   # !fix 4 class
    print('Y_true', y_true)
    # if y_true == [0, 0, 1, 0, 0]:           # !fix dsy
    #     e = 0.3
    # elif y_true == [0, 0, 0, 0, 1]:
    #     e = 0.2
    # else:
    #     e = 0.1
    return (1-e)*loss1 + e*loss2


def LayerNorm(x):
    # do the layer normalization
    relu_x = K.relu(x)
    ln = tf.contrib.layers.layer_norm(relu_x, begin_norm_axis=3)
    return ln


################################################################################################
################################################################################################
# Gradient Reverse Layer

def reverse_gradient(X, hp_lambda):
    """Flips the sign of the incoming gradient during training."""
    num_calls = 1
    try:
        reverse_gradient.num_calls = reverse_gradient.num_calls + 1
    except AttributeError:
        reverse_gradient.num_calls = num_calls
        num_calls = num_calls + 1

    grad_name = "GradientReversal_%d" % reverse_gradient.num_calls

    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y


class GradientReversal(Layer):
    """Layer that flips the sign of gradient during training."""

    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = True
        self.hp_lambda = hp_lambda

    @staticmethod
    def get_output_shape_for(input_shape):
        return input_shape

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_config(self):
        config = {}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



################################################################################################
################################################################################################
# Feature Block

# MSTGCN Block
def MSTGCN_Block(x, k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                 cheb_polynomials, time_conv_kernel, GLalpha, i=0):
    '''
    packaged Spatial-temporal convolution Block
    -------
    x: input data;
    k: k-order cheb GCN
    i: block number
    '''
    '''
    # Feature attention
    print('x shape', x.shape)
    feature_Att = FeatureAttention()(x)
    print('feature_Att shape', feature_Att.shape)
    x_FAtt = Lambda(reshape_dot_f, name='reshape_dot_f' + str(i))([x, feature_Att])
    print('x_FAtt shape', x_FAtt.shape)
    # temporal attention
    '''
    '''
    temporal_Att = TemporalAttention()(x)
    x_TAtt = Lambda(reshape_dot, name='reshape_dot'+str(i))([x, temporal_Att])
    print('temporal_Att shape', temporal_Att.shape)
    print('x_TAtt shape', x_TAtt.shape)
    '''
    # spatial attention
    spatial_Att = SpatialAttention()(x)
    print('spatial_Att shape', spatial_Att.shape)
    # multi-view GCN
    S = Graph_Learn(alpha=GLalpha)(x)
    # print(S)
    S = Dropout(0.5)(S)
    spatial_gcn_GL = cheb_conv_with_Att_GL(num_of_filters=num_of_chev_filters, k=k)([x, spatial_Att, S])
    spatial_gcn_SD = cheb_conv_with_Att_static(num_of_filters=num_of_chev_filters, k=k,
                                               cheb_polynomials=cheb_polynomials)([x, spatial_Att])

    # temporal convolution
    '''
    time_conv_output_GL = layers.Conv2D(
        filters=num_of_time_filters,
        kernel_size=(time_conv_kernel, 1),
        padding='same',
        strides=(1, time_conv_strides))(spatial_gcn_GL)

    time_conv_output_SD = layers.Conv2D(
        filters=num_of_time_filters,
        kernel_size=(time_conv_kernel, 1),
        padding='same',
        strides=(1, time_conv_strides))(spatial_gcn_SD)
    '''
    # LayerNorm
    end_output_GL = Lambda(LayerNorm, name='layer_norm' + str(2 * i))(spatial_gcn_GL)
    end_output_SD = Lambda(LayerNorm, name='layer_norm' + str(2 * i + 1))(spatial_gcn_SD)
    return end_output_GL, end_output_SD


################################################################################################
################################################################################################
# MSTGCN

def build_MSTGCN(k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials,
                 time_conv_kernel, sample_shape, num_block, dense_size, opt, GLalpha,
                 regularizer, dropout, lambda_reversal, loss_movie_p=0.2, num_classes=3, num_domain=13,
                 num_subjects=143):
    # 先做特征提取再做MSTGCN

    padding = 'same'
    time_second = 3
    time_slides = 1
    channels = 16 # !fix
    # channels = 62
    freq = 250
    node_feature = 128 * 4
    movies = 3
    activation = tf.nn.relu
    ######### Input ########
    input_signal = Input(shape=(time_second * freq, 1), name='input_signal')

    ######### CNNs with small filter size at the first layer #########
    cnn0 = Conv1D(kernel_size=25,  # 50, #25 50
                  filters=128,
                  strides=4,  # 6, #3 6
                  kernel_regularizer=keras.regularizers.l2(0.001))
    s = cnn0(input_signal)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn1 = MaxPool1D(pool_size=6, strides=6)
    #cnn1 = Conv1D(kernel_size=6, filters=32, strides=6, padding=padding)
    s = cnn1(s)
    cnn2 = Dropout(0.5)
    s = cnn2(s)
    #s = BatchNormalization()(s)
    #s = Activation(activation=activation)(s)
    cnn3 = Conv1D(kernel_size=25, filters=128, strides=1, padding=padding)
    s = cnn3(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn4 = Conv1D(kernel_size=8, filters=128, strides=1, padding=padding)
    s = cnn4(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    #cnn5 = Conv1D(kernel_size=8, filters=32, strides=1, padding=padding)
    #s = cnn5(s)
    #s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn6 = MaxPool1D(pool_size=8, strides=8)
    s = cnn6(s)
    dr1 = Dropout(0.5)
    s = dr1(s)
    cnn7 = Reshape((int(s.shape[1]) * int(s.shape[2]),))  # Flatten
    s = cnn7(s)

    ######### CNNs with large filter size at the first layer #########
    cnn8 = Conv1D(kernel_size=100,  # 200,
                  filters=128,
                  strides=12,  # 50,
                  kernel_regularizer=keras.regularizers.l2(0.001))
    l = cnn8(input_signal)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn9 = MaxPool1D(pool_size=3, strides=3)
    #cnn9 = Conv1D(kernel_size=3, filters=32, strides=3, padding=padding)
    l = cnn9(l)
    cnn10 = Dropout(0.5)
    l = cnn10(l)
    #l = BatchNormalization()(l)
    #l = Activation(activation=activation)(l)
    cnn11 = Conv1D(kernel_size=18, filters=128, strides=1, padding=padding)
    l = cnn11(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn12 = Conv1D(kernel_size=6, filters=128, strides=1, padding=padding)
    l = cnn12(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    #cnn13 = Conv1D(kernel_size=6, filters=32, strides=1, padding=padding)
    #l = cnn13(l)
    #l = BatchNormalization()(l)
    #l = Activation(activation=activation)(l)
    cnn14 = MaxPool1D(pool_size=16, strides=16)
    l = cnn14(l)
    dr2 = Dropout(0.5)
    l = dr2(l)
    cnn15 = Reshape((int(l.shape[1]) * int(l.shape[2]),))
    l = cnn15(l)
    '''
    cnn16 = Conv1D(kernel_size=10,  # 200,
                  filters=32,
                  strides=3,  # 50,
                  kernel_regularizer=keras.regularizers.l2(0.0001))
    m = cnn16(input_signal)
    m = BatchNormalization()(m)
    m = Activation(activation=activation)(m)
    cnn17 = MaxPool1D(pool_size=3, strides=3)
    m = cnn17(m)
    cnn18 = Dropout(0.5)
    m = cnn18(m)
    cnn19 = Conv1D(kernel_size=10, filters=32, strides=1, padding=padding)
    m = cnn19(m)
    m = BatchNormalization()(m)
    m = Activation(activation=activation)(m)
    cnn24 = MaxPool1D(pool_size=3, strides=3)
    m = cnn24(m)
    cnn20 = Conv1D(kernel_size=8, filters=32, strides=1, padding=padding)
    m = cnn20(m)
    m = BatchNormalization()(m)
    m = Activation(activation=activation)(m)
    cnn25 = MaxPool1D(pool_size=3, strides=3)
    m = cnn25(m)
    cnn21 = Conv1D(kernel_size=8, filters=32, strides=1, padding=padding)
    m = cnn21(m)
    m = BatchNormalization()(m)
    m = Activation(activation=activation)(m)
    cnn22 = MaxPool1D(pool_size=8, strides=8)
    m = cnn22(m)
    dr3 = Dropout(0.5)
    m = dr3(m)
    cnn23 = Reshape((int(m.shape[1]) * int(m.shape[2]),))
    m = cnn23(m)
    '''
    feature = keras.layers.concatenate([s, l])
    # feature = s
    fea_part = Model(input_signal, feature)
    # print('fea_part shape', fea_part.shape)
    ##################################################
    # 按照某种方式接进来数据
    # input = Input(shape=(channels, time_second * freq), name='input_signal')
    # reshape = Reshape((channels, time_second * freq, 1))  # Flatten
    # input_re = reshape(input)
    # fea_all = TimeDistributed(fea_part)(input_re)
    # print('fea_all shape', fea_all.shape)

    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    data_layer = Input(shape=sample_shape, name='Input_Layer')
    val_layer = Input(shape=sample_shape, name='val_Layer')
    print('data_layer_shape', data_layer.shape)
    reshape_0 = Reshape((channels, time_slides, time_second * freq, movies))
    data_temp = reshape_0(data_layer)
    val_temp = reshape_0(val_layer)
    permute_1 = Permute((4, 1, 2, 3))
    reshape_2 = Reshape((movies * time_slides * channels, time_second * freq, 1))
    data_temp = permute_1(data_temp)
    val_temp = permute_1(val_temp)
    print('data_temp shape', data_temp.shape)
    data_temp = reshape_2(data_temp)
    val_temp = reshape_2(val_temp)
    print('data_temp shape', data_temp.shape)
    data_tempinput = TimeDistributed(fea_part)(data_temp)
    val_tempinput = TimeDistributed(fea_part)(val_temp)
    print('data_tempinput shape', data_tempinput.shape)
    # data_layer_rest = reshape_2(data_layer[:, :, :, :, 0])
    # data_layer_2 = reshape_2(data_layer[:, :, :, :, 1])
    # data_layer_6 = reshape_2(data_layer[:, :, :, :, 2])
    # data_restinput = TimeDistributed(fea_part)(data_layer_rest)
    # data_2input = TimeDistributed(fea_part)(data_layer_2)
    # data_6input = TimeDistributed(fea_part)(data_layer_6)
    # print('data_restinput shape', data_restinput.shape)
    # print('data_2 shape', data_2input.shape)
    # print('data_6 shape', data_6input.shape)
    reshape_3 = Reshape((movies, time_slides, channels, node_feature))
    permute_2 = Permute((2, 3, 4, 1))
    reshape_4 = Reshape((time_slides, channels, node_feature * movies))
    #reshape_4 = Reshape((movies*time_slides, channels, node_feature))
    data_tempinput = reshape_3(data_tempinput)
    val_tempinput = reshape_3(val_tempinput)
    print('data_tempinput shape', data_tempinput.shape)
    data_tempinput = permute_2(data_tempinput)
    val_tempinput = permute_2(val_tempinput)
    print('data_tempinput shape', data_tempinput.shape)
    data_realinput = reshape_4(data_tempinput)
    val_realinput = reshape_4(val_tempinput)
    # data_realinput = layers.concatenate([reshape_3(data_restinput), reshape_3(data_2input), reshape_3(data_6input)], 3)
    print('data_realinput shape', data_realinput.shape)
    ##loss1

    # MSTGCN_Block
    block_out_GL, block_out_SD = MSTGCN_Block(data_realinput, k, num_of_chev_filters, num_of_time_filters,
                                              time_conv_strides, cheb_polynomials, time_conv_kernel, GLalpha)
    block_out_GL_V, block_out_SD_V = MSTGCN_Block(val_realinput, k, num_of_chev_filters, num_of_time_filters,
                                              time_conv_strides, cheb_polynomials, time_conv_kernel, GLalpha, 1)
    for i in range(1, num_block):
        block_out_GL, block_out_SD = MSTGCN_Block(block_out_GL, k, num_of_chev_filters, num_of_time_filters,
                                                  time_conv_strides, cheb_polynomials, time_conv_kernel, GLalpha, i)
        block_out_GL_V, block_out_SD_V = MSTGCN_Block(block_out_GL_V, k, num_of_chev_filters, num_of_time_filters,
                                                      time_conv_strides, cheb_polynomials, time_conv_kernel, GLalpha, 1)
    global_average_layer = GlobalAveragePooling2D()
    block_out = layers.concatenate([block_out_GL, block_out_SD])
    block_out_V = layers.concatenate([block_out_GL_V, block_out_SD_V])
    # block_out = block_out_SD
    # block_out = data_realinput
    print('block_out shape', block_out.shape)
    #block_out = global_average_layer(block_out)
    block_out = layers.Flatten()(block_out)
    block_out_V = layers.Flatten()(block_out_V)
    block_out = layers.Flatten()(data_realinput)
    block_out_V = layers.Flatten()(val_realinput)
    print('block_out shape', block_out.shape)
    # dropout
    block_out = layers.Dropout(dropout)(block_out)
    block_out_V = layers.Dropout(dropout)(block_out_V)
    # Global dense layer
    fc0 = layers.Dense(1024)
    fc1 = layers.Dense(512)
    fc2 = layers.Dense(128)
    fc3 = layers.Dense(32)
    fc4 = layers.Dense(16)
    #dense_out = fc0(block_out)
    dense_out = fc1(block_out)
    dense_out = fc2(dense_out)
    dense_out = fc3(dense_out)
    dense_out = fc4(dense_out)
    #dense_out_V = fc0(block_out_V)
    dense_out_V = fc1(block_out_V)
    dense_out_V = fc2(dense_out_V)
    dense_out_V = fc3(dense_out_V)
    dense_out_V = fc4(dense_out_V)
    print('dense_out shape', block_out.shape)
    # softmax classification
    Loss_mmd = compute_mmd(block_out, block_out_V)
    Loss_mmd2 = compute_mmd(layers.Flatten()(data_tempinput), layers.Flatten()(val_tempinput))
    Loss_mmd3 = compute_mmd(dense_out, dense_out_V)
    print('mmd shape', Loss_mmd.shape)
    softmax = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_regularizer=regularizer,
                           name='Label')(dense_out)
    softmax_V = layers.Dense(num_classes,
                           activation='softmax',
                           kernel_regularizer=regularizer,
                           name='Label_V')(dense_out_V)

    print('softmax shape', softmax.shape)

    Loss_MCC = MCCLoss(alpha=1)(softmax_V)
    print('loss_mcc', Loss_MCC)
    print(Loss_MCC.shape)
    # GRL & G_d
    flip_layer = GradientReversal(lambda_reversal)
    G_d_in = flip_layer(block_out)
    # for size in dense_size:
    #G_d_out = layers.Dense(1024)(G_d_in)
    G_d_out = layers.Dense(64)(G_d_in)
    G_d_out = layers.Dense(units=num_domain,
                           activation='softmax',
                           name='Domain')(G_d_out)
    print('G_d_out shape', G_d_out.shape)

    G_d_dist = layers.Dense(64)(G_d_in)
    G_d_dist = layers.Dense(units=num_subjects,
                           activation='softmax',
                           name='Time')(G_d_dist)
    # training model (with GRL & G_d)

    #model = models.Model(inputs=[data_layer, val_layer], outputs=softmax)
    model = models.Model(inputs=[data_layer, val_layer], outputs=[softmax, G_d_out])
    # model = models.Model(inputs=data_layer, outputs=[softmax, softmax_m])
    # model.load_weights(
    #    'C:/Users/dongm/PycharmProjects/MSTGCN-main-new/MSTGCN-main/output/143PRETRAIN/MSTGCN_Best_0.h5', by_name=True, skip_mismatch=True)
    # fine_tune_at = 0
    # print("Number of layers in the model: ", len(model.layers))
    # print("Name of layers in the model: ", model.layers)
    # model.trainable = True
    # Freeze all the layers before the `fine_tune_at` layer
    # for layer in model.layers[:fine_tune_at]:
    #    layer.trainable = False
    #model.compile(
    #    optimizer=opt,
    #    loss=['categorical_crossentropy', 'categorical_crossentropy', lambda y_true,y_pred: y_pred],
    #    #loss='categorical_crossentropy',
    #    loss_weights = [1., 1., 1.],
    #    metrics={'softmax':'accuracy'},
    #)
    #model.add_loss(Loss_MCC*10)
    model.add_loss(Loss_mmd)
    model.add_loss(Loss_mmd2)
    model.add_loss(Loss_mmd3)
    model.compile(
        optimizer=opt,
        #loss=[mycrossentropy, 'categorical_crossentropy', lambda y_true, y_pred: y_pred],
        loss=[mycrossentropy,'categorical_crossentropy'],
        #loss=mycrossentropy,
        #loss_weights=[1., 0.1, 0.001],
        metrics={'Label': 'acc', 'Domain': 'acc'},
        #metrics={'Label': 'acc'},
    )
    # testing model (without GRL & G_d)
    pre_model = models.Model(inputs=data_layer, outputs=softmax)
    pre_model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    #model_TA = models.Model(inputs=[data_layer, val_layer], outputs= [softmax, G_d_out])
    model_TA = models.Model(inputs=[data_layer, val_layer], outputs=softmax)
    #model_TA.add_loss(Loss_MCC)
    model_TA.add_loss(Loss_mmd*10)
    model_TA.add_loss(Loss_mmd2*10)
    model_TA.add_loss(Loss_mmd3*10)

    model_TA.compile(
        optimizer=opt,
        #loss=[mycrossentropy, 'categorical_crossentropy'],
        loss=mycrossentropy,
        #loss_weights=[1., 0.5],
        metrics=['acc'],
    )
    return model, pre_model, model_TA

def build_MSTGCN_test():
    # an example to test
    cheb_k = 3
    num_of_chev_filters = 10
    num_of_time_filters = 10
    time_conv_strides = 1
    time_conv_kernel = 3
    dense_size = np.array([64, 32])
    cheb_polynomials = [np.random.rand(26, 26), np.random.rand(26, 26), np.random.rand(26, 26)]

    model = build_MSTGCN(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials,
                         time_conv_kernel, sample_shape=(5, 26, 9), num_block=1, dense_size=dense_size,
                         opt='adam', useGL=True, GLalpha=0.0001, regularizer=None, dropout=0.0)
    model.summary()
    model.save('MSTGCN_build_test.h5')
    print("save ok")
    return model
