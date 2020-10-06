import tensorflow as tf
import numpy as np
from tensorflow import keras
import xgboost as xgb
from xgboost import XGBClassifier

# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    with tf.name_scope('dmat'):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        return D

# Implemented by Tobias Loesche
# this function gets an uncorrected delta_phi_squared distance matrix and computes and returns the correct one
def correct_phi(D_phi):
    dmask = tf.cast(tf.greater(D_phi, np.pi**2), dtype='float32')
    #set points where delta_phi**2 > pi**2 to zero
    current_D_phi = tf.multiply(D_phi, tf.cast(tf.equal(dmask, 0), dtype='float32'))
        
    corr_D_phi = D_phi-(4*np.pi*tf.sqrt(D_phi))+(4*(np.pi**2))
    corr_D_phi_full = tf.multiply(corr_D_phi, dmask)
        
    final_D_phi = tf.add(corr_D_phi_full, current_D_phi)
    return final_D_phi

def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    with tf.name_scope('knn'):
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return tf.gather_nd(features, indices)


def edge_conv(points, features, num_points, K, channels, with_bn=True, activation='relu', pooling='average', name='edgeconv', contains_angle=False):
    """EdgeConv
    Args:
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """

    with tf.name_scope('edgeconv'):

        # distance

        # if distance contains angular variable, we need to make sure the distances are calculated correctly:
        if contains_angle:
            # calculate distance for non-angular variable
            D_eta = batch_distance_matrix_general(points[:, :, 0:1], points[:, :, 0:1])
            # calculate distance for angular variable
            # TODO: implement feature that one can choose at which points in input angular variables are used
            D_phi = batch_distance_matrix_general(points[:, :, 1:2], points[:, :, 1:2])
            D_phi_corrected = correct_phi(D_phi)
            
            D = tf.add(D_eta, D_phi_corrected)
            #Tobias:
            #fun_D = tf.add(D_eta, D_phi_corrected)
            #D = tf.identity(fun_D, name='D_matrix_{}'.format(index)) # give tensor a name for better identification
        else:
            D = batch_distance_matrix_general(points, points)  # (N, P, P)

            #Tobias:
            #fun_D = batch_distance_matrix_general(points, points)  # (N, P, P)
            #D = tf.identity(fun_D, name='D_matrix_{}'.format(index)) # give tensor a name for better identification

        #D = batch_distance_matrix_general(points, points)  # (N, P, P)
        _, indices = tf.nn.top_k(-D, k=K + 1)  # (N, P, K+1)
        indices = indices[:, :, 1:]  # (N, P, K)

        fts = features
        knn_fts = knn(num_points, K, indices, fts)  # (N, P, K, C)
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, K, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

        x = knn_fts
        for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (name, idx))(x)
            if with_bn:
                x = keras.layers.BatchNormalization(name='%s_bn%d' % (name, idx))(x)
            if activation:
                x = keras.layers.Activation(activation, name='%s_act%d' % (name, idx))(x)

        if pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')

        # shortcut
        sc = keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % name)(tf.expand_dims(features, axis=2))
        if with_bn:
            sc = keras.layers.BatchNormalization(name='%s_sc_bn' % name)(sc)
        sc = tf.squeeze(sc, axis=2)

        if activation:
            return keras.layers.Activation(activation, name='%s_sc_act' % name)(sc + fts)  # (N, P, C')
        else:
            return sc + fts


def _particle_net_base(points, features=None, mask=None, setting=None, name='particle_net', contains_angle=False):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optinal

    with tf.name_scope(name):
        if features is None:
            features = points

        if mask is not None:
            #Important! Our mask value is pt, that is padded as -1 for empty points.
            #Instead of comparing if mask is equal to 0, compare if it's larger than 0.
            mask = tf.cast(tf.greater(mask, 0), dtype='float32')  # 1 if valid
            coord_shift = tf.multiply(999., tf.cast(tf.less_equal(mask, 0), dtype='float32'))  # make non-valid positions to 99

        fts = tf.squeeze(keras.layers.BatchNormalization(name='%s_fts_bn' % name)(tf.expand_dims(features, axis=2)), axis=2)
        for layer_idx, layer_param in enumerate(setting.conv_params):
            K, channels = layer_param
            pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
            if layer_idx != 0 and contains_angle:
                contains_angle = False

            fts = edge_conv(pts, fts, setting.num_points, K, channels, with_bn=True, activation='relu',
                            pooling=setting.conv_pooling, name='%s_%s%d' % (name, 'EdgeConv', layer_idx),contains_angle=contains_angle)

        if mask is not None:
            fts = tf.multiply(fts, mask)

        pool = tf.reduce_mean(fts, axis=1)  # (N, C)

        if setting.fc_params is not None:
            x = pool
            for layer_idx, layer_param in enumerate(setting.fc_params):
                units, drop_rate = layer_param
                x = keras.layers.Dense(units, activation='relu')(x)
                if drop_rate is not None and drop_rate > 0:
                    x = keras.layers.Dropout(drop_rate)(x)
            out = keras.layers.Dense(setting.num_class, activation='softmax')(x)
            return out  # (N, num_classes)
        else:
            return pool

def _particle_net_base_jet(points, features=None, mask=None, jetvars=None, setting=None, name='particle_net_jet', contains_angle=False):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optinal

    with tf.name_scope(name):
        if features is None:
            features = points

        if jetvars is not None:
            evt = keras.layers.Dense(64, activation='relu', name="L64")(jetvars)
            evt = keras.layers.Dropout(0.2, name="DO64")(evt)
            evt = keras.layers.Dense(32, activation='relu', name="L32")(evt)
            evt = keras.layers.Dropout(0.2, name="DO32")(evt)
            evt = keras.layers.Dense(16, activation='relu', name="L16")(evt)
            evt = keras.layers.Dropout(0.2, name="DO16")(evt)
            evt = keras.layers.Dense(8, activation='relu', name="L8")(evt)
            evt = keras.layers.Dropout(0.2, name="DO8")(evt)

        if mask is not None:
            #Important! Our mask value is pt, that is padded as -1 for empty points.
            #Instead of comparing if mask is equal to 0, compare if it's larger than 0.
            mask = tf.cast(tf.greater(mask, 0), dtype='float32')  # 1 if valid
            coord_shift = tf.multiply(999., tf.cast(tf.less_equal(mask, 0), dtype='float32'))  # make non-valid positions to 99

        fts = tf.squeeze(keras.layers.BatchNormalization(name='%s_fts_bn' % name)(tf.expand_dims(features, axis=2)), axis=2)
        for layer_idx, layer_param in enumerate(setting.conv_params):
            K, channels = layer_param
            pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)
            if layer_idx != 0 and contains_angle:
                contains_angle = False

            fts = edge_conv(pts, fts, setting.num_points, K, channels, with_bn=True, activation='relu',
                            pooling=setting.conv_pooling, name='%s_%s%d' % (name, 'EdgeConv', layer_idx),contains_angle=contains_angle)

        if mask is not None:
            fts = tf.multiply(fts, mask)

        pool = tf.reduce_mean(fts, axis=1)  # (N, C)

        if jetvars is not None:
            pool = tf.keras.layers.Concatenate(axis=-1)([pool, evt])

        if setting.fc_params is not None:
            x = pool
            for layer_idx, layer_param in enumerate(setting.fc_params):
                units, drop_rate = layer_param
                x = keras.layers.Dense(units, activation='relu')(x)
                if drop_rate is not None and drop_rate > 0:
                    x = keras.layers.Dropout(drop_rate)(x)
            out = keras.layers.Dense(setting.num_class, activation='softmax')(x)
            return out  # (N, num_classes)
        else:
            return pool


class _DotDict:
    pass


def get_particle_net(num_classes, input_shapes, contains_angle=False):
    r"""ParticleNet model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = 'average'
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(256, 0.1)]
    setting.num_points = input_shapes['points'][0]

    points = keras.Input(name='points', shape=input_shapes['points'])
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
    outputs = _particle_net_base(points, features, mask, setting, name='ParticleNet', contains_angle=contains_angle)

    return keras.Model(inputs=[points, features, mask], outputs=outputs, name='ParticleNet')

def get_particle_net_jet(num_classes, input_shapes, contains_angle=False):
    r"""ParticleNet model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = 'average'
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(256, 0.1)]
    setting.num_points = input_shapes['points'][0]

    points = keras.Input(name='points', shape=input_shapes['points'])
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
    jetvars = keras.Input(name='jetvars', shape=input_shapes['jetvars']) if 'jetvars' in input_shapes else None

    outputs = _particle_net_base_jet(points, features, mask, jetvars, setting, name='ParticleNetJet', contains_angle=contains_angle)

    inputs = [points]
    if 'features' in input_shapes:
        inputs.append(features)
    if 'mask' in input_shapes:
        inputs.append(mask)
    if 'jetvars' in input_shapes:
        inputs.append(jetvars)

    return keras.Model(inputs=inputs, outputs=outputs, name='ParticleNetJet')


def get_particle_net_lite(num_classes, input_shapes, contains_angle=False):
    r"""ParticleNet-Lite model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (7, (32, 32, 32)),
        (7, (64, 64, 64)),
        ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = 'average'
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(128, 0.1)]
    setting.num_points = input_shapes['points'][0]

    points = keras.Input(name='points', shape=input_shapes['points'])
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
    outputs = _particle_net_base(points, features, mask, setting, name='ParticleNetLite', contains_angle=contains_angle)

    return keras.Model(inputs=[points, features, mask], outputs=outputs, name='ParticleNetLite')

def get_FCN_jets(num_classes, input_shapes):

    #original:
    #model 0
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(16, input_shape = input_shapes, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.2))#0.3
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))

    #more complex:
    #model 1
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(32, input_shape = input_shapes))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.ReLU())
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(16))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.ReLU())
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(8))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.ReLU())
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))

    #model 2, accuracy of training drops for some reasons...
    #dropout at 0.2
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, activation='relu', name='input'))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(32, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(16, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(8, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(num_classes, activation='softmax', name='output'))

    #model 2, accuracy of training drops for some reasons...
    #dropout at 0.1
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, activation='relu', name='input'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(32, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(16, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(8, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(num_classes, activation='softmax', name='output'))

    #ReLU and layer norm
    #dropout at 0.2
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, name='input'))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.ReLU())
    ##model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(32))

    ##model.add(keras.layers.Dense(32, input_shape = input_shapes, name='input'))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.ReLU())
    ##model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(16))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.ReLU())
    ##model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(8))
    ##model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.ReLU())
    ##model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(num_classes, activation='softmax', name='output'))


    #model 2, accuracy of training drops for some reasons...
    #Leaky relu
    #dropout at 0.2
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, name='input'))
    #model.add(keras.layers.LeakyReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(32))
    #model.add(keras.layers.LeakyReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(16))
    #model.add(keras.layers.LeakyReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(8))
    #model.add(keras.layers.LeakyReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(num_classes, activation='softmax', name='output'))


    #model 3 is model 2 w/o dropout
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(32, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(16, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(8, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))

    #model 4, more layers
    model = keras.models.Sequential(name="FCN")
    model.add(keras.layers.Dense(128, input_shape = input_shapes))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.ReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.ReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.ReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(16))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.ReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(8))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.ReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(4))
    model.add(keras.layers.LayerNormalization())
    model.add(keras.layers.ReLU())
    #model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))

    #model 5, like model 2 but more dropout and linear activations in between# and batch norm
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, activation='relu'))
    ##model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(32, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(16))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(8))
    #model.add(keras.layers.Dropout(rate=0.2))
    ##model.add(keras.layers.Dense(8, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(4, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))

    #model 6, one more layer w.r.t. model 2
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(32, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(16, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(8, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(4, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))

    #model 7, is model 6 with more dropout
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(32, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(16, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.2))
    #model.add(keras.layers.Dense(8, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(4, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))

    #model 8, is model 6 with LayerNormalization
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(32, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(16, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(8, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(4, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))

    #model 9, is model 6 w/0 Dropout and only LayerNormalization
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(64, input_shape = input_shapes, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(32, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(16, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(8, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.1))
    #model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(4, activation='relu'))
    ##model.add(keras.layers.Dropout(rate=0.1))
    ##model.add(keras.layers.LayerNormalization())
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))

    #model thumb, rule of thumbs
    #model = keras.models.Sequential(name="FCN")
    #model.add(keras.layers.Dense(16, input_shape = input_shapes, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.3))
    #model.add(keras.layers.Dense(8, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.3))
    #model.add(keras.layers.Dense(4, activation='relu'))
    #model.add(keras.layers.Dropout(rate=0.3))
    #model.add(keras.layers.Dense(num_classes, activation='softmax'))

    return model


def get_FCN_constituents(num_classes, input_shapes):
    model = keras.models.Sequential(name="FCN_constituents")
    model.add(keras.layers.Dense(1000, input_shape = input_shapes, activation='relu'))
    model.add(keras.layers.Dense(500, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    return model


def get_BDT(n_epochs,features):

    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=n_epochs, verbosity=1, n_jobs=4, reg_lambda=1.0, feature_names=features)

    return model
