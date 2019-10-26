import h5py as h5
import numpy as np
import tensorflow as tf
from keras import backend as K 
from keras.layers import Dense,Flatten,Dropout,add  
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.initializers import Initializer
from keras.layers import Input,Activation
from keras.engine.training import Model
from group_compensation import CompenastionLayer
from keras.regularizers import l2,l1

def VGG_16():
    inputs=Input(shape=(224,224,3))    
    # Block 1
    x=conv_block(inputs,64,(3,3),padding='same',name='block1_conv1',fist_layer=True)
    #x=BatchNormalization(axis=-1,name='BN1')(x)
    x=Activation('relu')(x)
    
    x=conv_block(x,64,(3,3),padding='same',name='block1_conv2')
    #x=BatchNormalization(axis=-1,name='BN2')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block1_pool')(x)
    
    #Block 2
    x=conv_block(x,128,(3,3),padding='same',name='block2_conv1')
    #x=BatchNormalization(axis=-1,name='BN3')(x)
    x=Activation('relu')(x)
    
    x=conv_block(x,128,(3,3),padding='same',name='block2_conv2')
    #x=BatchNormalization(axis=-1,name='BN4')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block2_pool')(x)
    
    # Block 3
    x=conv_block(x,256,(3,3),padding='same',name='block3_conv1')
    #x=BatchNormalization(axis=-1,name='BN5')(x)
    x=Activation('relu')(x)
    
    x=conv_block(x,256,(3,3),padding='same',name='block3_conv2')
    #x=BatchNormalization(axis=-1,name='BN6')(x)
    x=Activation('relu')(x)
    
    x=conv_block(x,256,(3,3),padding='same',name='block3_conv3')
    #x=BatchNormalization(axis=-1,name='BN7')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block3_pool')(x)
    
    #Black 4
    x=conv_block(x,512,(3,3),padding='same',name='block4_conv1')
    #x=BatchNormalization(axis=-1,name='BN8')(x)
    x=Activation('relu')(x)
    
    x=conv_block(x,512,(3,3),padding='same',name='block4_conv2')
    #x=BatchNormalization(axis=-1,name='BN9')(x)
    x=Activation('relu')(x)
    
    x=conv_block(x,512,(3,3),padding='same',name='block4_conv3')
    #x=BatchNormalization(axis=-1,name='BN10')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block4_pool')(x)
    
    #Black 5
    x=conv_block(x,512,(3,3),padding='same',name='block5_conv1')
    #x=BatchNormalization(axis=-1,name='BN11')(x)
    x=Activation('relu')(x)
    
    x=conv_block(x,512,(3,3),padding='same',name='block5_conv2')
    #x=BatchNormalization(axis=-1,name='BN12')(x)
    x=Activation('relu')(x)
    
    x=conv_block(x,512,(3,3),padding='same',name='block5_conv3')
    #x=BatchNormalization(axis=-1,name='BN13')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block5_pool')(x)
    
    x=Flatten()(x)
    
    # Classification block, 全连接3层
    x=dense_block(x,4096,name='fc1')
    #x=BatchNormalization(axis=-1,name='BN14')(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    
    x=dense_block(x,4096,name='fc2')
    x=Activation('relu')(x)
    #x=BatchNormalization(axis=-1,name='BN15')(x)
    x=Dropout(0.5)(x)
    
    x=dense_block(x,1000,name='predictions')
    predictions=Activation('softmax')(x)
   
    model=Model(inputs=inputs, outputs=predictions)
    #model.load_weights('F:/weight_point/vgg16_weights.h5',by_name=True)
   
    model.summary()
    '''
    layer=model.get_layer(name='fc3')
    print(K.eval(layer.weights[1]))
    '''
    return model


def conv_block(inputs, filters, kernel_size, padding, strides=(1, 1), fist_layer=False, name=None):
    
    with h5.File('vgg16_weights.h5', mode='r') as f:
        x1 = f[name + '/'+name+'_W_1:0'].value
        x2 = f[name + '/'+name+'_b_1:0'].value  # 偏置
        x2 = tf.convert_to_tensor(x2)

        k1, k2, compen_value = SVD(x1, alph=0.65)
    
    if fist_layer:
        x = Conv2D(k1.shape[-1], kernel_size, strides=strides, input_shape=(227, 227, 3), use_bias=False,
                   kernel_initializer=init(value=k1),
                   padding=padding, name=name + '_fist', kernel_regularizer=l2(0.0005))(inputs)
    else:
        x = Conv2D(k1.shape[-1], kernel_size, strides=strides, padding=padding, name=name + '_fist', use_bias=False,
                   kernel_initializer=init(value=k1),
                   kernel_regularizer=l2(0.0005))(inputs)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='valid', name=name + '_second',
               kernel_initializer=init(value=k2),
               bias_initializer=init(value=x2), kernel_regularizer=l2(0.0005))(x)
    x1 = CompenastionLayer(filters, kernel_size, strides, padding, fist_layer=fist_layer,
                           kernel_regularizer=l2(0.0005))(inputs)
    outputs = add([x, x1])
    return outputs


def dense_block(inputs, output_dim, name=None):
    with h5.File('vgg16_weights.h5', mode='r') as f:
        x1 = f[name + '/'+name+'_W_1:0'].value
        x2 = f[name + '/'+name+'_b_1:0'].value  # 偏置
        x2 = tf.convert_to_tensor(x2)

        k1, k2, compen_value = SVD(x1, alph=0.6)

    x = Dense(k1.shape[-1], kernel_regularizer=l2(0.0005), use_bias=False, kernel_initializer=init(value=k1),
              name=name + '_fist')(inputs)
    outputs = Dense(output_dim, kernel_regularizer=l2(0.0005), kernel_initializer=init(value=k2),
                    bias_initializer=init(value=x2), name=name + '_second')(x)

    return outputs


def SVD(kernel, alph=1):
    kernel_shape = np.shape(kernel)
    if np.ndim(kernel) == 4:
        reshape_kernel = np.reshape(kernel, (kernel_shape[0] * kernel_shape[1] * kernel_shape[2], kernel_shape[3]))
    else:
        reshape_kernel = kernel

    U, s, V = np.linalg.svd(reshape_kernel)

    principal_sum = np.sum(s)
    sum_characteristic_value = 0
    for i in range(0, len(s)):
        sum_characteristic_value = sum_characteristic_value + s[i]
        if sum_characteristic_value > principal_sum * alph:
            break
    print(i + 1)
    U = U[:, 0:i + 1]
    S = np.diag(s[0:i + 1])
    U = np.matmul(U, S)
    V = V[0:i + 1, :]

    compensation_kernel = reshape_kernel - np.matmul(U, V)

    if np.ndim(kernel) == 4:
        # 还原滤波器1
        kernel_1 = U.reshape(kernel_shape[0], kernel_shape[1], kernel_shape[2], U.shape[1])
        # 还原滤波器2
        kernel_2 = V.reshape((1, 1, V.shape[0], V.shape[1]))

        compensation_value = compensation_kernel.reshape(kernel_shape[0], kernel_shape[1], kernel_shape[2],
                                                         compensation_kernel.shape[1])
    else:
        kernel_1 = U
        kernel_2 = V
        compensation_value = compensation_kernel

    return kernel_1, kernel_2, tf.convert_to_tensor(compensation_value)


class init(Initializer):
    def __init__(self, shape=None, value=None):
        self.value = value
        self.shape = shape

    def __call__(self, dtype=None):
        return K.variable(self.value)

if __name__=='__main__':
    VGG_16()