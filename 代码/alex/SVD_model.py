from keras import backend as K 
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D 
from keras.initializers import RandomNormal 
import h5py as h5
from keras.layers import Input,Activation,Lambda,add
from keras.engine.training import Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2,l1
import numpy as np
#import os
import tensorflow as tf
from keras.initializers import Initializer 
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from data import Data_load
import math
from keras.callbacks import LearningRateScheduler
from group_compensation import CompenastionLayer


def Alexnet():
   
    inputs=Input(shape=(227,227,3))
    
    x=conv_block(inputs,96,(11,11),(4,4),'valid',name='conv1',fist_layer=True) 
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    
    x=conv_block(x,256,(5,5),(1,1),name='conv2',padding='same')
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)

    x=conv_block(x,384,(3,3),strides=(1,1),name='conv3',padding='same')
    x=Activation('relu')(x)
    
    x=conv_block(x,384,(3,3),strides=(1,1),name='conv4',padding='same')
    x=Activation('relu')(x)
    
    x=conv_block(x,256,(3,3),strides=(1,1),name='conv5',padding='same')
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    
    x=Flatten()(x)
    """
    x=Dense(4096, kernel_regularizer=l2(0.0005),
            bias_initializer='ones',name='fc6')(x)
    """
    x=dense_block(x,4096,name='fc6')
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    '''
    x=Dense(4096,kernel_regularizer=l2(0.0005),
            bias_initializer='ones',name='fc7')(x)
    '''
    x=dense_block(x,4096,name='fc7')
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    '''
    x=Dense(1000,kernel_regularizer=l2(0.0005),
            bias_initializer='ones',name='fc8')(x)
    '''
    x=dense_block(x,1000,name='fc8')
    predictions=Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    '''
    for i in ['fc7','fc8']:#['conv1','conv2','conv3','conv4','conv5','fc8']:#,'fc6','fc7']:
        layer=model.get_layer(name=i)
        #with h5.File('model_0.54.h5',mode='r') as f:
        with h5.File('model.h5',mode='r') as f:
            x1=f['model_14/'+i+'_9/kernel:0'].value
            x2=f['model_14/'+i+'_9/bias:0'].value
           
            K.set_value(layer.weights[0],x1)       #设置权重的值
            K.set_value(layer.weights[1],x2)
    '''    
    model.summary()
    return model

def conv_block(inputs,filters,kernel_size,strides,padding,fist_layer=False,name=None):
    with h5.File('model.h5',mode='r') as f:
        x1=f['model_14/'+name+'_9/kernel:0'].value
        x2=f['model_14/'+name+'_9/bias:0'].value  #偏置
        x2=tf.convert_to_tensor(x2)
        
        k1,k2,compen_value=SVD(x1,alph=0.65)
            
    if fist_layer:
        x=Conv2D(k1.shape[-1],kernel_size,strides=strides,input_shape=(227,227,3),use_bias=False,kernel_initializer=init(value=k1),
                 padding=padding,name=name+'_fist',kernel_regularizer=l2(0.0005))(inputs)
    else:
        x=Conv2D(k1.shape[-1],kernel_size,strides=strides,padding=padding,name=name+'_fist',use_bias=False,kernel_initializer=init(value=k1),
                 kernel_regularizer=l2(0.0005))(inputs)

    x=Conv2D(filters,(1,1),strides=(1,1),padding='valid',name=name+'_second',kernel_initializer=init(value=k2),
             bias_initializer=init(value=x2),kernel_regularizer=l2(0.0005))(x)
    x1=CompenastionLayer(filters,kernel_size,strides,padding,fist_layer=fist_layer,kernel_regularizer=l2(0.0005))(inputs)
    outputs=add([x,x1])
    return outputs

def dense_block(inputs,output_dim,name=None):
    with h5.File('model.h5',mode='r') as f:
        x1=f['model_14/'+name+'_9/kernel:0'].value
        x2=f['model_14/'+name+'_9/bias:0'].value  #偏置
        x2=tf.convert_to_tensor(x2)
        
    k1,k2,compen_value=SVD(x1,alph=0.5)
        
    x=Dense(k1.shape[-1], kernel_regularizer=l2(0.0005),use_bias=False,kernel_initializer=init(value=k1),name=name+'_fist')(inputs)
    outputs=Dense(output_dim, kernel_regularizer=l2(0.0005),kernel_initializer=init(value=k2),bias_initializer=init(value=x2),name=name+'_second')(x)
    
    return outputs
    
def SVD(kernel,alph=1):
    kernel_shape=np.shape(kernel)
    if np.ndim(kernel)==4:
        reshape_kernel=np.reshape(kernel,(kernel_shape[0]*kernel_shape[1]*kernel_shape[2],kernel_shape[3]))
    else:
        reshape_kernel=kernel
        
    U,s,V=np.linalg.svd(reshape_kernel)
   
    principal_sum=np.sum(s)
    sum_characteristic_value=0      
    for i in range(0,len(s)):
        sum_characteristic_value=sum_characteristic_value+s[i]
        if sum_characteristic_value>principal_sum*alph:
            break
    print(i+1)
    U=U[:,0:i+1]
    S=np.diag(s[0:i+1])
    U=np.matmul(U,S)   
    V=V[0:i+1,:]
    
    compensation_kernel=reshape_kernel-np.matmul(U,V)
    
    if np.ndim(kernel)==4:
        #还原滤波器1
        kernel_1=U.reshape(kernel_shape[0],kernel_shape[1],kernel_shape[2],U.shape[1]) 
        #还原滤波器2     
        kernel_2=V.reshape((1,1,V.shape[0],V.shape[1]))

        compensation_value=compensation_kernel.reshape(kernel_shape[0],kernel_shape[1],kernel_shape[2],compensation_kernel.shape[1])
    else:
        kernel_1=U
        kernel_2=V
        compensation_value=compensation_kernel
        
    return kernel_1,kernel_2,tf.convert_to_tensor(compensation_value)

class init(Initializer):
    def __init__(self, shape=None,value=None):
        self.value=value
        self.shape=shape

    def __call__(self, dtype=None):
        return K.variable(self.value)

if __name__=='__main__': 

    train_file='/media/yc/新加卷/train.tfrecords'
    test_file='/media/yc/新加卷/test.tfrecords'
    epochs=10
    batch_size=128
    data=Data_load(train_file,test_file,epochs,batch_size)
    
    model=Alexnet()    
    sgd=SGD(lr=0.001, momentum=0.9, nesterov=True)
    model= multi_gpu_model(model, gpus=2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy','top_k_categorical_accuracy'])  
    
    def scheduler(epoch):
            '''
            调整学习率，每10个epoch学习率除以10，最低为0.00001
            '''
            lr = K.get_value(model.optimizer.lr)
            if epoch == 4 or epoch==8:            
                K.set_value(model.optimizer.lr, lr * 0.1)
            if epoch>=20 and epoch%5==0 :
                pass
            if  epoch%2==0 and epoch!=0:
                acc=model.evaluate_generator(generator=data.Generate_TestData(),steps=100)
                print('在',epoch,'处的测试精度为：',acc)
            print('学习率为：',K.get_value(model.optimizer.lr))
            return K.get_value(model.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)
    print(model.evaluate_generator(generator=data.Generate_TestData(),steps=100))
    NUM_SAMPLE=1281167.0     
    history=model.fit_generator(generator=data.Generate_TrainData(),
                                steps_per_epoch=int(NUM_SAMPLE/batch_size),
                                epochs=epochs,
                                callbacks=[reduce_lr])
    
    accuracy=model.evaluate_generator(generator=data.Generate_TestData(),steps=100)
    print(model.evaluate_generator(generator=data.Generate_TestData(),steps=100))
    m=model.get_layer(index=3)
    m.save_weights("SVD.h5")




















