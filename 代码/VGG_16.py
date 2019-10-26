from keras import backend as K 
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D 
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomNormal 
import h5py as h5
from keras.layers import Input,Activation
from keras.engine.training import Model

from keras.regularizers import l2,l1

def VGG_16():
    
    inputs=Input(shape=(224,224,3))    
    # Block 1
    x=Conv2D(64,(3,3),input_shape=(224,224,3),padding='same',name='block1_conv1',
             kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(inputs)
    x=BatchNormalization(axis=-1,name='BN1')(x)
    x=Activation('relu')(x)
    
    x=Conv2D(64,(3,3),padding='same',name='block1_conv2'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN2')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block1_pool')(x)
    
    #Block 2
    x=Conv2D(128,(3,3),padding='same',name='block2_conv1'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN3')(x)
    x=Activation('relu')(x)
    
    x=Conv2D(128,(3,3),padding='same',name='block2_conv2'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN4')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block2_pool')(x)
    
    # Block 3
    x=Conv2D(256,(3,3),padding='same',name='block3_conv1'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN5')(x)
    x=Activation('relu')(x)
    
    x=Conv2D(256,(3,3),padding='same',name='block3_conv2'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN6')(x)
    x=Activation('relu')(x)
    
    x=Conv2D(256,(3,3),padding='same',name='block3_conv3'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN7')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block3_pool')(x)
    
    #Black 4
    x=Conv2D(512,(3,3),padding='same',name='block4_conv1'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN8')(x)
    x=Activation('relu')(x)
    
    x=Conv2D(512,(3,3),padding='same',name='block4_conv2'
             ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN9')(x)
    x=Activation('relu')(x)
    
    x=Conv2D(512,(3,3),padding='same',name='block4_conv3'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN10')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block4_pool')(x)
    
    #Black 5
    x=Conv2D(512,(3,3),padding='same',name='block5_conv1'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN11')(x)
    x=Activation('relu')(x)
    
    x=Conv2D(512,(3,3),padding='same',name='block5_conv2'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN12')(x)
    x=Activation('relu')(x)
    
    x=Conv2D(512,(3,3),padding='same',name='block5_conv3'
                ,kernel_initializer=RandomNormal(0.0,0.01),kernel_regularizer=l2(0.0005))(x)
    x=BatchNormalization(axis=-1,name='BN13')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D(pool_size=(2,2),strides=(2,2),name='block5_pool')(x)
    
    x=Flatten()(x)
    
    # Classification block, 全连接3层
    x=Dense(4096,kernel_initializer=RandomNormal(0.0,0.01),
            bias_initializer='zeros',name='fc1',kernel_regularizer=l2(0.0005))(x)
    #x=BatchNormalization(axis=-1,name='BN14')(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    
    x=Dense(4096,kernel_initializer=RandomNormal(0.0,0.01),
            bias_initializer='zeros',name='fc2',kernel_regularizer=l2(0.0005))(x)
    x=Activation('relu')(x)
    #x=BatchNormalization(axis=-1,name='BN15')(x)
    x=Dropout(0.5)(x)
    
    x=Dense(1000,kernel_initializer=RandomNormal(0.0,0.01),
            bias_initializer='zeros',name='fc3',kernel_regularizer=l2(0.0005))(x)
    predictions=Activation('softmax')(x)
   
    model=Model(inputs=inputs, outputs=predictions)
    model.summary()
    #model.load_weights('F:/weight_point/vgg16_weights.h5',by_name=True)
    '''
    for i in ['block1_conv1','block1_conv2','block2_conv1','block2_conv2','block3_conv1',
              'block3_conv2','block3_conv3','block4_conv1','block4_conv2','block4_conv3','block5_conv1'
              ,'block5_conv2','block5_conv3','fc1','fc2','fc3']:
        layer=model.get_layer(name=i)
        with h5.File('F:/weight_point/vgg16_weights.h5',mode='r') as f:
            if i !='fc3':
                x1=f[i+'/'+i+'_W_1:0']
                x2=f[i+'/'+i+'_b_1:0']
            else:          
                x1=f['predictions/predictions_W_1:0']
                x2=f['predictions/predictions_b_1:0']
                
            K.set_value(layer.weights[0],x1)       #设置权重的值
            K.set_value(layer.weights[1],x2)
    model.summary()
    '''
    '''
    layer=model.get_layer(name='fc3')
    print(K.eval(layer.weights[1]))
    '''
    return model
'''    
model=VGG_16()
sgd=SGD(lr=0.001, momentum=0.9, nesterov=True)
model= multi_gpu_model(model, gpus=2)
model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy','top_k_categorical_accuracy'])  
train_file='F:/Train_Tfrecords'
test_file='F:/test.tfrecords'    
data=Data_load(train_file,test_file,10,64)
acc=model.evaluate_generator(generator=data.Generate_TestData(),steps=500)    
print(acc)    
'''    
