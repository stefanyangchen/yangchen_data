from keras import backend as K 
from keras.layers import Dense,Flatten,Dropout  
from keras.layers.convolutional import Conv2D,MaxPooling2D 
from keras.initializers import RandomNormal 
import h5py as h5
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from data import Data_load
from keras.layers import Input,Activation
from keras.engine.training import Model
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization

def Alexnet():
    inputs=Input(shape=(227,227,3))
    
    x=Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),
             padding='valid',name='conv1',
             kernel_initializer=RandomNormal(0.0,0.01))(inputs)
    '''
    x=BatchNormalization(axis=-1,momentum=0.99, epsilon=0.001, 
                         center=True, scale=True
                         )(x)
    '''
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    
    x=Conv2D(256,(5,5),strides=(1,1),name='conv2',
                     padding='same',bias_initializer='ones',
                     kernel_initializer=RandomNormal(0.0,0.01))(x)
    '''
    x=BatchNormalization(axis=-1,momentum=0.99, epsilon=0.001, 
                         center=True, scale=True
                         )(x)
    '''
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    
    x=Conv2D(384,(3,3),strides=(1,1),name='conv3',
                     padding='same',
                     kernel_initializer=RandomNormal(0,0.01))(x)
    '''
    x=BatchNormalization(axis=-1,momentum=0.99, epsilon=0.001, 
                         center=True, scale=True
                         )(x)
    '''
    x=Activation('relu')(x)
    
    x=Conv2D(384,(3,3),strides=(1,1),name='conv4',
                     padding='same',bias_initializer='ones',
                     kernel_initializer=RandomNormal(0,0.01))(x)
    '''
    x=BatchNormalization(axis=-1,momentum=0.99, epsilon=0.001, 
                         center=True, scale=True
                         )(x)
    '''
    x=Activation('relu')(x)
    
    x=Conv2D(256,(3,3),strides=(1,1),name='conv5',
                     padding='same',bias_initializer='ones',
                     kernel_initializer=RandomNormal(0,0.01))(x)
    '''
    x=BatchNormalization(axis=-1,momentum=0.99, epsilon=0.001, 
                         center=True, scale=True
                         )(x)
    '''
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    
    x=Flatten()(x)
    
    x=Dense(4096,kernel_initializer=RandomNormal(0.0,0.01),
            bias_initializer='ones',name='fc6')(x)
    '''
    x=BatchNormalization(axis=1,momentum=0.99, epsilon=0.001, 
                         center=True, scale=True
                         )(x)
    '''
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    
    x=Dense(4096,kernel_initializer=RandomNormal(0.0,0.01),
            bias_initializer='ones',name='fc7')(x)
    '''
    x=BatchNormalization(axis=1,momentum=0.99, epsilon=0.001, 
                         center=True, scale=True
                         )(x)
    '''
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    
    x=Dense(1000,kernel_initializer=RandomNormal(0.0,0.01),
            bias_initializer='ones',name='fc8')(x)

    predictions=Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.summary()
    #model.load_weights('again_model_10.hdf5',by_name=True)
    
    for i in ['conv1','conv2','conv3','conv4','conv5','fc8','fc6','fc7']:
        layer=model.get_layer(name=i)
        #with h5.File('model_0.54.h5',mode='r') as f:
        with h5.File('model.h5',mode='r') as f:
            x1=f['model_14/'+i+'_9/kernel:0'].value
            x2=f['model_14/'+i+'_9/bias:0'].value
           
            K.set_value(layer.weights[0],x1)       #设置权重的值
            K.set_value(layer.weights[1],x2)
  
    #model.load_weights('F:/weight_point/epoch_10.h5py',by_name=True)
    
    return model

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
    '''
    NUM_SAMPLE=1281167.0     
    history=model.fit_generator(generator=data.Generate_TrainData(),
                                steps_per_epoch=int(NUM_SAMPLE/batch_size),
                                epochs=epochs,
                                callbacks=[reduce_lr])
    
    accuracy=model.evaluate_generator(generator=data.Generate_TestData(),steps=100)
    print(model.evaluate_generator(generator=data.Generate_TestData(),steps=100))
    m=model.get_layer(index=3)
    m.save_weights("SVD.h5")
    '''





