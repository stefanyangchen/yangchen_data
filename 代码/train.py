# from model import Alexnet
from VGG_16 import VGG_16
from keras.optimizers import SGD,Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,Callback,ReduceLROnPlateau
from keras.utils import multi_gpu_model
from data import Data_load 
import math
import tensorflow as tf



NUM_TRAIN=7173.0
NUM_TEST=2386
NUM_VALID=2398

def train(train_file,
          test_file,
          valid_file,
          epochs,
          batch_size,
          lr,
          momentum):
    model=VGG_16()    
    sgd=SGD(lr=lr, momentum=momentum, nesterov=True)
    model= multi_gpu_model(model, gpus=2)   #gpu调用
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])  
    
    data=Data_load(train_file,test_file,valid_file,epochs,batch_size)
    
    def scheduler(epoch):
        '''
        调整学习率，每10个epoch学习率除以10，最低为0.00001
        '''
        lr = K.get_value(model.optimizer.lr)
        if epoch == 15 or epoch==30:            
            K.set_value(model.optimizer.lr, lr * 0.1)
        print('学习率为：',K.get_value(model.optimizer.lr))
        return K.get_value(model.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)

    history=model.fit_generator(generator=data.Generate_TrainData(),
                                steps_per_epoch=int(NUM_TRAIN/batch_size),
                                epochs=epochs,
                                validation_data=data.Generate_validData(),
                                validation_steps=int(NUM_VALID/64),
                                callbacks=[reduce_lr])
    
    accuracy=model.evaluate_generator(generator=data.Generate_TestData(),steps=math.ceil(NUM_TEST/64))
    m=model.get_layer(index=3)
    m.save_weights('BN_model_{}.h5'.format(accuracy[1]))
    #model.save_weights('路径')   cpu时使用这段保存参数
    #print(history)
    print(accuracy)
                                        
if __name__=='__main__':
    
    train_file='/home/yc/yl/train.tfrecords'
    test_file='/home/yc/yl/test.tfrecords'
    valid_file='/home/yc/yl/valid.tfrecords'
    epochs=45
    batch_size =50
    lr=0.01
    momentum=0.9   
    train(train_file,test_file,valid_file, epochs,batch_size,lr,momentum)
    
   
    
    
    