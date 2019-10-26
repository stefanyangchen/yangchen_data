from keras import backend as K 
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from data import Data_load
from keras.callbacks import LearningRateScheduler
from SVD_VGG16 import VGG_16
import h5py as h5

if __name__=='__main__':
    train_file='/media/yc/新加卷/train.tfrecords'
    test_file='/media/yc/新加卷/test.tfrecords'
    epochs=4
    batch_size=32
    data=Data_load(train_file,test_file,epochs,batch_size)   
    model=VGG_16()
    model.load_weights('SVD.h5',by_name=True)
    
    for i in ['block1_conv1','block1_conv2','block2_conv1','block2_conv2','block3_conv1',
              'block3_conv2','block3_conv3','block4_conv1','block4_conv2','block4_conv3','block5_conv1'
              ,'block5_conv2','block5_conv3','fc1','fc2','predictions']:
        layer=model.get_layer(name=i)
        with h5.File('SVD.h5',mode='r') as f:
            x1=f[i+'_first/'+i+'_first/'+'kernel:0']
            
            x2=f[i+'_second/'+i+'_second/'+'kernel:0']
            x3=f[i+'_second/'+i+'_second/'+'bias:0']
           
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
            if epoch == 2 or epoch==3:            
                K.set_value(model.optimizer.lr, lr * 0.1)
            if epoch>=20 and epoch%5==0 :
                pass
            if  epoch%2==0 and epoch!=0:
                acc=model.evaluate_generator(generator=data.Generate_TestData(),steps=500)
                print('在',epoch,'处的测试精度为：',acc)
            print('学习率为：',K.get_value(model.optimizer.lr))
            return K.get_value(model.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)
    #print(model.evaluate_generator(generator=data.Generate_TestData(),steps=100))
    NUM_SAMPLE=1281167.0  
    '''
    history=model.fit_generator(generator=data.Generate_TrainData(),
                                steps_per_epoch=int(NUM_SAMPLE/batch_size),
                                epochs=epochs,
                                callbacks=[reduce_lr])
    '''
    accuracy=model.evaluate_generator(generator=data.Generate_TestData(),steps=500)
    print(accuracy)
    #m=model.get_layer(index=3)
    #m.save_weights("SVD.h5")























