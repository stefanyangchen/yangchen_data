import tensorflow as tf
import os
import time

class Data_load:
    
    def __init__(self,train_file,test_file,epochs,batch_size):
        self.train_filename=train_file
        self.test_filename=test_file
        self.epochs=epochs
        self.batch_size=batch_size
        
    def parser_train(self,filename):
        """
        读取并解析TF文件,并将图片裁剪到需要的大小
        Arguement:
            filename:TF文件目录
        Return:
            img:图片
            labels:标签            
        """
        
        features = tf.parse_single_example(
                            filename,
                            features = {'label':tf.FixedLenFeature([], tf.int64),
                                       'image': tf.FixedLenFeature([], tf.string)})
        #将字符串解析成图像对应的像素数组
        img = tf.decode_raw(features['image'], tf.uint8)
        labels=tf.cast(features['label'],tf.int32)
        
        img = tf.reshape(img,[256,256, 3]) #reshape为256*256*3通道图片
        img,labels=self.image_preprocess(img,labels,'train')
        return img, labels
    
    def parser_test(self,filename):
        """
        读取并解析TF文件,并将图片裁剪到需要的大小
        Arguement:
            filename:TF文件目录
        Return:
            img:图片
            labels:标签            
        """
        
        features = tf.parse_single_example(
                            filename,
                            features = {'label':tf.FixedLenFeature([], tf.int64),
                                       'image': tf.FixedLenFeature([], tf.string)})
        #将字符串解析成图像对应的像素数组
        img = tf.decode_raw(features['image'], tf.uint8)
        labels=tf.cast(features['label'],tf.int32)
        
        img = tf.reshape(img,[256,256,3]) #reshape为256*256*3通道图片
        img,labels=self.image_preprocess(img,labels,'test')
        return img, labels
    def image_preprocess(self,img,labels,model):
        """
        图片预处理函数，包括减去均值，随机翻转，随机裁剪
        """
        #创建均值张量
        IMAGENET_MEAN =tf.ones((227,227,3))*tf.constant([122.68,116.64,104.00],shape=(1,1,3))
        #裁剪图片
        image =tf.cast(tf.random_crop(img,[227,227,3]),tf.float32)
        #随机翻转图片
        if model=='train':
            image=tf.image.random_flip_left_right(image) 
        
        image-=IMAGENET_MEAN
                           
        labels=tf.one_hot(labels,1000)
        #image=tf.stack([image,arguement_image])
        #labels=tf.stack([labels,labels])
        
        return image,labels
    
    def Generate_TrainData(self):
        """
        训练文件生成器
        """
        btc=self.train_filename
        #for i in os.listdir(self.train_filename):
        #    btc.append(os.path.join(self.train_filename,i))  
        dataset=tf.data.TFRecordDataset(btc)
        dataset=dataset.shuffle(12800).repeat(self.epochs).map(self.parser_train,num_parallel_calls=32).batch(self.batch_size,drop_remainder=True)
        iterator=dataset.make_one_shot_iterator()
        images,labels=iterator.get_next()
        
        with tf.Session() as sess:
            while True:
                try:
                    im,la=sess.run([images,labels])
                    yield (im,la)                    
                except tf.errors.OutOfRangeError:
                    print('Epoch End')
                    break
                            
    def Generate_TestData(self):
        """
        测试文件生成器
        """
        dataset=tf.data.TFRecordDataset(self.test_filename)
        dataset=dataset.repeat(1).map(self.parser_test,num_parallel_calls=32).batch(500)
        iterator=dataset.make_one_shot_iterator()
        images,labels=iterator.get_next()
        
        with tf.Session() as sess:
            while True:
                try:
                    im,la=sess.run([images,labels])
                    yield (im,la)                  
                except tf.errors.OutOfRangeError:
                    print('Test End')
                    break
                
               
if __name__=='__main__':
    d=Data_load('/media/yc/新加卷/train.tfrecords','/media/yc/新加卷/test.tfrecords',10,128)
    a=d.Generate_TestData()
    x=time.time()
    n=0
    for i in a:
        n=n+1
        #print(i[1])
    print(n)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    