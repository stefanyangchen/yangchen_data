import tensorflow as tf
import os 
from PIL import Image

import random 
import time
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
 
#生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

#制作TFRecord格式
def create_trainTFRecord(filename):
    #加载训练集的地址和标签
    path = np.load('D:/train.npy')
    # path=list(np.load('D:/YC-biaozhun/train.npy'))
    #加载分类标准
    num=list(np.load('D:/num_label.npy'))
    l=[]
    p=[]
    for i in num:
        l.append(i[0])
        p.append(i[1])
    #创建一个写入
    writer = tf.python_io.TFRecordWriter(filename)
    #将训练集混洗
    random.shuffle(path)
    #读取文件   
    im=np.zeros((256,256,3))
    for n,j in enumerate(path):
        x1=time.time()
        #验证训练集的标签和分类标准是否相等
        if not int(l[p.index(j[2])])==int(j[1]):
            print('标签不匹配')
            print(j)
        img = Image.open(os.path.join(j[0])).convert('RGB')
        img = img.resize((256, 256), Image.ANTIALIAS)

        img_raw = img.tobytes()          #将图片转化成二进制格式
        example = tf.train.Example(features = tf.train.Features(feature = {
                    'label':_int64_feature(int(j[1])),
                    'image': _bytes_feature(img_raw),
                })) 
        writer.write(example.SerializeToString())
        if n%10000==0:
            print('train',n)
            print(time.time()-x1)
        im=im+img
    writer.close()
    btc=im/(n+1)
    np.save('btc.npy',btc)
    eth=btc.sum(axis=0)
    xrp=eth.sum(axis=0)
    print(xrp/(256*256))
    

def create_testTFRecord(filename):
    #加载测试集文件地址和标签
    path=list(np.load('D:/test.npy'))

    #创建写入
    writer = tf.python_io.TFRecordWriter(filename)
    # test_path='D:/Imagenet2012/ILSVRC2012_img_val'
    for n,j in enumerate(path):
        #验证测试集标签和分类标准是否一一对应
        '''
        if not int(l[p.index(j[1])])==int(j[2]):
            print('标签不匹配')
            print(j)
        '''    
        img = Image.open(os.path.join(j[0])).convert('RGB')
        img = img.resize((256, 256), Image.ANTIALIAS)
       
        img_raw = img.tobytes()          #将图片转化成二进制格式
        example = tf.train.Example(features = tf.train.Features(feature = {
                    'label':_int64_feature(int(j[1])),
                    'image': _bytes_feature(img_raw),
                    
                })) 
        writer.write(example.SerializeToString())
        
    writer.close()   

def create_validset(filename):
        #加载测试集文件地址和标签
    path=list(np.load('D:/valid.npy'))

    #创建写入
    writer = tf.python_io.TFRecordWriter(filename)
    # test_path='D:/Imagenet2012/ILSVRC2012_img_val'
    for n,j in enumerate(path):
           
        img = Image.open(os.path.join(j[0])).convert('RGB')
        img = img.resize((256, 256), Image.ANTIALIAS)
       
        img_raw = img.tobytes()          #将图片转化成二进制格式
        example = tf.train.Example(features = tf.train.Features(feature = {
                    'label':_int64_feature(int(j[1])),
                    'image': _bytes_feature(img_raw),
                    
                })) 
        writer.write(example.SerializeToString())
        
    writer.close()   
if __name__ =="__main__":
    #训练图片两张为一个batch,进行训练，测试图片一起进行测试
    
    train_filename = "D:/train.tfrecords"
    create_trainTFRecord(train_filename)
    
    test_filename = "D:/test.tfrecords"
    create_testTFRecord(test_filename)
    
    valid_filename='D:/valid.tfrecords'
    create_validset(valid_filename)