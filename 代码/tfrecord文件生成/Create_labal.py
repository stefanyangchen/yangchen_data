
import os
import random
import tensorflow as tf 
import numpy as np
from scipy.io import loadmat
from PIL import Image


def get_data_path(images_path):
    path=[]   #每个图片的路径
    floder=[] #图片所属的父文件夹
    label_num=[] #文件夹的分类0——999(也即图片的分类)
    img_name=os.listdir(images_path)
    #['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']输出文件夹名
    
    for n,i in enumerate(img_name): # 0,'daisy'  1,'dandelion'  2,'roses'  3,'sunflowers'  4,'tulips'
        img=os.listdir(os.path.join(images_path,i))#img是图片名（15458787091_3edc6cd1eb.jpg）
        # print(img)
        #输出各个文件夹下的图片名
        for j in img:
            x=os.path.join(i,j)
            path.append(x)#x是父文件夹\图片名
            floder.append(i)#i是父文件夹名
            label_num.append(n)#n是第几类 0 1 2 3 4
    Save_label(path,floder,label_num,img_name)
    #error()

def Save_label(path,floder,label_num,img_name): 
    '''
    将创建的训练集的地址和标签保存，
    并同时保存分类标准
    '''
    # zip():用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    # list():用于将元组转换为列表。元组是放在圆括号中，列表放在方括号，元组元素值不能修改
    btc=list(zip(path,label_num,floder))
    np.save('D:/train.npy',btc)
    f=open('D:/train.txt','w')
    for j in btc:
        f.write(str(j)+'\n')
    f.close()
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    label=list(enumerate(img_name))
    
    np.save('D:/num_label.npy',label)
    
    f=open('D:/num_label.txt','w')
    for i in label:
        f.write(str(i)+'\n')
    f.close()
    
def error():
    train_set=np.load('D:/YC-biaozhun/train.npy')
    error1=[]
    valid_set=np.load('D:/YC-biaozhun/test.npy')
    error2=[]
    num_label=np.load('D:/YC-biaozhun/num_label.npy')
    f=[] #标准文件夹
    l=[] #标准标签
    for i in num_label:
        f.append(i[1])
        l.append(int(i[0]))

    for i in train_set:
        if l[f.index(i[2])]==int(i[1]):
            pass
        else:
            error1.append(i)
    for i in valid_set:
        if l[f.index(i[1])]==int(i[2]):
            pass
        else:
            error2.append(i)
    if len(error1)==0 and len(error2) ==0:
        print('训练集和验证集标签一一对应')
    else:
        return error1,error2


     
if  __name__ == '__main__': 
    #error()
    get_data_path('H:/flower_classi/flower_photos')#('D:/flower_photos')
    # get_data_path('D:/Imagenet2012/ILSVRC2012_img_train(1)')

    
    
    
    
    
    
    
    
    
    
    