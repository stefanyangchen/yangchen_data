import os
import numpy as np
import random

def get_data_path(images_path):
    trainDataPath = []
    trainLabel = []
    
    floder = []  # 图片所属的父文件夹

    validDataPath = []
    validLabel = []

    testDataPath = []
    testLabel = []

    fileList = os.listdir(images_path)
    
    for j,n in enumerate(fileList):
        data = os.listdir(os.path.join(images_path, fileList[j]))
        random.shuffle(data)

        trainData=np.array(data[:-(int(len(data)*0.4))])
        
        validData=np.array(data[(int(len(data)*0.6)):-(int(len(data)*0.2))])
       
        testData=np.array(data[-(int(len(data)*0.2)):])
        
        for i in range(len(trainData)):
            if(trainData[i][-3:]=="jpg"):
                image=os.path.join(os.path.join(images_path, fileList[j]), trainData[i])
                #把windows默认路径中的\转化为统一的/
                image=image.replace('\\','/')
               
                trainDataPath.append(image)
                trainLabel.append(int(j))
                floder.append(n)
   
        for i in range(len(validData)):
            if (validData[i][-3:] == "jpg"):
                image = os.path.join(os.path.join(images_path, fileList[j]), trainData[i])
                image = image.replace('\\', '/')
                validDataPath.append(image)
                validLabel.append(int(j))
        
    
        for i in range(len(testData)):
            if (validData[i][-3:] == "jpg"):
                image = os.path.join(os.path.join(images_path, fileList[j]), trainData[i])
                image = image.replace('\\', '/')
                testDataPath.append(image)
                testLabel.append(int(j))
                
    print(len(trainDataPath),    len(validDataPath),    len(testDataPath))        
    Save_trainlabel(trainDataPath, floder, trainLabel, fileList)  
    Save_validlabel(validDataPath, validLabel)
    Save_testlabel(testDataPath, testLabel)      
    

def Save_trainlabel(path, floder, label,fileList):
    btc = list(zip(path, label, floder))
    np.save('D:/train.npy', btc)

    label = list(enumerate(fileList))
    np.save('D:/num_label.npy', label)

def Save_testlabel(path, label):
    btc = list(zip(path, label))
    np.save('D:/test.npy', btc)

def Save_validlabel(path, label):
    btc = list(zip(path, label))
    np.save('D:/valid.npy', btc)


if  __name__ == '__main__':
    #error()
    get_data_path('D:/tomato_photos')
    # get_data_path('D:/Imagenet2012/ILSVRC2012_img_train(1)')