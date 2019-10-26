import numpy as np                        
import h5py as h5

def printname(name):
     print (name)
#print(g[j].value.shape)     
with h5.File('D:/yangchen/ImageNet-VGG-16/vgg16_weights.h5',mode='r') as h:
    h.visit(printname)
    #print(list(h['predictions/predictions_W_1:0'].shape))
    #print(h['conv5/conv5_1/kernel:0'].value)
    '''
    for key in h:
            g=h[key]
            for j in g:
                print(g[j].value) 
                
    '''
    '''
            a=g[key+'_weights'].value
            b=a.copy() 
            c=np.append(a,b,axis=2)
            g[key+'_weights']=c               
    '''
        