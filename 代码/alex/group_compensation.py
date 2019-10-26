from keras.engine.topology import Layer
import keras.backend as K
from keras.utils import conv_utils

class CompenastionLayer(Layer):
    """
    完成卷积分解后的补偿,主要先通过一个可分离卷积获取形状然后通过1*1的卷积补充通道
    
    """
    def __init__(self,filters,kernel_size,strides=(1,1),padding='valid',kernel_regularizer=None,fist_layer=False,alph=0.125,kernel_initializer='glorot_uniform',**kwargs):
        
        self.filters=filters
        self.kernel_size=kernel_size
        self.strides=strides
        self.padding=padding
        self.kernel_regularizer=kernel_regularizer
        self.kernel_initializer=kernel_initializer
        self.fist_layer=fist_layer
        if self.fist_layer:
            self.alph=1.0
        else:
            self.alph=alph
        super(CompenastionLayer, self).__init__(**kwargs)
        
    def build(self,input_shape):
       
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_shape[3],
                                  1)
        self.depthwise_kernel = self.add_weight(
                                    shape=depthwise_kernel_shape,
                                    initializer=self.kernel_initializer,
                                    name='depthwise_kernel',
                                    regularizer=self.kernel_regularizer)
        
        self.kernel=self.add_weight(
                                    shape=(1,1,int(self.alph*input_shape[3]),self.filters),
                                    initializer=self.kernel_initializer,
                                    name='kernel',
                                    regularizer=self.kernel_regularizer)
        
        self.built = True
    
    def call(self,x):
        depthwise_outputs=K.depthwise_conv2d(x,self.depthwise_kernel,
                                                strides=self.strides,
                                                padding=self.padding,)
        if self.fist_layer:
            x1=depthwise_outputs
        else:
            num=int(self.alph*K.int_shape(depthwise_outputs)[-1])
            step=int(1/self.alph)
            T=[]
            for i in range(0,num,2):
                a1=depthwise_outputs[:,:,:,(i*step):(i+1)*step]
                a2=depthwise_outputs[:,:,:,((i+1)*step):(i+2)*step]
                T.append(K.sum(a1,axis=-1,keepdims=True))
                T.append(K.sum(a2,axis=-1,keepdims=True))
            x1=K.concatenate(T)
            
        x1=K.relu(x1)
        outputs=K.conv2d(x1,self.kernel)
        
        return outputs
    
    def compute_output_shape(self,input_shape):
        
        rows = input_shape[1]
        cols = input_shape[2]
        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])
        
        return (input_shape[0],rows,cols,self.filters)
    

