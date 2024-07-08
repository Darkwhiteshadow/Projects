import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras.layers import Layer


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

class L1Dist(Layer):
    def __init__(self,**kwargs):
        super().__init__()

    def call(self,input_embedding,validation_embedding):
        return tf.math.abs(input_embedding-validation_embedding)
    

model = tf.keras.models.load_model(
    'siamesemodel.h5',
    custom_objects={'L1Dist':L1Dist})

def preprocessing(image):

    byte_img =tf.io.read_file(image)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img,(100,100))
    img = img /255.0
    return img


def verify( image,folder,detectthresh,verifthresh):



    for img in os.listdir(folder):

        results=[]
       

        
        input_img = preprocessing(image)
        validation_img = preprocessing(os.path.join(folder,img ))

        
        
        result = model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
        
        
        print(result)
        results.append(result)

    detection = np.sum(np.array(results)>detectthresh)
    
    verification = detection/len(os.listdir(folder))
    print(detection)
    verified = verification>verifthresh
    
    

    if(verified): 
        print(img)
        return str(img)

    return "No match found"


