import string
import os
from keras.models import load_model
from model import logits_to_sentence
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# gpuoptions = tf.compat.v1.GPUOptions(allow_growth=True)
# Session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpuoptions))
# model_graph = Graph()
# with model_graph.as_default():
#     tf_session = Session()
#     with tf_session.as_default():

# print(string.punctuation.replace("'",'').replace("-",'').replace+ "¡" + '¿')
model=load_model('./backend/my_model_v3.h5')
print(logits_to_sentence(model ,"je suis bien "))
# def test1():
    #  with model_graph.as_default():
    #     with tf_session.as_default():
    #          print("eoioerfie")
            # predi=model.predict(x)