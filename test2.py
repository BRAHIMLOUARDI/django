# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import numpy as np
# # import matplotlib.pyplot as plt
# from tensorflow import keras

# from tensorflow.keras import optimizers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.models import load_model

# # from tensorflow.keras.datasets import mnist
# # from tensorflow.keras.utils import to_categorical

# modele = Sequential()

# def f(x):
#  return np.cos(2*x) + x*np.sin(3*x) + x**0.5 - 2
# a, b = 0, 5 # intervalle [a,b]
# N = 10 # taille des données
# X = np.linspace(a, b, N) # abscisses
# Y = f(X) # ordonnées
# X_train = X.reshape(-1,1)
# Y_train = Y.reshape(-1,1)
# print(X_train)
# print(Y_train)
# Partie B. Réseau
# modele = Sequential()
# p = 10
# modele.add(Dense(p, input_dim=1, activation='tanh'))
# modele.add(Dense(p, activation='tanh'))
# modele.add(Dense(p, activation='tanh'))
# modele.add(Dense(p, activation='tanh'))
# modele.add(Dense(1, activation='linear'))

# mysgd = optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9, nesterov=True)
# modele.compile(loss='mean_squared_error', optimizer=mysgd,metrics=['accuracy'])
# print(modele.summary())

# history = modele.fit(X_train, Y_train, epochs=4, batch_size=N)

# for x in range(25):
#    modele.save(f"media/my_model_exp{x}.h5")
#    modele=load_model(f"media/my_model_exp{x}.h5") 
#    history = modele.fit(X_train, Y_train, epochs=4, batch_size=N)
   
   


# plt.plot(X_train, Y_train, color='blue')
# plt.plot(X_train, Y_predict, color='red')
# plt.show()
# # Affichage de l'erreur au fil des époques
# plt.plot(history.history['loss'])
# plt.show()








#1model
# import os
# import numpy as np
# import tensorflow as tf
# import keras as keras
# from keras.models import Sequential
# from keras.layers import  Input, Dense
# from keras.models import load_model

# modelexp = Sequential()
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# # modelexp.add(Input(shape=(1,)))
# modelexp.add(Dense(10,input_dim=1))
# modelexp.add(Dense(100))
# modelexp.add(Dense(100))
# modelexp.add(Dense(100))
# modelexp.add(Dense(100))
# modelexp.add(Dense(100))
# modelexp.add(Dense(1))


# modelexp.compile(optimizer='sgd', loss=keras.losses.MeanSquaredError( name="mean_squared_error"),metrics=['accuracy'])
# xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# xs=np.reshape(xs,(12,1))
# ys = np.array([1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
# ys=np.reshape(ys,(12,1))
# print(modelexp.summary())
# modelexp.fit(xs, ys,epochs=1, batch_size=1 )
# x1=np.array([40])
# x1=np.reshape(x1,(1,1))
# print(x1.shape)

# for x in range(25):
#    print(modelexp.predict(x1))
#    modelexp.save(f"media/my_model_exp{x}.h5")
#    modelexp=load_model(f"media/my_model_exp{x}.h5") 
#    modelexp.fit(xs, ys,epochs=4, batch_size=1 )
   














# from re import X
# import string
# import os
# from keras.models import load_model
# from model import logits_to_sentence
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# gpuoptions = tf.compat.v1.GPUOptions(allow_growth=True)
# Session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpuoptions))
# model_graph = Graph()
# with model_graph.as_default():
#     tf_session = Session()
#     with tf_session.as_default():

# print(string.punctuation.replace("'",'').replace("-",'').replace+ "¡" + '¿')
# model=load_model('./backend/my_model_v3.h5')
# print(logits_to_sentence(model ,"je suis bien "))
# def test1():
    #  with model_graph.as_default():
    #     with tf_session.as_default():
    #          print("eoioerfie")
            # predi=model.predict(x)


x='le "non" l\'emporte'

x=x
y=[x]


print(y)