
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.wrappers import Bidirectional
from keras.utils import to_categorical
from keras.callbacks import CSVLogger,Callback,TensorBoard


# In[3]:

import numpy as np
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt


# In[1]:

data_dim = 2048
timesteps = 200
epoch = 30
classes = {'normal':0,'adenoma':1,'tub':2}
num_classes = len(classes)


# In[7]:

model = Sequential()
model.add(LSTM(256, return_sequences=True,
               input_shape=(timesteps, data_dim)))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.summary()

class prednor(Callback):
    def on_train_begin(self, logs={}):
        self.pred = []

    def on_batch_end(self, batch, logs={}):
        normal_list = model.predict(normal_test)[:,0]
        self.pred.append(sum(normal_list)/len(normal_list))

class predade(Callback):
    def on_train_begin(self, logs={}):
        self.pred = []

    def on_batch_end(self, batch, logs={}):
        adenoma_list = model.predict(adenoma_test)[:,1]
        self.pred.append(sum(adenoma_list)/len(adenoma_list))

class predtub(Callback):
    def on_train_begin(self, logs={}):
        self.pred = []

    def on_batch_end(self, batch, logs={}):
        tub_list = model.predict(tub_test)[:,2]
        self.pred.append(sum(tub_list)/len(tub_list))
# In[8]:

def plot_history(history):
    # 精度の履歴をプロット
    plt.plot(history.history['acc'],"o-",label="accuracy")
    plt.plot(history.history['val_acc'],"o-",label="validation_accuracy")
    for i,true in enumerate(prednor.pred):
        i += 1
        if i == 1:
            p_true = true
            t_list = []
        else:
            p_true += true
        if i % 5 == 0:
            t_ave = p_true/5
            p_true = 0
            t_list.append(t_ave)

    for i,false in enumerate(predtub.pred):
        i += 1
        if i == 1:
            p_false = false
            f_list = []
        else:
            p_false += false
        if i % 5 == 0:
            f_ave = p_false/5
            p_false = 0
            f_list.append(f_ave)

    for i,ade in enumerate(predade.pred):
        i += 1
        if i == 1:
            p_ade = ade
            f_ade = []
        else:
            p_ade += ade
        if i % 5 == 0:
            ade_ave = p_ade/5
            p_ade = 0
            f_ade.append(ade_ave)

    plt.plot(t_list,"o-",label="normal")
    plt.plot(f_list,"o-",label="tub")
    plt.plot(f_ade,"o-",label="adenoma")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim([0,1.05])
    plt.legend(loc="lower right")
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'],"o-",label="loss",)
    #plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.show()


# In[10]:

x_train = np.load('./numpy_data/x_train.npy')
y_train = np.load('./numpy_data/y_train.npy')
y_train = to_categorical(y_train, num_classes)
x_test = np.load('./numpy_data/x_test.npy')
y_test = np.load('./numpy_data/y_test.npy')

adenoma_test = []
normal_test = []
tub_test = []

for i in range(len(y_test)):
    if y_test[i] == 0:
        normal_test.append(x_test[i])
    elif y_test[i] == 1:
        adenoma_test.append(x_test[i])
    elif y_test[i] == 2:
        tub_test.append(x_test[i])

adenoma_test = np.asarray(adenoma_test)
normal_test = np.asarray(normal_test)
tub_test = np.asarray(tub_test)
y_test = to_categorical(y_test, num_classes)

callbacks = []
callbacks.append(CSVLogger("history.csv"))
callbacks.append(TensorBoard(log_dir='./log', histogram_freq=1))
prednor = prednor()
predade = predade()
predtub = predtub()
callbacks.append(prednor)
callbacks.append(predade)
callbacks.append(predtub)
history =model.fit(x_train, y_train,batch_size=2, epochs=epoch, callbacks=callbacks, validation_data=(x_test,y_test))
plot_history(history)
model.save("model.h5")
