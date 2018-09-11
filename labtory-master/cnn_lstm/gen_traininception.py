
# coding: utf-8

# In[1]:
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import Input
from keras import backend as K
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.preprocessing import image


import os
import glob


# In[3]:
root_dir = '/media/dl-box/HD-PSFU3/GltractTIFF/大腸/images/train'
num_stack = 200
classes = {'normal':0,'adenoma':1,'tub':2}


input_tensor = Input(shape=(224, 224, 3))  # this assumes K.image_data_format() == 'channels_last'

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
#model.summary()
def vgg19_features(filename):
    img = load_img(filename,target_size=(224,224))
    x = np.asarray(img, dtype='float')
    x.flags.writeable = True
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    get_24rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
    [model.layers[-2].output])
    layer_output = get_24rd_layer_output([x, 0])[0][0]

    return layer_output


# In[ ]:

x_train = []
y_train = []
for _,dirs,_ in os.walk(root_dir):
    for dirname in dirs:
        if dirname in classes:
            print(dirname)
            category_dir = os.path.join(root_dir,dirname)
            for _,img_dirs,files in os.walk(category_dir):
                for img_dir in img_dirs:
                    print(img_dir)
                    dir_path = os.path.join(category_dir,img_dir)
                    files = sorted(glob.glob(dir_path + '/*.tif'))
                    if len(files) != 0:
                        sparse = len(files) // num_stack
                        print('sparse:{0}'.format(sparse))
                        counter = 0
                        for i,filename in enumerate(files):
                            if i % sparse == 0:
                                counter += 1
                                if counter > num_stack:
                                    break;
                                print(filename)
                                if i == 0:
                                    stacked_features = vgg19_features(filename)
                                else:
                                    features = vgg19_features(filename)
                                    stacked_features = np.vstack((stacked_features,features))
                        x_train.append(stacked_features)
                        y_train.append(classes[dirname])
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
np.save('./numpy_data/x_train.npy',x_train)
np.save('./numpy_data/y_train.npy',y_train)
print(x_train.shape)


# In[ ]:

train_data = np.load('./numpy_data/x_train.npy')
print(train_data.shape)


# In[ ]:
