'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics

batch_size = 8
input_shape = (256,256,3)
img_rows,img_cols = input_shape[:2]
latent_dim = 2
intermediate_dim = 256
epochs = 20
epsilon_std = 1.0

# encoder architecture
x = Input(shape=input_shape)
conv_1 = Conv2D(16,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(x)
conv_2 = Conv2D(32,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(32,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_2)
conv_4 = Conv2D(64,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)
#2次元平面の平均と分散
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(64 * 16 * 16, activation='relu')
decoder_reshape = Reshape((16, 16, 64))
decoder_deconv_1_upsamp = Conv2DTranspose(64,
                                   kernel_size=(2, 2),
                                   padding='same',
                                   strides=(2, 2),
                                   activation='relu')
decoder_deconv_2_upsamp = Conv2DTranspose(32,
                                   kernel_size=(2, 2),
                                   padding='same',
                                   strides=(2, 2),
                                   activation='relu')
decoder_deconv_3_upsamp = Conv2DTranspose(32,
                                          kernel_size=(2, 2),
                                          strides=(2, 2),
                                          padding='same',
                                          activation='relu')
decoder_deconv_4_upsamp = Conv2DTranspose(16,
                                          kernel_size=(2, 2),
                                          strides=(2, 2),
                                          padding='same',
                                          activation='relu')
decoder_mean_squash = Conv2D(3,
                             kernel_size=2,
                             padding='same',
                             activation='sigmoid')
hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1_upsamp(reshape_decoded)
deconv_2_decoded = decoder_deconv_2_upsamp(deconv_1_decoded)
deconv_3_decoded = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_relu = decoder_deconv_4_upsamp(deconv_3_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean_squash)
        self.add_loss(loss, inputs=inputs)
        return x

y = CustomVariationalLayer()([x, x_decoded_mean_squash])

# entire model
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

#ディレクトリ配下の拡張子がsuffixファイルをすべて読み込む
def load_img_to_numpy(dirname,suffix):
    import glob
    from PIL import Image
    numpy_images = []
    filenames = glob.glob(dirname+suffix)
    for filename in filenames:
        raw_img = Image.open(filename).resize((256,256))
        array_img = np.asarray(raw_img)
        array_img.flags.writeable = True
        numpy_images.append(array_img)
    numpy_images = np.asarray(numpy_images,dtype="float")
    numpy_images /= 255
    return numpy_images

x_train = load_img_to_numpy('./cropped_images/','*.tif')
x_test = load_img_to_numpy('./cropped_test/','*.tif')
print(x_train.shape,x_test.shape)
vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,)
        #validation_data=(x_test, None))
vae.save_weights('colon_cancer_encoder.h5')
# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_train_encoded = encoder.predict(x_train, batch_size=batch_size)
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_train_encoded[:, 0], x_train_encoded[:, 1],s = 150,c='blue')
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1],s = 150,c='red')
plt.show()

#with open('plot_data.csv','w')

# build a digit generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1_upsamp(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2_upsamp(_deconv_1_decoded)
_deconv_3_decoded = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_relu = decoder_deconv_4_upsamp(_deconv_3_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)

# display a 2D manifold of the digits
n = 5  # figure with 15x15 digits
digit_size = 256

figure = np.zeros((digit_size * n, digit_size * n,3))
# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# to produce values of the latent variables z, since the prior of the latent space is Gaussian
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))


for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)[0]
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size,:] = x_decoded

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
"""
figure = np.zeros((digit_size, digit_size * n,3))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
for j, xi in enumerate(grid_y):
    z_sample = np.array([[xi, 0]])
    x_decoded = generator.predict(z_sample)[0]
    figure[:, j * digit_size: (j + 1) * digit_size, :] = x_decoded

plt.figure(figsize=(15, 1))
plt.imshow(figure, cmap='Greys_r')
plt.show()
"""
