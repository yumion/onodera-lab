'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.utils import plot_model
from keras.layers import Dense, Lambda, Flatten, Reshape, Layer, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from util.summary_model import summary_and_png


def vae_model(channel):

    batch_size = 8
    input_shape = (256,256,3)
    img_rows,img_cols = input_shape[:2]
    latent_dim = 2
    intermediate_dim = 256
    epochs = 100
    epsilon_std = 1.0


    # encoder architecture
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    flat = Flatten()(encoded)
    hidden = Dense(intermediate_dim, activation='relu')(flat)
    #2次元平面の平均と分散
    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(_sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')(z)
    decoder_upsample = Dense(64 * 16 * 16, activation='relu')(decoder_hid)
    decoder_reshape = Reshape((16, 16, 64))(decoder_upsample)
    x = Conv2D(64, (3, 3), padding='same')(decoder_reshape)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(channel, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    y = CustomVariationalLayer()([input_img, decoded])
    vae = Model(input_img, y)

    summary_and_png(vae, summary, to_png, png_file)



    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                                stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon


    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean_squash):
            x = K.flatten(x)
            x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
            xent_loss = img_rows * img_cols * metrics.binary_crossentropy(x, x_decoded_mean_squash)
            kl_loss = - 0.5 * K.mean(1 + .z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean_squash = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean_squash)
            self.add_loss(loss, inputs=inputs)
            return x

    return vae


if __name__ == '__main__':
    network = VAE(channel=3)
    model = network.get_model(to_png=True, png_file='vae_model.png')
    model.compile(optimizer='adam', loss=None)