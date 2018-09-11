# coding: utf-8

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam
import keras.backend as K

# self made function
from util.summary_model import summary_and_png


class UNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 512
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        # (256 x 256 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

        # エンコーダーの作成
        # (128 x 128 x N)
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (64 x 64 x 2N)
        filter_count = first_layer_filter_count*2
        enc2 = self._add_encoding_layer(filter_count, enc1)

        # (32 x 32 x 4N)
        filter_count = first_layer_filter_count*4
        enc3 = self._add_encoding_layer(filter_count, enc2)

        # (16 x 16 x 8N)
        filter_count = first_layer_filter_count*8
        enc4 = self._add_encoding_layer(filter_count, enc3)

        # (8 x 8 x 8N)
        enc5 = self._add_encoding_layer(filter_count, enc4)

        # (4 x 4 x 8N)
        enc6 = self._add_encoding_layer(filter_count, enc5)

        # (2 x 2 x 8N)
        enc7 = self._add_encoding_layer(filter_count, enc6)

        # (1 x 1 x 8N)
        enc8 = self._add_encoding_layer(filter_count, enc7)

        enc9 = self._add_encoding_layer(filter_count, enc8)

        dec0 = self._add_decoding_layer(filter_count, True, enc9)
        dec0 = concatenate([dec0, enc8], axis=self.CONCATENATE_AXIS)

        # デコーダーの作成
        # (2 x 2 x 8N)
        dec1 = self._add_decoding_layer(filter_count, True, dec0)
        dec1 = concatenate([dec1, enc7], axis=self.CONCATENATE_AXIS)

        # (4 x 4 x 8N)
        dec2 = self._add_decoding_layer(filter_count, True, dec1)
        dec2 = concatenate([dec2, enc6], axis=self.CONCATENATE_AXIS)

        # (8 x 8 x 8N)
        dec3 = self._add_decoding_layer(filter_count, True, dec2)
        dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)

        # (16 x 16 x 8N)
        dec4 = self._add_decoding_layer(filter_count, False, dec3)
        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)

        # (32 x 32 x 4N)
        filter_count = first_layer_filter_count*4
        dec5 = self._add_decoding_layer(filter_count, False, dec4)
        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)

        # (64 x 64 x 2N)
        filter_count = first_layer_filter_count*2
        dec6 = self._add_decoding_layer(filter_count, False, dec5)
        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)

        # (128 x 128 x N)
        filter_count = first_layer_filter_count
        dec7 = self._add_decoding_layer(filter_count, False, dec6)
        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)

        # (256 x 256 x output_channel_count)
        dec8 = Activation(activation='relu')(dec7)
        dec8 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        dec8 = Activation(activation='sigmoid')(dec8)

        self.UNET = Model(input=inputs, output=dec8)

    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self, summary=True, to_png=False, png_file=None):
        summary_and_png(self.UNET, summary, to_png, png_file)
        return self.UNET


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


if __name__ == '__main__':

    # U-Netの生成
    network = UNet(
        input_channel_count=3,  # 入力はBGR3チャンネル
        output_channel_count=1,  # 出力はグレースケール1チャンネル
        first_layer_filter_count=64  # 一番初めのConvolutionフィルタ枚数は64
    )
    model = network.get_model(to_png=True, png_file='unet_model.png')
    model.compile(loss=dice_coef_loss, optimizer=Adam(lr=0.01), metrics=[dice_coef])
