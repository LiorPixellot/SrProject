import os
import re

from keras.applications import VGG19
from keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from keras.models import Model
import tensorflow as tf
import common
from keras.layers import UpSampling2D

from common import pixel_shuffle, normalize_01, normalize_m11, denormalize_m11
def upsample(x_in, num_filters):
    x = UpSampling2D(size=(2, 2), interpolation='nearest')(x_in)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    return PReLU(shared_axes=[1, 2])(x)


def res_block(x_in, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x


def sr_resnet(num_filters=64, num_res_blocks=16):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)

    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample(x, num_filters * 4)
    x = upsample(x, num_filters * 4)

    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11)(x)

    return Model(x_in, x)


generator = sr_resnet


def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)


def discriminator(num_filters=64,hr_size=96):
    x_in = Input(shape=(hr_size, hr_size, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)

    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)

    x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)

    x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)



def load_last_weights(dir_path):
    file_list = os.listdir(dir_path)

    regex = r'^(\d+)\.'

    numbers = []
    for filename in file_list:
        match = re.match(regex, filename)
        if match:
            number = int(match.group(1))
            numbers.append(number)
    max_number = max(numbers)
    load_weight = os.path.join(dir_path, str(max_number))+'.h5'
    print("loading weights "+ load_weight)
    return tf.keras.models.load_model(load_weight) , max_number



def load_generator(dir):
    try:
        model , step =  load_last_weights(dir)
    except Exception as e:
        step = -1
        model =  sr_resnet()
    print("starting generator from step " +str(step))
    return model, step

def load_discriminator(dir):
    try:
        model, step = load_last_weights(dir)
    except Exception as e:
        step = -1
        model = discriminator()
    print("starting discriminator from epoch " + str(step))
    return model, step


def build_vgg():
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
    return Model(inputs=vgg.inputs, outputs=vgg.layers[20].output)