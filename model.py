import os
import re

from keras.applications import VGG19
from keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from keras.models import Model
import tensorflow as tf
import common
from keras.layers import UpSampling2D
from keras.layers import LayerNormalization
from keras.layers import concatenate

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
    x = discrimator_bloccks(num_filters, x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)


def discrimator_bloccks(num_filters, x):
    x = discriminator_block(x, num_filters, batchnorm=False)
    x = discriminator_block(x, num_filters, strides=2)
    x = discriminator_block(x, num_filters * 2)
    x = discriminator_block(x, num_filters * 2, strides=2)
    x = discriminator_block(x, num_filters * 4)
    x = discriminator_block(x, num_filters * 4, strides=2)
    x = discriminator_block(x, num_filters * 8)
    x = discriminator_block(x, num_filters * 8, strides=2)
    return x


def discriminator_condition(num_filters=64,hr_size=96):
    # The low resolution image will act as the condition
    condition_in = Input(shape=(hr_size, hr_size, 3))
    # The high resolution image will be the target
    target_in = Input(shape=(hr_size, hr_size, 3))
    x = concatenate([target_in,condition_in], axis=-1)
    x = Lambda(normalize_m11)(x)
    x = discrimator_bloccks(num_filters, x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model([target_in,condition_in], x)

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
    print(dir)
    try:
        model , step =  load_last_weights(dir)
    except Exception as e:
        step = 0
        model =  sr_resnet()
    print("starting generator from step " +str(step))
    return model, step

def load_discriminator(dir,type = "gan"):
    try:
        model, step = load_last_weights(dir)
    except Exception as e:
        step = 0
        if type == "cgan":
            model = discriminator_condition()
        elif type == "pix2pix":
            model = discriminator_pix2pix()
        else:
            model = discriminator()
    print("starting discriminator from epoch " + str(step))
    return model, step


def build_vgg():
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
    return Model(inputs=vgg.inputs, outputs=vgg.layers[20].output)


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def discriminator_pix2pix(hr_size=96):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[hr_size, hr_size, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[hr_size, hr_size, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)