import os
import re

from keras.applications import VGG19
from keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from keras.models import Model
import tensorflow as tf
from keras.layers import UpSampling2D
from keras.layers import LayerNormalization
from keras.layers import concatenate
from keras.applications.inception_v3 import InceptionV3



def upsample(x_in, num_filters,model_name = ""):
    x = UpSampling2D(size=(2, 2), interpolation='nearest',name=model_name+"UpSampling2D")(x_in)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    return PReLU(shared_axes=[1, 2], name=model_name+'prelu_up')(x)



def res_block(x_in, num_filters,block_num,model_name = "", momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2], name=f'{model_name}PReLU_layer_resblock_{block_num}')(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add(name=f'{model_name}add_layer_resblock_{block_num}')([x_in, x])
    return x


def sr_resnet(num_filters=64, num_res_blocks=16):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize_01)(x_in)
    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)
    for i in range(num_res_blocks):
        x = res_block(x, num_filters,i)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add(name='add_layer')([x_1, x])
    x = upsample(x, num_filters * 4,"first")
    x = upsample(x, num_filters * 4,"second")
    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11)(x)
    return Model(x_in, x)

def progressive_sr_resnet(low_res_model=None, num_filters=64, num_res_blocks=16):

    """
    Creates a new super-resolution ResNet model which incorporates the layers of a lower resolution model (if provided).
    The layers of the low resolution model are set to non-trainable.
    The new layers are initialized but not trained.
    """
    name = ""
    # Start with the layers from the low resolution model, if provided
    if low_res_model is not None:
        name = "sec"
        # Set the layers of the low resolution model to non-trainable
        for layer in low_res_model.layers:
            layer.trainable = False

        x_in = low_res_model.input
        x = low_res_model.output
    else:
        name = "first"
        x_in = Input(shape=(None, None, 3))
        x = x_in
    x = Lambda(normalize_01, name=name+'normalize_layer')(x)

    # Add the new layers for the higher resolution
    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2], name=name+'prelu')(x)

    for i in range(num_res_blocks):
        x = res_block(x, num_filters,i,name)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add(name=name+'add')([x_1, x])  # added name here
    x = upsample(x, num_filters * 4,name)
    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)
    x = Lambda(denormalize_m11, name=name+'denormalize_layer')(x)

    return Model(x_in, x)

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



def load_generator(dir,type = "resnet",prev_model = None):
    print(dir)
    model = None
    try:
        model , step =  load_last_weights(dir)
    except Exception as e:
        step = 0
        if type == "progressive":
            model =  progressive_sr_resnet(prev_model)
        else:
            model =  sr_resnet()
    print("starting generator from step " +str(step))
    return model, step

def load_discriminator(dir,type = "gan",hr_size = 96,num_of_filters= 64):
    try:
        model, step = load_last_weights(dir)
    except Exception as e:
        step = 0
        if type == "cgan":
            model = discriminator_condition(num_of_filters,hr_size)
        elif type == "patch_gan":
            model = patch_gan(hr_size)
        elif type == "patch_gan_no_condiation":
            model = patch_gan_no_condition(hr_size)
        else:
            model = discriminator(num_of_filters,hr_size)
    print("starting discriminator from epoch " + str(step))
   # tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)
    return model, step


def build_vgg(hr_size = 96):
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(hr_size, hr_size, 3))
    return Model(inputs=vgg.inputs, outputs=vgg.layers[20].output)

def build_inception_model():
        model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        model.trainable = False
        return model


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

def patch_gan_no_condition(hr_size=96):
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[hr_size, hr_size, 3], name='input_image')
    last = patch_gan_layers(initializer, inp)
    return tf.keras.Model(inputs=[inp], outputs=last)




def patch_gan(hr_size=96):
  initializer = tf.random_normal_initializer(0., 0.02)
  inp = tf.keras.layers.Input(shape=[hr_size, hr_size, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[hr_size, hr_size, 3], name='target_image')
  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)
  last = patch_gan_layers(initializer, x)
  return tf.keras.Model(inputs=[inp, tar], outputs=last)


def patch_gan_layers(initializer, x):
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)
    return last




def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5




