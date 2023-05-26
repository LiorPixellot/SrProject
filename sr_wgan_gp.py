"""
In this modified version of the `SrGan` class, I made the following changes:

1. Added the `gradient_penalty` method to compute the gradient penalty for the Wasserstein GAN with gradient penalty loss.

2. Modified the `train_step_dis` method to replace the binary cross-entropy loss with the Wasserstein loss, which is computed as the difference between the means of real and fake scores. Also added the gradient penalty term to the discriminator loss.

3. Changed the `train_step_gen` method to replace the binary cross-entropy loss with the Wasserstein loss for the generative loss. The generative loss is now calculated as the negation of the mean of the discriminator's scores for the generated images.

With these changes, the `SrGan` class now uses the Wasserstein GAN with gradient penalty loss for training the generator and discriminator. The rest of the class remains unchanged, so the perceptual loss and feature loss calculations are still based on the VGG19 model as in the original implementation.
"""



import train
import model
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg

class SrWganGp(train.AbsTrainer):

    def __init__(self, generator, discriminator, train_dir, start_epoch=-1, demo_mode=False):
        super().__init__(generator, discriminator, train_dir, start_epoch, demo_mode)
        self.vgg = model.build_vgg()
        self.k_times_train_dis = 5


    @tf.function
    def gradient_penalty(self, real, fake):
        batch_size = real.shape[0]
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        interpolated_images = epsilon * real + (1 - epsilon) * fake
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images) # Tell the tape to watch the `interpolated` tensor
            validity = self.discriminator(interpolated_images, training=True)
        gradients = gp_tape.gradient(validity, interpolated_images)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gp = tf.reduce_mean((gradients_norm - 1.0)**2)
        return gp

    @tf.function
    def train_step_dis(self, lr_batch, hr_batch):
        with tf.GradientTape() as tape:
            # get fake images from the generator
            fake_images = self.generator(lr_batch, training=False)
            # get the prediction from the discriminator
            real_validity = self.discriminator(hr_batch, training=True)
            fake_validity = self.discriminator(fake_images, training=True)
            # compute the loss
            d_loss_real = tf.reduce_mean(real_validity)
            d_loss_fake = tf.reduce_mean(fake_validity)
            gp = self.gradient_penalty(hr_batch, fake_images)
            d_loss = d_loss_fake - d_loss_real  + 10 * gp

        # compute the gradients
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # optimize the discriminator weights according to the gradients
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        self.dis_loss_metric.update_state(d_loss)



    @tf.function
    def train_step_gen(self, lr_images, hr_images):
        with tf.GradientTape() as tape:
            # get fake images from the generator
            fake_images = self.generator(lr_images,training=True)
            # get the prediction from the discriminator
            predictions = self.discriminator(fake_images, training=False)
            # compute the adversarial loss
            generative_loss = -tf.reduce_mean(predictions)

            sr = preprocess_input_vgg(fake_images)
            hr = preprocess_input_vgg(hr_images)
            sr_features = self.vgg(sr) / 12.75
            hr_features = self.vgg(hr) / 12.75

            # compute the perceptual loss
            mse = tf.keras.losses.MeanSquaredError()
            feature_Loss = mse(sr_features, hr_features)

            # calculate the total generator loss
            perceptual_Loss = generative_loss * 1e-3 + feature_Loss

        # compute the gradients
        grads = tape.gradient(perceptual_Loss, self.generator.trainable_variables)

        # optimize the generator weights with the computed gradients
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        self.perceptual_loss_metric.update_state(perceptual_Loss)
        self.generative_loss_metric.update_state(generative_loss)
        self.feature_loss_metric.update_state(feature_Loss)





