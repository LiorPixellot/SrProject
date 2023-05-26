import train
import model
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg

class SrGan(train.AbsTrainer):

    def __init__(self, generator, discriminator, train_dir, start_epoch=-1, demo_mode=False):
        super().__init__(generator, discriminator, train_dir, start_epoch, demo_mode)
        self.vgg = model.build_vgg()



    @tf.function
    def train_step_dis(self, lr_batch, hr_batch):
        with tf.GradientTape() as tape:
            # get fake images from the generator
            fake_images = self.generator(lr_batch, training=False)
            # get the prediction from the discriminator
            real_validity = self.discriminator(hr_batch, training=True)
            fake_validity = self.discriminator(fake_images, training=True)

            combined_predicted_labels = tf.concat([fake_validity, real_validity], axis=0)
            combined_real_labels = tf.concat([tf.zeros_like(fake_validity), tf.ones_like(real_validity)], axis=0)
            # compute the loss
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            d_loss = bce(combined_real_labels,combined_predicted_labels)

        # compute the gradients
        grads = tape.gradient(d_loss,self.discriminator.trainable_variables)
        # optimize the discriminator weights according to the gradients
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables) )

        self.dis_loss_metric.update_state(d_loss)


    @tf.function
    def train_step_gen(self, lr_images, hr_images):
        with tf.GradientTape() as tape:
            # get fake images from the generator
            fake_images = self.generator(lr_images,training=True)
            # get the prediction from the discriminator
            predictions = self.discriminator(fake_images, training=False)
            # compute the adversarial loss
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            generative_loss = bce(tf.ones_like(predictions), predictions)

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



