import train
import model
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg

class pix2pix(train.AbsTrainer):

    def __init__(self, train_dir, l1_loss_factor ,demo_mode=False):
        super().__init__( train_dir, demo_mode)
        #tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)
        self.l1_loss_factor = l1_loss_factor



    @tf.function
    def train_step_dis(self, lr_batch, hr_batch):
        # Calculate the size of hr_batch
        hr_size = hr_batch.shape[1:3]  # Assuming hr_batch has shape (batch_size, height, width, channels)

        # Resize low-resolution images to the same size as high-resolution images
        lr_resized = tf.image.resize(lr_batch, hr_size,method=tf.image.ResizeMethod.BICUBIC)

        with tf.GradientTape() as tape:
            # get fake images from the generator
            fake_images = self.generator(lr_resized, training=False)
            # get the prediction from the discriminator
            real_validity = self.discriminator([hr_batch, lr_resized], training=True)
            fake_validity = self.discriminator([fake_images, lr_resized], training=True)

            combined_predicted_labels = tf.concat([fake_validity, real_validity], axis=0)
            combined_real_labels = tf.concat([tf.zeros_like(fake_validity), tf.ones_like(real_validity)], axis=0)
            # compute the loss
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            d_loss = bce(combined_real_labels, combined_predicted_labels)

        # compute the gradients
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # optimize the discriminator weights according to the gradients
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        self.dis_loss_metric.update_state(d_loss)

    @tf.function
    def train_step_gen(self, lr_images, hr_images):
        # Calculate the size of hr_batch
        hr_size = hr_images.shape[1:3]  # Assuming hr_batch has shape (batch_size, height, width, channels)

        # Resize low-resolution images to the same size as high-resolution images
        lr_resized = tf.image.resize(lr_images, hr_size,method=tf.image.ResizeMethod.BICUBIC)
        with tf.GradientTape() as tape:
            # get fake images from the generator
            fake_images = self.generator(lr_resized,training=True)
            # get the prediction from the discriminator
            predictions = self.discriminator([fake_images,lr_resized], training=False)
            # compute the adversarial loss
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            generative_loss = bce(tf.ones_like(predictions), predictions)

            # Mean absolute error
            l1_loss = tf.reduce_mean(tf.abs(hr_images - fake_images))

            # calculate the total generator loss
            perceptual_Loss = generative_loss + l1_loss*self.l1_loss_factor

        # compute the gradients
        grads = tape.gradient(perceptual_Loss, self.generator.trainable_variables)

        # optimize the generator weights with the computed gradients
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        self.perceptual_loss_metric.update_state(perceptual_Loss)
        self.generative_loss_metric.update_state(generative_loss)






    def create_generator(self):
        return model.load_generator(self.train_dir /"weights" / "generator","unet")


    def create_discriminator(self):
        return model.load_discriminator(self.train_dir /"weights" / "discriminator","pix2pix")

