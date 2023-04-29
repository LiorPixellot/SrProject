import train
import model
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg

class SrGan(train.AbsTrainer):

    def __init__(self, generator, discriminator, train_dir, start_epoch=-1, demo_mode=False,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.09)):
        super().__init__(generator, discriminator, train_dir, start_epoch, demo_mode, optimizer)
        self.vgg = model.build_vgg()



    @tf.function
    def train_step_dis(self, hr_batch, lr_batch):
        # train the discriminator
        with tf.GradientTape() as tape:
            # get the discriminator predictions
            predictions_real = self.discriminator(hr_batch ,training=True)
            predictions_fake = self.discriminator(self.generator(lr_batch,training=False), training=True)

            combined_predicted_labels = tf.concat([predictions_fake, predictions_real], axis=0)
            combined_real_labels = tf.concat([tf.zeros_like(predictions_fake), tf.ones_like(predictions_real)], axis=0)
            # compute the loss
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            d_loss = bce(combined_real_labels,combined_predicted_labels)

        # compute the gradients
        grads = tape.gradient(d_loss,self.discriminator.trainable_variables)

        # optimize the discriminator weights according to the
        # gradients computed

        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables) )

        self.dis_loss_metric.update_state(d_loss)


    @tf.function
    def train_step_gen(self, lr_images, hr_images):
        with tf.GradientTape() as tape:
            # get fake images from the generator
            fakeImages = self.generator(lr_images,training=True)
            # get the prediction from the discriminator
            predictions = self.discriminator(fakeImages, training=False)
            # compute the adversarial loss
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            generative_loss = bce(tf.ones_like(predictions), predictions)

            sr = preprocess_input_vgg(fakeImages)
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



