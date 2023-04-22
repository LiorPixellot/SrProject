import train
import model
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg

class SrGan(train.AbsTrainer):

    def __init__(self, generator, discriminator, train_dir, start_epoch=-1, demo_mode=False,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.09)):
        super().__init__(generator, discriminator, train_dir, start_epoch, demo_mode, optimizer)
        self.vgg = model.build_vgg()

    def train_step(self, lr_batch, hr_batch):
        dis_loss = self.train_step_dis(hr_batch, lr_batch)
        perceptual_Loss, generative_loss, feature_Loss = self.train_step_gen(lr_batch, hr_batch)
        return {"dis_loss": dis_loss, "perceptual_Loss": perceptual_Loss,
                "generative_loss": generative_loss, "feature_Loss": feature_Loss}



    @tf.function
    def train_step_dis(self, hr_batch, lr_batch):
        # train the discriminator
        with tf.GradientTape() as tape:
            # get the discriminator predictions
            predictions_real = self.discriminator(hr_batch)
            predictions_fake = self.discriminator(self.generator(lr_batch))
            tf.concat([predictions_real, predictions_fake], 0)
            combined_predicted_labels = tf.concat([predictions_fake, predictions_real], axis=0)
            combined_real_labels = tf.concat([tf.zeros_like(predictions_fake), tf.ones_like(predictions_real)], axis=0)
            # compute the loss
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            dLoss = bce(combined_predicted_labels, combined_real_labels)

        # compute the gradients
        grads = tape.gradient(dLoss,self.discriminator.trainable_variables)

        # optimize the discriminator weights according to the
        # gradients computed

        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables) )
        return dLoss

    @tf.function
    def train_step_gen(self, lr_images, hr_images):
        with tf.GradientTape() as tape:
            # get fake images from the generator
            fakeImages = self.generator(lr_images)

            # get the prediction from the discriminator
            predictions = self.discriminator(fakeImages)

            # compute the adversarial loss
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            generative_loss = bce(tf.ones_like(predictions), predictions)

            # compute the normalized vgg outputs
            sr_vgg = self.vgg(preprocess_input_vgg(fakeImages))
            hr_vgg = self.vgg(preprocess_input_vgg(hr_images))
            # compute the perceptual loss
            mse = tf.keras.losses.MeanSquaredError()
            feature_Loss = mse(hr_vgg, sr_vgg)

            # calculate the total generator loss
            perceptual_Loss = generative_loss * 1e-3 + feature_Loss

        # compute the gradients
        grads = tape.gradient(perceptual_Loss, self.generator.trainable_variables)

        # optimize the generator weights with the computed gradients
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        return perceptual_Loss, generative_loss, feature_Loss



