import train
import model
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg

class SrWPix2PixHr(train.AbsTrainer):

    def __init__(self, train_dir, hr_size,l2_loss_factor,gp_factor,prev_gen = None,demo_mode=False):
        super().__init__( train_dir,hr_size, demo_mode,prev_gen)
        self.vgg = model.build_vgg(hr_size)
        self.l2_loss_factor = l2_loss_factor
        self.gp_factor = gp_factor
        self.prev_gen = prev_gen


    @tf.function
    def gradient_penalty(self, real, fake):
        batch_size = real.shape[0]
        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0, maxval=1)
        interpolated_images = epsilon * real + (1 - epsilon) * fake
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_images) # Tell the tape to watch the `interpolated` tensor
            validity = self.discriminator(interpolated_images, training=True)
        gradients = gp_tape.gradient(validity, interpolated_images)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])+ 1e-10)
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
            
            d_loss_real = tf.reduce_mean(real_validity)
            d_loss_fake = tf.reduce_mean(fake_validity)
            gp = self.gradient_penalty(hr_batch, fake_images)


            w_loss = d_loss_fake - d_loss_real
            d_loss = w_loss  + self.gp_factor*gp

          

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
            generative_loss = -tf.reduce_mean(predictions)
            
            sr = preprocess_input_vgg(fake_images)
            hr = preprocess_input_vgg(hr_images)
            sr_features = self.vgg(sr) / 12.75
            hr_features = self.vgg(hr) / 12.75

            # compute the perceptual loss
            mse = tf.keras.losses.MeanSquaredError()
            feature_Loss = mse(sr_features, hr_features)

            #l2_loss
            l2_loss = tf.reduce_mean(tf.square(hr_images - fake_images))

            # calculate the total generator loss
            perceptual_Loss = generative_loss * 1e-3 + feature_Loss  + l2_loss*self.l2_loss_factor

        # compute the gradients
        grads = tape.gradient(perceptual_Loss, self.generator.trainable_variables)

        # optimize the generator weights with the computed gradients
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        self.perceptual_loss_metric.update_state(perceptual_Loss)
        self.generative_loss_metric.update_state(generative_loss)
        self.feature_loss_metric.update_state(feature_Loss)




    def create_generator(self):
        return model.load_generator(self.train_dir /"weights" / "generator","progressive",self.prev_gen)


    def create_discriminator(self):
        return model.load_discriminator(self.train_dir /"weights" / "discriminator","patch_gan_no_condiation",self.hr_size)

