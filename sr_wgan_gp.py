import train
import tensorflow as tf

class SrWganGp(train.AbsTrainer):
    def __init__(self, generator, discriminator, train_dir, start_epoch=-1, demo_mode=False,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.09)):
        super().__init__(generator, discriminator, train_dir, start_epoch, demo_mode, optimizer)
        self.gp_lambda = 10  # The recommended value is 10

    def train_step(self, lr_batch, hr_batch):
        self.train_step_dis(hr_batch, lr_batch)
        self.train_step_gen(lr_batch, hr_batch)

    @tf.function
    def train_step_gen(self, lr_images, hr_images):
        with tf.GradientTape() as tape:
            # Get fake images from the generator
            fake_images = self.generator(lr_images)

            # Get the prediction from the discriminator
            predictions = self.discriminator(fake_images)

            loss = -tf.reduce_mean(predictions)

        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

    @tf.function
    def train_step_dis(self, hr_batch, lr_batch):
        with tf.GradientTape() as tape:
            fake_images = self.generator(lr_batch)

            predictions_real = self.discriminator(hr_batch)
            predictions_fake = self.discriminator(fake_images)

            alpha = tf.random.uniform(shape=[hr_batch.shape[0], 1, 1, 1], minval=0, maxval=1)
            inter_sample = alpha * hr_batch + (1 - alpha) * fake_images

            with tf.GradientTape() as tape_gp:
                tape_gp.watch(inter_sample)
                inter_score = self.discriminator(inter_sample)

            gp_gradients = tape_gp.gradient(inter_score, inter_sample)
            gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis=[1, 2, 3]))
            gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)

            loss = tf.reduce_mean(predictions_fake) - tf.reduce_mean(predictions_real) + gp * self.gp_lambda

        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))