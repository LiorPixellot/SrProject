import train
import tensorflow as tf

class SrResnet(train.AbsTrainer):

    @tf.function
    def train_step_gen(self, lr_images, hr_images):
        with tf.GradientTape() as tape:
            sr = self.generator(lr_images, training=True)
            mse = tf.keras.losses.MeanSquaredError()
            loss_value = mse(hr_images, sr)
        gradients = tape.gradient(loss_value, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        self.generative_loss_metric.update_state(loss_value)


    def train_step_dis(self, hr_batch, lr_batch):
            pass