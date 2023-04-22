import train
import tensorflow as tf

class SrResnet(train.AbsTrainer):
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            sr = self.generator(lr, training=True)
            mse = tf.keras.losses.MeanSquaredError()
            loss_value = mse(hr, sr)
        gradients = tape.gradient(loss_value, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

