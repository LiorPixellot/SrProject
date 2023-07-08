import train
import tensorflow as tf
import model
class   SrResnet(train.AbsTrainer):

    def __init__(self, train_dir, hr_size ,demo_mode=False):
        super().__init__( train_dir,hr_size, demo_mode)

    @tf.function
    def train_step_gen(self, lr_images, hr_images):
        with tf.GradientTape() as tape:
            sr = self.generator(lr_images, training=True)
            mse = tf.keras.losses.MeanSquaredError()
            loss_value = mse(hr_images, sr)
        gradients = tape.gradient(loss_value, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        self.generative_loss_metric.update_state(loss_value)


    def train_step_dis(self, lr_batch, hr_batch):
            pass



    def create_generator(self):
        return model.load_generator(self.train_dir /"weights" / "generator","resnet_pixel_shuffle")


    def create_discriminator(self):
        return None,None
