import train
import tensorflow as tf

class SrWganGp(train.AbsTrainer):
    def __init__(self, generator, discriminator, train_dir, start_epoch=-1, demo_mode=False,
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.09)):
        super().__init__(generator, discriminator, train_dir, start_epoch, demo_mode, optimizer)





        def train_step(self, lr_batch, hr_batch):
            dis_loss = self.train_step_dis(hr_batch, lr_batch)
            perceptual_Loss, generative_loss, feature_Loss = self.train_step_gen(lr_batch, hr_batch)
            return {"dis_loss": dis_loss, "perceptual_Loss": perceptual_Loss,
                    "generative_loss": generative_loss, "feature_Loss": feature_Loss}




        def train_step_gen(args):
            with tf.GradientTape() as tape:
                z = tf.random.uniform([args.batch_size, args.noise_dim], -1.0, 1.0)
                fake_sample = args.gen(z)
                fake_score = args.dis(fake_sample)
                loss = - tf.reduce_mean(fake_score)
            gradients = tape.gradient(loss, args.gen.trainable_variables)
            args.gen_opt.apply_gradients(zip(gradients, args.gen.trainable_variables))
            args.gen_loss(loss)

        def train_step_dis(args, real_sample):
            batch_size = real_sample.get_shape().as_list()[0]
            with tf.GradientTape() as tape:
                z = tf.random.uniform([batch_size, args.noise_dim], -1.0, 1.0)
                fake_sample = args.gen(z)
                real_score = args.dis(real_sample)
                fake_score = args.dis(fake_sample)

                alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
                inter_sample = fake_sample * alpha + real_sample * (1 - alpha)
                with tf.GradientTape() as tape_gp:
                    tape_gp.watch(inter_sample)
                    inter_score = args.dis(inter_sample)
                gp_gradients = tape_gp.gradient(inter_score, inter_sample)
                gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis=[1, 2, 3]))
                gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)

                loss = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp * args.gp_lambda

            gradients = tape.gradient(loss, args.dis.trainable_variables)
            args.dis_opt.apply_gradients(zip(gradients, args.dis.trainable_variables))

            args.dis_loss(loss)
            args.adv_loss(loss - gp * args.gp_lambda)



