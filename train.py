import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import common
import display_handler
import tensorflow as tf
from keras.applications.vgg19 import preprocess_input

import model

#TODO LB
AMOUNT_TO_TAKE = -1

class GeneratorTrainer:
    def __init__(self,generator,train_dir,start_epoch = -1,optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.09),loss = tf.keras.losses.MeanSquaredError()):
        self.train_dir = train_dir
        self.loss = loss
        self.optimizer = optimizer
        self.generator = generator
        self.start_epoch = start_epoch + 1
    def fit(self,train_dataset,test_dataset,epochs = 50):
        psnr_vals = []
        for epoch in tqdm(range(epochs)):
            real_epch = epoch + self.start_epoch
            for lr,hr in train_dataset.take(AMOUNT_TO_TAKE).cache():
                self.train_step(lr,hr)
            print("epoc "+ str(real_epch))
            psnr_vals.append( self.eval(test_dataset,real_epch))
            self.generator.save(self.train_dir+"/weights/generator/"+str(real_epch)+'.h5')
            print("epoc_" + str(real_epch) + "_psnr_" + str(psnr_vals[-1]))
            plt.plot(psnr_vals)
            plt.savefig(self.train_dir+"/psnr_epoch_"+ str(real_epch)  + ".png")
            plt.close()

    def eval(self,test_dataset,epoch):
        cont = 0
        psnr_vals = []
        for lr, hr in test_dataset:
            cont+=1
            display_handler.display_hr_lr(self.train_dir,self.generator,hr,lr,epoch,cont)
            psnr_vals.append(common.psnr(self.generator(lr), hr))
        return np.mean(psnr_vals)



    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:
            sr = self.generator(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.generator.trainable_variables))







class SrGanTrainer:
    def __init__(self,generator,discriminator,train_dir,start_epoch = -1,optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.09),loss = tf.keras.losses.MeanSquaredError()):
        self.train_dir = train_dir
        self.loss = loss
        self.optimizer = optimizer
        self.generator = generator
        self.discriminator = discriminator
        self.start_epoch = start_epoch + 1
        self.vgg = model.build_vgg()



    def fit(self,train_dataset,test_dataset,epochs = 50):
        psnr_vals = []
        for epoch in tqdm(range(epochs)):
            real_epch = epoch + self.start_epoch
            for lr,hr in train_dataset.take(AMOUNT_TO_TAKE).cache():
                self.train_step(lr,hr)
            psnr_vals.append(self.eval(test_dataset,real_epch))
            self.generator.save(self.train_dir+"/weights/generator/"+str(real_epch)+'.h5')
            self.discriminator.save(self.train_dir + "/weights/discriminator/" + str(real_epch) + '.h5')
            print("epoc_" + str(real_epch) +"_psnr_" +str(psnr_vals[-1]))
            plt.plot(psnr_vals)
            plt.savefig(self.train_dir + "/psnr_epoch_" + str(real_epch) + ".png")
            plt.close()


   # @tf.function
    def train_step(self, lr_batch, hr_batch):
        dis_loss = self.train_step_dis(hr_batch, lr_batch)
        perceptual_Loss, generative_loss, feature_Loss = self.train_step_gen(lr_batch, hr_batch)
        return {"dis_loss": dis_loss, "perceptual_Loss": perceptual_Loss,
         "generative_loss": generative_loss, "feature_Loss": feature_Loss}

    def eval(self,test_dataset,epoch):
        cont = 0
        psnr_vals = []

        for lr, hr in test_dataset:
            cont+=1
            display_handler.display_hr_lr(self.train_dir,self.generator,hr,lr,epoch,cont)
            psnr_vals.append(common.psnr(self.generator(lr),hr))
        return np.mean(psnr_vals)




    #@tf.function
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
        grads = tape.gradient(dLoss,
                              self.discriminator.trainable_variables)

        # optimize the discriminator weights according to the
        # gradients computed

        self.optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )
        return dLoss

   # @tf.function
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
            sr_vgg = self.vgg(fakeImages)
            hr_vgg = self.vgg(hr_images)
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
