import pathlib
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
import display_handler
import model
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
from numpy import iscomplexobj
import time


class AbsTrainer(ABC):
    def __init__(self,
                 train_dir: str,
                 hr_size = 64,
                 demo_mode: bool = False,
                 ):
        self.train_dir = pathlib.Path(train_dir)
        self.hr_size = hr_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5,beta_2=0.9)
        self.generator, self.start_step = self.create_generator()
        self.real_step = self.start_step
        self.discriminator,_ =  self.create_discriminator()
        self._settings = self._get_settings_according_to_mode(demo_mode)
        self.inception_model = model.build_inception_model()
        self.creat_run_dirs()
        self.add_metrics()
        self.k_times_train_dis = 2
        self.cur_k_val = 0



    def add_metrics(self):
        self.dis_loss_metric = tf.keras.metrics.Mean()
        self.perceptual_loss_metric = tf.keras.metrics.Mean()
        self.generative_loss_metric = tf.keras.metrics.Mean()
        self.feature_loss_metric = tf.keras.metrics.Mean()
        self.log_writer = tf.summary.create_file_writer( str((self.train_dir/"logs")))

    def creat_run_dirs(self):
        self.images_path = self.train_dir / "images"
        self.images_path.mkdir(parents=True, exist_ok=True)



    def _get_settings_according_to_mode(self, demo_mode: bool) -> dict:
        demo_settings = {'steps_to_save_progress': 1,
                         'steps_to_eval':1}

        normal_mode = {'steps_to_save_progress': 100000,
                       'steps_to_eval':100000}
        settings = demo_settings if demo_mode else normal_mode
        return settings

    def fit(self, datasets, steps_to_train: int = 1000000):
        while self.real_step <= steps_to_train:
            for step_count_curr_epoch, (lr_batch, hr_batch) in enumerate(datasets.train_dataset):
                start_time = time.time()  # Record the start time
                self.train_step(lr_batch, hr_batch)
                self.save_progress(datasets)
                end_time = time.time()  # Record the end time
                time_taken = end_time - start_time  # Calculate the time difference
                print(f"step_{self.real_step}_time_taken_{time_taken:.2f}s")
                self.real_step += len(lr_batch)
                if self.real_step >= steps_to_train:
                    break

    def save_weights(self):
        gen_path = self.train_dir / "weights/generator"
        gen_path.mkdir(parents=True, exist_ok=True)
        self.generator.save(gen_path / f"{self.real_step}.h5")

        if self.discriminator is not None:
            dis_path = self.train_dir / "weights/discriminator"
            dis_path.mkdir(parents=True, exist_ok=True)
            self.discriminator.save(dis_path / f"{self.real_step }.h5")


    def save_progress(self,datasets):
        self.log_losses()
        modulo =  self.real_step   % self._settings['steps_to_save_progress']
        if (modulo <= 0 and  self.real_step  > self.start_step):
            print("save_progress")
            self.save_weights()
            self.save_display_examples(datasets.validation_dataset)

        modulo = self.real_step % self._settings['steps_to_eval']
        print(modulo)
        if (modulo <= 0 and self.real_step > self.start_step):
            self.eval(datasets)

    def log_losses(self):
        #whatc with tensorboard --logdir logs  --bind_all
        with self.log_writer.as_default():
            tf.summary.scalar("dis_loss", self.dis_loss_metric.result(),step=self.real_step)
            tf.summary.scalar("perceptual_Loss", self.perceptual_loss_metric.result(),step=self.real_step)
            tf.summary.scalar("generative_loss", self.generative_loss_metric.result(),step=self.real_step)
            tf.summary.scalar("feature_Loss", self.feature_loss_metric.result(),step=self.real_step)
        self.dis_loss_metric.reset_states()
        self.perceptual_loss_metric.reset_states()
        self.generative_loss_metric.reset_states()
        self.feature_loss_metric.reset_states()



    def _calculate_fid_ssim_psnr_cur_step(self,datasets):

        real_features = []
        gen_features = []
        psnr_vals = []
        ssim_vals = []
        for lr, hr in datasets.validation_dataset:
            # Calculate the size of hr_batch
            # Resize low-resolution images to the same size as high-resolution images
            fake_hr = self.generator(lr, training=False)
            real_image_resized = preprocess_input_inception(tf.image.resize(hr, (299, 299)))
            generated_image_resized = preprocess_input_inception(tf.image.resize(fake_hr, (299, 299)))

            hr = hr / 255.0
            fake_hr = fake_hr / 255.0

            psnr_vals.append(tf.image.psnr(hr, fake_hr, max_val=1.0))
            ssim_vals.append(tf.image.ssim(hr, fake_hr, max_val=1.0))
            real_features.append(np.squeeze(self.inception_model(real_image_resized)))
            gen_features.append(np.squeeze(self.inception_model(generated_image_resized)))

        # calculate mean and covariance statistics
        if(datasets.batch_size == 1):
            mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
            mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
        else:
             concatenated_real_features =  np.concatenate(real_features, axis=0)
             concatenated_gen_features = np.concatenate(gen_features, axis=0)
             mu1, sigma1 = np.mean(concatenated_real_features, axis=0), np.cov(concatenated_real_features, rowvar=False)
             mu2, sigma2 = np.mean(concatenated_gen_features, axis=0), np.cov(concatenated_gen_features, rowvar=False)

        print("sigma1 shape:", sigma1.shape)
        print("sigma1 type:", type(sigma1))
        print("sigma2 shape:", sigma2.shape)
        print("sigma2 type:", type(sigma2))
        print("Dot product shape:", np.dot(sigma1, sigma2).shape)
        print("Dot product type:", type(np.dot(sigma1, sigma2)))
        
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(np.dot(sigma1, sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

        print("PSNR",np.mean(psnr_vals))
        print("sigma1 shape:", fid)
        print("SSIM:", np.mean(ssim_vals))
        with self.log_writer.as_default():
            tf.summary.scalar("PSNR",np.mean(psnr_vals),step=self.real_step)
            tf.summary.scalar("FID", fid,step=self.real_step)
            tf.summary.scalar("SSIM", np.mean(ssim_vals),step=self.real_step)




    def eval(self,datasets):
        print(f"eval_step_{ self.real_step }")
        self._calculate_fid_ssim_psnr_cur_step(datasets)



    def train_step(self, lr_batch, hr_batch):
        self.cur_k_val += 1
        if self.cur_k_val % (self.k_times_train_dis) > 0:
            print("train dis")
            self.train_step_dis(lr_batch,hr_batch)
        else:
            print("train gen")
            self.train_step_gen(lr_batch, hr_batch)

    def save_display_examples(self, test_dataset, num_images = 30):
        count = 0
        for idx, (lr_batch, hr_batch) in enumerate(test_dataset):
            for i in range(len(lr_batch)):  # Assuming lr_batch and hr_batch have the same number of images
                lr_image = lr_batch[i]
                hr_image = hr_batch[i]
                #display_handler.dump_hr_lr_side_by_side(self.train_dir, self.generator, hr_image, lr_image, self.real_step,
                #                              idx * len(lr_batch) + i)
                display_handler.dump_hr_lr_images(self.train_dir, self.generator, hr_image, lr_image, self.real_step,
                                              idx * len(lr_batch) + i)
                count += 1
                if count >= num_images:
                    return

    @abstractmethod
    def train_step_dis(self, hr_batch, lr_batch):
        pass

    @abstractmethod
    def train_step_gen(self, lr_batch, hr_batch):
        pass


    @abstractmethod
    def create_generator(self):
        pass

    @abstractmethod
    def create_discriminator(self):
        pass


