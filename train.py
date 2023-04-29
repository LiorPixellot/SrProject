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
                 generator: tf.keras.Model,
                 discriminator: tf.keras.Model,
                 train_dir: str,
                 real_step : int = -1,
                 demo_mode: bool = False,
                 optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.09)):

        self.real_step = real_step +1
        self.train_dir = pathlib.Path(train_dir)
        self.optimizer = optimizer
        self.generator = generator
        self.discriminator = discriminator
        self._settings = self._get_settings_according_to_mode(demo_mode)
        self.inception_model = self._build_inception_model()
        self.creat_run_dirs()
        self.add_metrics()

    def add_metrics(self):
        self.dis_loss_metric = tf.keras.metrics.Mean()
        self.perceptual_loss_metric = tf.keras.metrics.Mean()
        self.generative_loss_metric = tf.keras.metrics.Mean()
        self.feature_loss_metric = tf.keras.metrics.Mean()
        self.log_writer = tf.summary.create_file_writer( str((self.train_dir/"logs")))

    def creat_run_dirs(self):
        self.psnr_plots_path = self.train_dir / "psnr_plots"
        self.fid_plots_path = self.train_dir / "fid_plots"
        self.images_path = self.train_dir / "images"
        self.ssim_plots_path = self.train_dir / "ssim_plots"

        self.psnr_plots_path.mkdir(parents=True, exist_ok=True)
        self.fid_plots_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.ssim_plots_path.mkdir(parents=True, exist_ok=True)


    def _build_inception_model(self):
        model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        model.trainable = False
        return model


    def _get_settings_according_to_mode(self, demo_mode: bool) -> dict:
        demo_settings = {'steps_to_save_progress': 1,
                         'steps_to_eval':1}

        normal_mode = {'steps_to_save_progress': 5000,
                       'steps_to_eval':60000}
        settings = demo_settings if demo_mode else normal_mode
        return settings

    def fit(self, datasets, steps_to_train: int = 1000000):
        for self.real_step in range(self.real_step, steps_to_train):
            for step_count_curr_epoch, (lr_batch, hr_batch) in enumerate(datasets.train_dataset):
                start_time = time.time()  # Record the start time
                self.train_step(lr_batch, hr_batch)
                self.save_progress(datasets)
                end_time = time.time()  # Record the end time
                time_taken = end_time - start_time  # Calculate the time difference
                print(f"step_{self.real_step}_time_taken_{time_taken:.2f}s")

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
        if (modulo == 0 and  self.real_step  > 0):
            print("save_progress")
            self.save_weights()
            self.save_display_examples(datasets.validation_dataset)

        modulo = self.real_step % self._settings['steps_to_eval']
        if (modulo == 0 and self.real_step > 0):
            self.eval(datasets.validation_dataset)

    def log_losses(self):
        #whatc with tensorboard --logdir logs
        with self.log_writer.as_default():
            tf.summary.scalar("dis_loss", self.dis_loss_metric.result(),step=self.real_step)
            tf.summary.scalar("perceptual_Loss", self.perceptual_loss_metric.result(),step=self.real_step)
            tf.summary.scalar("generative_loss", self.generative_loss_metric.result(),step=self.real_step)
            tf.summary.scalar("feature_Loss", self.feature_loss_metric.result(),step=self.real_step)
        self.dis_loss_metric.reset_states()
        self.perceptual_loss_metric.reset_states()
        self.generative_loss_metric.reset_states()
        self.feature_loss_metric.reset_states()


    def _calculate_fid_ssim_psnr_cur_step(self,test_dataset):

        real_features = []
        gen_features = []
        psnr_vals = []
        ssim_vals = []
        for lr, hr in test_dataset:
            fake_hr = self.generator(lr, training=False)
            real_image_resized = preprocess_input_inception(tf.image.resize(hr, (299, 299)))
            generated_image_resized = preprocess_input_inception(tf.image.resize(fake_hr, (299, 299)))
            psnr_vals.append(tf.image.psnr(hr, fake_hr, max_val=255))
            ssim_vals.append(tf.image.ssim(hr, fake_hr, max_val=1.0))
            real_features.append(np.squeeze(self.inception_model(real_image_resized)))
            gen_features.append(np.squeeze(self.inception_model(generated_image_resized)))

        # calculate mean and covariance statistics
        concatenated_real_features =  np.concatenate(real_features, axis=0)
        concatenated_gen_features = np.concatenate(gen_features, axis=0)
        mu1, sigma1 = np.mean(concatenated_real_features, axis=0), np.cov(concatenated_real_features, rowvar=False)
        mu2, sigma2 = np.mean(concatenated_gen_features, axis=0), np.cov(concatenated_gen_features, rowvar=False)

        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(np.dot(sigma1, sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


        with self.log_writer.as_default():
            tf.summary.scalar("PSNR",np.mean(psnr_vals),step=self.real_step)
            tf.summary.scalar("FID", fid,step=self.real_step)
            tf.summary.scalar("SSIM", np.mean(ssim_vals),step=self.real_step)



    def _calculate_fid_cur_step(self,test_dataset):
        print("_calculate_fid_cur_step")
        real_features = []
        gen_features = []
        for lr, hr in test_dataset:
            real_image_resized = preprocess_input_inception(tf.image.resize(hr, (299, 299)))
            generated_image_resized = preprocess_input_inception(tf.image.resize(self.generator(lr), (299, 299)))

            real_features.append(np.squeeze(self.inception_model(real_image_resized)))
            gen_features.append(np.squeeze(self.inception_model(generated_image_resized)))

        # calculate mean and covariance statistics
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(np.dot(sigma1, sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        self._fid_vals.append(fid)


    def eval(self,test_dataset):
        print(f"eval_step_{ self.real_step }")
        self._calculate_fid_ssim_psnr_cur_step(test_dataset)



    def train_step(self, lr_batch, hr_batch):
         self.train_step_dis(lr_batch,hr_batch)
         self.train_step_gen(lr_batch, hr_batch)

    def save_display_examples(self, test_dataset, num_images = 60):
        count = 0
        for idx, (lr_batch, hr_batch) in enumerate(test_dataset):
            for i in range(len(lr_batch)):  # Assuming lr_batch and hr_batch have the same number of images
                lr_image = lr_batch[i]
                hr_image = hr_batch[i]
                display_handler.display_hr_lr(self.train_dir, self.generator, hr_image, lr_image, self.real_step,
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



