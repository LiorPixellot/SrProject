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


class AbsTrainer(ABC):
    def __init__(self,
                 generator: tf.keras.Model,
                 discriminator: tf.keras.Model,
                 train_dir: str,
                 start_epoch: int = -1,
                 demo_mode: bool = False,
                 optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.09)):

        self.real_step = 0
        self.train_dir = pathlib.Path(train_dir)
        self.optimizer = optimizer
        self.generator = generator
        self.discriminator = discriminator
        self.start_epoch = start_epoch + 1
        self._psnr_vals = []
        self._fid_vals = []
        self._ssim_vals = []
        self._real_epch = 0
        self._settings = self._get_settings_according_to_mode(demo_mode)
        self.inception_model = self._build_inception_model()
        self.creat_run_dirs()
        self.add_merics()

    def add_merics(self):
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
        demo_settings = {'steps_to_save_progress': 1}
        normal_mode = {'steps_to_save_progress': 100000}
        settings = demo_settings if demo_mode else normal_mode
        return settings


    def fit(self,datasets,epochs: int = 50):
        for epoch in tqdm(range(epochs)):
            self._real_epch = epoch + self.start_epoch
            for step_count_curr_epoch,(lr,hr) in enumerate(datasets.train_dataset):
                self.train_step(lr,hr)
                self.save_progress(datasets,step_count_curr_epoch)
                self.real_step +=1
            self.save_weights()

    def save_weights(self):
        gen_path = self.train_dir / "weights/generator"
        gen_path.mkdir(parents=True, exist_ok=True)
        self.generator.save(gen_path / f"{self._real_epch}.h5")

        if self.discriminator is not None:
            dis_path = self.train_dir / "weights/discriminator"
            dis_path.mkdir(parents=True, exist_ok=True)
            self.discriminator.save(dis_path / f"{self._real_epch}.h5")


    def save_progress(self,datasets,step: int):
        modulo = step  % self._settings['steps_to_save_progress']
        if (modulo == 0 and step > 0):
            self.eval(datasets.test_dataset,step)
            self.log_metrics(step)
            self.save_display_examples(datasets.display_dataset,step)

    def log_metrics(self, step):
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
            fake_hr = self.generator(lr)
            real_image_resized = preprocess_input_inception(tf.image.resize(hr, (299, 299)))
            generated_image_resized = preprocess_input_inception(tf.image.resize(fake_hr, (299, 299)))
            psnr_vals.append(tf.image.psnr(hr, fake_hr, max_val=255))
            ssim_vals.append(tf.image.ssim(hr, fake_hr, max_val=1.0))
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
        self._psnr_vals.append(np.mean(psnr_vals))
        self._ssim_vals.append(np.mean(ssim_vals))

    def calc_avg_psnr_ssim_curr_step(self,test_dataset):
        print("calc_avg_psnr_ssim_curr_step")
        psnr_vals = []
        ssim_vals = []
        for lr, hr in test_dataset:
            fake_hr = self.generator(lr)
            psnr_vals.append(tf.image.psnr(hr,fake_hr, max_val=255))
            ssim_vals.append(tf.image.ssim(hr, fake_hr, max_val=1.0))
        self._psnr_vals.append(np.mean(psnr_vals))
        self._ssim_vals.append(np.mean(ssim_vals))


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


    def eval(self,test_dataset, step: int):
        print(f"eval_epoch_{self._real_epch}_step_{step}")
        self._calculate_fid_ssim_psnr_cur_step(test_dataset)
        display_handler.plot_graph(self.psnr_plots_path, step, self._real_epch, self._psnr_vals)
        display_handler.plot_graph(self.fid_plots_path, step, self._real_epch, self._fid_vals)
        display_handler.plot_graph(self.ssim_plots_path, step, self._real_epch, self._ssim_vals)




    def _plot_psnr(self, step: int):
        plt.plot(self._psnr_vals)
        plt.savefig(self.train_dir / "psnr_plots" / f"psnr_epoch_{self._real_epch}_step_{step}.png")
        plt.close()

    def _plot_fid(self, step: int):
        plt.plot(self._fid_vals)
        plt.savefig(self.train_dir / "fid_plots" / f"fid_epoch_{self._real_epch}_step_{step}.png")
        plt.close()

    def train_step(self, lr_batch, hr_batch):
         self.train_step_dis(hr_batch, lr_batch)
         self.train_step_gen(lr_batch, hr_batch)

    def save_display_examples(self,test_dataset,step):
        for idx, (lr, hr) in enumerate(test_dataset):
            display_handler.display_hr_lr(self.train_dir,self.generator,hr,lr,self._real_epch,idx,step)

    @abstractmethod
    def train_step_dis(self, hr_batch, lr_batch):
        pass

    @abstractmethod
    def train_step_gen(self, lr_batch, hr_batch):
        pass



