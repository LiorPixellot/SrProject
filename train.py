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

        self.train_dir = pathlib.Path(train_dir)
        self.optimizer = optimizer
        self.generator = generator
        self.discriminator = discriminator
        self.start_epoch = start_epoch + 1
        self._psnr_vals = []
        self._fid_vals = []
        self._real_epch = 0
        self._settings = self._get_settings_according_to_mode(demo_mode)
        self.inception_model = self._build_inception_model()
        self.creat_run_dirs()

    def creat_run_dirs(self):
        self.psnr_plots_path = self.train_dir / "psnr_plots"
        self.fid_plots_path = self.train_dir / "fid_plots"
        self.images_path = self.train_dir / "images"

        self.psnr_plots_path.mkdir(parents=True, exist_ok=True)
        self.fid_plots_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)


    def _build_inception_model(self):
        model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        model.trainable = False
        return model


    def _get_settings_according_to_mode(self, demo_mode: bool) -> dict:
        demo_settings = {'steps_to_save_progress': 1}
        normal_mode = {'steps_to_save_progress': 100}
        settings = demo_settings if demo_mode else normal_mode
        return settings


    def fit(self,datasets,epochs: int = 50):
        for epoch in tqdm(range(epochs)):
            self._real_epch = epoch + self.start_epoch
            for step_count,(lr,hr) in tqdm(enumerate(datasets.train_dataset)):
                self.train_step(lr,hr)
                self.save_progress(datasets,step_count)
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
            self.save_display_examples(datasets.display_dataset,step)



    def calc_avg_psnr_curr_step(self,test_dataset):
        print("psnr calc")
        psnr_vals = []
        for lr, hr in test_dataset:
            psnr_vals.append(tf.image.psnr(self.generator(lr),hr, max_val=255))
        return np.mean(psnr_vals)

    def _calculate_fid_cur_step(self,test_dataset):
        print("fid calc")

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
        return fid


    def eval(self,test_dataset, step: int):
        self._fid_vals.append(self._calculate_fid_cur_step(test_dataset))
        display_handler.plot_graph(self.fid_plots_path, step, self._real_epch, self._fid_vals)
        self._psnr_vals.append(self.calc_avg_psnr_curr_step(test_dataset))
        display_handler.plot_graph(self.psnr_plots_path,step,self._real_epch,self._psnr_vals)




    def _plot_psnr(self, step: int):
        plt.plot(self._psnr_vals)
        plt.savefig(self.train_dir / "psnr_plots" / f"psnr_epoch_{self._real_epch}_step_{step}.png")
        plt.close()

    def _plot_fid(self, step: int):
        plt.plot(self._fid_vals)
        plt.savefig(self.train_dir / "fid_plots" / f"fid_epoch_{self._real_epch}_step_{step}.png")
        plt.close()

    @abstractmethod
    def train_step(self, lr, hr):
        pass

    def save_display_examples(self,test_dataset,step):
        for idx, (lr, hr) in enumerate(test_dataset):
            display_handler.display_hr_lr(self.train_dir,self.generator,hr,lr,self._real_epch,idx,step)



