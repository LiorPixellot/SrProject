import tensorflow as tf
import pathlib
import os


class MyDataLoader:
    def __init__(self, image_dir,hr_size=96,downsample_factor=4,demo_mode = False, batch_size=1):
        self.image_dir = image_dir
        self.hr_size = hr_size
        self.downsample_factor = downsample_factor
        self.lr_size = self.hr_size // self.downsample_factor
        self.batch_size = batch_size
        self.val_split = 1.0 if demo_mode else 0.05

        self._load_datasets()

    def _load_datasets(self):
        self.total_images = len(
            [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))])

        if self.val_split == 1:
            # All images to validation dataset
            self.total_val_images = self.total_images
            self.total_train_images = 0

            val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.image_dir,
                labels='inferred',
                label_mode=None,
                image_size=(self.hr_size, self.hr_size),
                batch_size=self.batch_size,
                shuffle=False,
                seed=42,
                interpolation='bicubic'
            )

            train_dataset = None  # As no images for training
        else:
            # Split images into training and validation datasets
            train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.image_dir,
                labels='inferred',
                label_mode=None,
                image_size=(self.hr_size, self.hr_size),
                batch_size=self.batch_size,
                shuffle=False,
                seed=42,
                validation_split=self.val_split,
                subset='training',
                interpolation='bicubic'
            )

            val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                self.image_dir,
                labels='inferred',
                label_mode=None,
                image_size=(self.hr_size, self.hr_size),
                batch_size=self.batch_size,
                shuffle=False,
                seed=42,
                validation_split=self.val_split,
                subset='validation',
                interpolation='bicubic'
            )

            self.total_val_images = int(self.total_images * (self.val_split))
            self.total_train_images = self.total_images - self.total_val_images

        print("images amount total {0} train {1} val {2}".format(self.total_images, self.total_train_images,
                                                                 self.total_val_images))

        def generate_image_pairs(hr_image):
            hr_image = tf.clip_by_value(hr_image, 0.0, 255.0)
            lr_image = tf.image.resize(hr_image, [self.lr_size, self.lr_size], method='bicubic')
            lr_image = tf.clip_by_value(lr_image, 0.0, 255.0)
            image_pair = (lr_image, hr_image)

            return image_pair

        def filter_incomplete_batches(*args):
            return tf.shape(args[0])[0] == self.batch_size

        if train_dataset is not None:
            self.train_dataset = train_dataset.filter(filter_incomplete_batches).map(generate_image_pairs)
        else:
            self.train_dataset = None

        self.validation_dataset = val_dataset.filter(filter_incomplete_batches).map(generate_image_pairs)
