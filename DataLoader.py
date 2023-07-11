import tensorflow as tf
import pathlib
import os


class MyDataLoader:
    def __init__(self, image_dir,demo_mode = False,
                  hr_size=96,downsample_factor=4,val_split=0.05, batch_size=1):
        self.image_dir = image_dir
        self.hr_size = hr_size
        self.downsample_factor = downsample_factor
        self.lr_size = self.hr_size // self.downsample_factor
        self.batch_size = batch_size
        self.val_split = val_split

        # self._download_dataset()
        self._load_datasets()

    def _load_datasets(self):
        # Define the training and validation datasets without batching
        train_dataset = tf.keras.utils.image_dataset_from_directory(
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

        val_dataset = tf.keras.utils.image_dataset_from_directory(
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
        self.total_images = len([f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))])
        self.total_val_images = int(self.total_images * (self.val_split))
        self.total_train_images = self.total_images - self.total_val_images
        print("images amount {0} {1} {2}", self.total_images, self.total_train_images,self.total_val_images)
        def generate_image_pairs(hr_image):
            # Resize the image to the HR size
            hr_image= tf.clip_by_value(hr_image, 00, 255.0)
            # Downsample the image to the LR size using bicubic interpolation
            lr_image = tf.image.resize(hr_image, [self.lr_size, self.lr_size], method='bicubic')
            lr_image = tf.clip_by_value(lr_image, 00, 255.0)
            # Create a dictionary of LR and HR images
            image_pair = (lr_image, hr_image)

            return image_pair

        def filter_incomplete_batches(*args):
            return tf.shape(args[0])[0] == self.batch_size

        # Create LR and HR datasets for training and validation
        self.train_dataset = train_dataset.filter(filter_incomplete_batches).map(generate_image_pairs)
        self.validation_dataset = val_dataset.filter(filter_incomplete_batches).map(generate_image_pairs)
