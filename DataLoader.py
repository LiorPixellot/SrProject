import  tensorflow as tf
import pathlib
import os
class MyDataLoader:
    def __init__(self, image_dir, hr_size=96, downsample_factor=4, batch_size=1, val_split=0.1):
        self.image_dir = image_dir
        self.hr_size = hr_size
        self.downsample_factor = downsample_factor
        self.lr_size = self.hr_size // self.downsample_factor
        self.batch_size = batch_size
        self.val_split = val_split

        self._download_dataset()
        self._load_datasets()

    def _download_dataset(self):
        # Download and extract the dataset
        archive = tf.keras.utils.get_file(
            os.path.basename(self.image_dir),
            self.image_dir,
            extract=True
        )

        self.image_dir = pathlib.Path(archive).with_suffix('')

    def _load_datasets(self):
        # Define the training and validation datasets
        train_dataset = tf.keras.utils.image_dataset_from_directory(
            self.image_dir,
            labels='inferred',
            label_mode=None,
            image_size=(self.hr_size, self.hr_size),
            batch_size=self.batch_size,
            shuffle=True,
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
            shuffle=True,
            seed=42,
            validation_split=self.val_split,
            subset='validation',
            interpolation='bicubic'
        )


        def generate_image_pairs(hr_image):
            # Resize the image to the HR size
            hr_image = tf.image.resize(hr_image, [self.hr_size, self.hr_size])

            # Downsample the image to the LR size using bicubic interpolation
            lr_image = tf.image.resize(hr_image, [self.lr_size, self.lr_size], method='bicubic')

            # Create a dictionary of LR and HR images
            image_pair = (lr_image, hr_image)

            return image_pair


        # Create LR and HR datasets for training and validation

        self.train_dataset = train_dataset.map(generate_image_pairs).cache().prefetch(tf.data.AUTOTUNE)

        self.validation_dataset = val_dataset.map(generate_image_pairs).cache().prefetch(tf.data.AUTOTUNE)


