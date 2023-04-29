import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pathlib import Path
from typing import List


def display_images(dataset, num_images=5):
    # Create iterators for the HR and LR datasets

    fig, axs = plt.subplots(num_images, 2, figsize=(12, 12))
    for i in range(num_images):
        lr, hr = next(iter(dataset))

        print("HR image shape:", hr[0].shape)
        print("HR image min-max:", np.min(hr[0]), np.max(hr[0]))

        print("LR image shape:", lr[0].shape)
        print("LR image min-max:", np.min(lr[0]), np.max(lr[0]))

        axs[i][0].imshow(tf.cast(hr[0], tf.uint8))
        axs[i][0].set_title('HR Image')
        axs[i][1].imshow(tf.cast(lr[0], tf.uint8))
        axs[i][1].set_title('LR Image')
        axs[i][0].axis('off')
        axs[i][1].axis('off')
        print(hr[0])

    plt.show()

def display_hr_lr(data_dir,generator,hr,lr,step,image_num):
    fig, axs = plt.subplots(1, 3, figsize=(9, 9))
    axs[0].imshow(tf.cast(hr, tf.uint8))
    axs[0].set_title('HR Image')
    axs[1].imshow(tf.cast(lr, tf.uint8))
    axs[1].set_title('LR Image')
    axs[2].imshow(tf.cast(tf.squeeze(generator( tf.expand_dims(lr, 0)),axis=0), tf.uint8))
    axs[2].set_title('gen Image')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    template = '{}/images/image_num_{}_step_{}.png'
    plt.savefig(template.format(data_dir,image_num, step))
    plt.close()

def show_progress(dataset,generator,step,feature_Loss,num_images=5):
    # Create iterators for the HR and LR datasets

    pair_images = next(iter(dataset))

    # Plot the first num_images HR and LR images side by side
    fig, axs = plt.subplots(num_images, 3, figsize=(12, 12))
    for i in range(num_images):
        pair_images = next(iter(dataset))
        axs[i][0].imshow(tf.cast(pair_images['lr_image'][i], tf.uint8))
        axs[i][0].set_title('HR Image')
        axs[i][1].imshow(tf.cast(pair_images['hr_image'][i], tf.uint8))
        axs[i][1].set_title('LR Image')
        axs[i][2].imshow(tf.cast(pair_images['hr_image'][i], tf.uint8))
        axs[i][2].set_title('generated')
        tf.squeeze(generator(np.expand_dims(pair_images['lr_image'][i], axis=0)), axis=0)
        axs[i][0].axis('off')
        axs[i][1].axis('off')
        axs[i][2].axis('off')

    template = 'data/images/step_{}_feature_Loss_{}.png'
    plt.savefig(template.format( step , feature_Loss))

def plot_graph(path: Path, step: int, vals: List[float]) -> None:
    plt.plot(vals)
    plt.savefig( path/ f"step_{step}.png")
    plt.close()
