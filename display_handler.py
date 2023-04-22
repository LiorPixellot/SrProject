import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pathlib import Path
from typing import List



def display_images(dataset, num_images=5):
    # Create iterators for the HR and LR datasets

    pair_images = next(iter(dataset))

    # Plot the first num_images HR and LR images side by side
    fig, axs = plt.subplots(num_images, 2, figsize=(12, 12))
    for i in range(num_images):
        pair_images = next(iter(dataset))
        axs[i][0].imshow(tf.cast(pair_images['lr_image'][i], tf.uint8))
        axs[i][0].set_title('HR Image')
        axs[i][1].imshow(tf.cast(pair_images['hr_image'][i], tf.uint8))
        axs[i][1].set_title('LR Image')
        axs[i][0].axis('off')
        axs[i][1].axis('off')

    plt.show()


def display_hr_lr(data_dir,generator,hr,lr,epoch,image_num,step):
    fig, axs = plt.subplots(1, 3, figsize=(9, 9))
    axs[0].imshow(tf.cast(tf.squeeze(hr,axis=0), tf.uint8))
    axs[0].set_title('HR Image')
    axs[1].imshow(tf.cast(tf.squeeze(lr,axis=0), tf.uint8))
    axs[1].set_title('LR Image')
    axs[2].imshow(tf.cast(tf.squeeze(generator(lr),axis=0), tf.uint8))
    axs[2].set_title('gen Image')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    template = '{}/images/image_num_{}_Epoch_{}_step_{}.png'
    plt.savefig(template.format(data_dir,image_num, epoch,step))
    plt.close()

def show_progress(dataset,generator,epoch,feature_Loss,num_images=5):
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

    template = 'data/images/Epoch_{}_feature_Loss_{}.png'
    plt.savefig(template.format(epoch, feature_Loss))

def plot_graph(path: Path, step: int, epoch: int, vals: List[float]) -> None:
    plt.plot(vals)
    plt.savefig( path/ f"epoch_{epoch}_step_{step}.png")
    plt.close()
