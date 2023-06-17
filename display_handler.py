import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from pathlib import Path
from typing import List
import matplotlib.gridspec as gridspec
import os

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


def display_hr_lr(data_dir, generator, hr, lr, step, image_num):
    fig = plt.figure(figsize=(15, 5))

    gs = gridspec.GridSpec(1, 3, width_ratios=[4, 1, 4])  # Define the width ratio of the subplots

    ax0 = plt.subplot(gs[0])
    ax0.imshow(tf.cast(hr, tf.uint8))
    ax0.set_title('HR Image')
    ax0.axis('off')

    ax1 = plt.subplot(gs[1])
    ax1.imshow(tf.cast(lr, tf.uint8))
    ax1.set_title('LR Image')
    ax1.axis('off')

    ax2 = plt.subplot(gs[2])
    ax2.imshow(tf.cast(tf.squeeze(generator(tf.expand_dims(lr, 0)), axis=0), tf.uint8))
    ax2.set_title('gen Image')
    ax2.axis('off')

    # Adjust the space between subplots
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    template = '{}/images/image_num_{}_step_{}.png'
    plt.savefig(template.format(data_dir, image_num, step))
    plt.close()

def save_image(img, path):
    """Save a tensor image to a file."""
    img = tf.cast(img, tf.uint8)
    img = Image.fromarray(img.numpy())
    img.save(path)

def display_hr_lr_2(data_dir, generator, hr, lr, step, image_num):
    # Define the paths for saving the images
    hr_path = f"{data_dir}/images/image_num_{image_num}_step_{step}_hr.png"
    lr_path = f"{data_dir}/images/image_num_{image_num}_step_{step}_lr.png"
    gen_path = f"{data_dir}/images/image_num_{image_num}_step_{step}_gen.png"

    # Save the images
    #save_image(hr, hr_path)
   #save_image(lr, lr_path)
    save_image(tf.squeeze(generator(tf.expand_dims(lr, 0)), axis=0), gen_path)



import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def show_image(title, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def show_image_diff():

    # load the two input images
    image1 = cv2.imread('0.png')
    image2 = cv2.imread('1.png')

    # convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two images
    (score, diff) = ssim(gray1, gray2, full=True)

    # the diff image contains the actual image differences between the two images
    # to make the differences visible, we scale the diff image to range [0,255]
    diff = (diff * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # compute the total difference in each contour and store it with the contour
    contour_diffs = [(cv2.sumElems(thresh[y:y+h, x:x+w])[0], contour) for contour in contours for x, y, w, h in [cv2.boundingRect(contour)]]

    # sort the contours by total difference, in descending order, and keep the top 3
    contours = [contour for _, contour in sorted(contour_diffs, key=lambda x: x[0], reverse=True)[:3]]

    # Create a directory to save the regions
    if not os.path.exists('regions'):
        os.makedirs('regions')

    # loop over the contours
    for i, contour in enumerate(contours):
        # compute the bounding box of the contour and then draw the bounding box on both input images to represent where the two images differ
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save the region of interest
        region = image2[y:y+h, x:x+w]
        cv2.imwrite(f'regions/2_region_{i}.png', region)
        # Save the region of interest
        region = image1[y:y + h, x:x + w]
        cv2.imwrite(f'regions/1_region_{i}.png', region)

    # show the output images
    show_image("Original", image1)
    show_image("Modified", image2)
    show_image("Diff", diff)
    show_image("Thresh", thresh)

    print("done")

