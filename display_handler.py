import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from pathlib import Path
from typing import List
import matplotlib.gridspec as gridspec
import os
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def dump_hr_lr_side_by_side(data_dir, generator, hr, lr, step, image_num):
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

def dump_hr_lr_images(data_dir, generator, hr, lr, step, image_num):
    # Define the paths for saving the images
    hr_path = f"{data_dir}/images/image_num_{image_num}_step_{step}_hr.png"
    lr_path = f"{data_dir}/images/image_num_{image_num}_step_{step}_lr.png"
    gen_path = f"{data_dir}/images/image_num_{image_num}_step_{step}_gen.png"

    # Calculate the size of hr_batch
    hr_size = hr.shape[0:2]  # Assuming hr_batch has shape (batch_size, height, width, channels)  TODO!!!!!!!!!!
    # Resize low-resolution images to the same size as high-resolution images
    lr = tf.image.resize(lr, hr_size, method=tf.image.ResizeMethod.BICUBIC)  #TODO!!!!!!!!!!
    # Save the images
    save_image(hr, hr_path)
    save_image(lr, lr_path)
    save_image(tf.squeeze(generator(tf.expand_dims(lr, 0)), axis=0), gen_path)

def show_image(title, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def show_image_diff(models,lr,hr,index):

    image1 = hr.numpy()
    image2 = models[0][1](np.expand_dims(lr, axis=0))[0].numpy()
    # convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two images
    (score, diff) = ssim(gray1, gray2, full=True, data_range=255)

    # the diff image contains the actual image differences between the two images
    # to make the differences visible, we scale the diff image to range [0,255]
    diff = (diff * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # compute the total difference in each contour and store it with the contour
    contour_diffs = [(cv2.sumElems(thresh[y:y+h, x:x+w])[0], contour) for contour in contours for x, y, w, h in [cv2.boundingRect(contour)]]

    # sort the contours by total difference, in descending order, and keep the top 1
    contours = [contour for _, contour in sorted(contour_diffs, key=lambda x: x[0], reverse=True)[:1]]

    # Create a directory to save the regions
    if not os.path.exists('regions'):
        os.makedirs('regions')
    cv2.imwrite(f'regions/{index}_hr.png', cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
    for model_name, model in models:
        cv2.imwrite(f'regions/{index}_{model_name}.png',cv2.cvtColor(model(np.expand_dims(lr, axis=0))[0].numpy(), cv2.COLOR_RGB2BGR))
    # loop over the contours
    for i, contour in enumerate(contours):
        # compute the bounding box of the contour and then draw the bounding box on both input images to represent where the two images differ
        (x, y, w, h) = cv2.boundingRect(contour)


        # Save the region of interest from image1
        hr_region = image1[y:y + h, x:x + w]
        cv2.imwrite(f'regions/{index}_hr_region_{i}.png', cv2.cvtColor(hr_region, cv2.COLOR_RGB2BGR))

        # Save the region of interest from image2
        for model_name, model in models:
            region = model(np.expand_dims(lr, axis=0))[0].numpy()[y:y + h, x:x + w]
            cv2.imwrite(f'regions/{index}_{model_name}_region_{i}.png', cv2.cvtColor(region, cv2.COLOR_RGB2BGR))

    for i, contour in enumerate(contours):
        # compute the bounding box of the contour and then draw the bounding box on both input images to represent where the two images differ
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Save the region of interest from image1

    cv2.imwrite(f'regions/{index}_hr_colored.png', cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))

    print("done")


def calculate_ssim_values(models, dataset):
    ssim_vals = {}
    # Loop over each model
    for model_name, model in models:
        ssim_vals[model_name] = []
        # Get the output of the model
        for lr, hr in dataset:
            fake_hr = model(lr, training=False)

            # Calculate the SSIM for each image in the batch
            for i in range(hr.shape[0]):  # iterate over the batch size
                ssim = tf.image.ssim(hr[i:i+1], fake_hr[i:i+1], max_val=1.0)
                ssim_vals[model_name].append(ssim[0])

    return ssim_vals


def get_max_difference_indices(ssim_vals, N):
    # Get the model names
    model_names = list(ssim_vals.keys())

    # Compute the absolute difference of SSIM values between the models for each image
    ssim_differences = np.abs(np.array(ssim_vals[model_names[0]]) - np.array(ssim_vals[model_names[1]]))

    # Find the indices of the N images that have the maximum differences in SSIM values
    max_diff_indices = ssim_differences.argsort()[-N:][::-1]

    return max_diff_indices.tolist()

def get_n_images_with_most_diffrancess(dataset,models,N):
    # Get the number of batches and batch size
    num_batches = 192469/16
    batch_size = 16

    # Calculate SSIM values
    ssim_vals = calculate_ssim_values(models, dataset)

    # Get indices of N images with most significant SSIM differences
    indices = get_max_difference_indices(ssim_vals, N)

    # For each index, calculate the batch number and the index within the batch
    for index in indices:
        batch_num = index // batch_size
        within_batch_index = index % batch_size


        hr = None
        lr = None
        for lr_batch,hr_batch in dataset.take(batch_num + 1):
            if batch_num == 0:
                lr = lr_batch[within_batch_index]
                hr = hr_batch[within_batch_index]
            else:
                batch_num -= 1


        # Show the difference between the high-resolution images of the two models
        show_image_diff(models,lr,hr,index)










  #tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)
  #tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)