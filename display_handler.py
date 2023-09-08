from PIL import Image
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from DataLoader import MyDataLoader
import  model
import itertools

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
    save_image(hr, hr_path)
    save_image(lr, lr_path)
    save_image(tf.squeeze(generator(tf.expand_dims(lr, 0)), axis=0), gen_path)

def show_image(title, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()

def show_image_diff(models,lr,hr,index,metric):
    for model_name, model in models:
        cv2.imwrite(f'demo/{metric}/{index}_{model_name}.png',cv2.cvtColor(model(np.expand_dims(lr, axis=0))[0].numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'demo/{metric}/{index}_hr_colored.png', cv2.cvtColor(hr.numpy(), cv2.COLOR_RGB2BGR))


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





def save_display_examples(train_dir, generator, test_dataset, real_step =0, num_images = 30):
    count = 0
    for idx, (lr_batch, hr_batch) in enumerate(test_dataset):
        for i in range(len(lr_batch)):
            lr_image = lr_batch[i]
            hr_image = hr_batch[i]
            dump_hr_lr_images(train_dir, generator, hr_image, lr_image, real_step,
                                                  idx * len(lr_batch) + i)
            count += 1
            if count >= num_images:
               return

def calculate_fid_values(models, dataset):
    import model
    inception_model = model.build_inception_model()
    vals = {}

    # Loop over each model
    for model_name, model in models:

        vals[model_name] = []

        # Get the output of the model
        for lr_batch, hr_batch in dataset:

            fake_hr_batch = model(lr_batch, training=False)
            real_image_resized_batch = preprocess_input_inception(tf.image.resize(hr_batch, (299, 299)))
            generated_image_resized_batch = preprocess_input_inception(tf.image.resize(fake_hr_batch, (299, 299)))

            a_batch = np.squeeze(inception_model(real_image_resized_batch))
            b_batch = np.squeeze(inception_model(generated_image_resized_batch))

            # Calculate the MSE for each image in the batch
            for i in range(hr_batch.shape[0]):  # iterate over the batch size
                a = a_batch[i]
                b = b_batch[i]
                mse = np.sqrt(np.mean((a - b)**2))
                vals[model_name].append(mse)

    return vals


def get_max_difference_indices(diff_vals, N):
    # Get the model names
    model_names = list(diff_vals.keys())
    # Compute the difference of values between the models for each image
    ssim_differences = np.array(diff_vals[model_names[0]]) - np.array(diff_vals[model_names[1]])
    # Find the indices of the N images that have the maximum differences in diff values
    max_diff_indices = ssim_differences.argsort()[-N:][::-1]
    return max_diff_indices.tolist()


def get_n_images_with_most_differences(data_loader : MyDataLoader, models, N : int, metric = "fid"):
    dataset = data_loader.validation_dataset

    # Calculate SSIM values
    if metric == "fid":
        diff_vals = calculate_fid_values(models, dataset)
    else:
        diff_vals = calculate_ssim_values(models, dataset)
    # Get indices of N images with most significant SSIM differences
    indices = get_max_difference_indices(diff_vals, N)

    dataset = data_loader.validation_dataset.unbatch()  # Flatten the dataset
    # For each index, fetch the image at the corresponding position
    for index in indices:
        lr, hr = next(itertools.islice(dataset, index, index + 1))

        # Show the difference between the high-resolution images of the two models
        show_image_diff(models, lr, hr, index, metric)










def plot_pca_fid(models, dataset):
    inception_model = model.build_inception_model()
    fid_vals = {}
    scaler = StandardScaler()
    pca = PCA(n_components=2)

    # Loop over the HR images
    fid_vals["hr"] = []
    for lr, hr in dataset:
        hr = hr / 255.0
        real_image_resized = preprocess_input_inception(tf.image.resize(hr, (299, 299)))
        fid_vals["hr"].append(inception_model(real_image_resized))

    # Compute PCA for the real high-resolution images and store results
    fid_vals["hr"] =  np.vstack(fid_vals["hr"])
    fid_vals["hr"] = scaler.fit_transform(fid_vals["hr"])  # Add this line
    pca_hr = pca.fit_transform(fid_vals["hr"])

    plt.figure(figsize=(10, 10))
    plt.hist2d(pca_hr[:, 0], pca_hr[:, 1], bins=(50, 50), cmap= plt.cm.jet_r, range=[[-50, 50], [-50, 50]],vmin=0, vmax=300, cmin=1)
    plt.colorbar()
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig("demo/plots/hr_histogram.png")
    plt.close()

    # Loop over each model
    for model_name, model_instance in models:
        fid_vals = []
        # Get the output of the model
        for lr, hr in dataset:
            fake_hr = model_instance(lr, training=False)
            fake_hr = fake_hr / 255.0
            generated_image_resized = preprocess_input_inception(tf.image.resize(fake_hr, (299, 299)))
            fid_vals.append(inception_model(generated_image_resized))

        # Compute PCA for each model and store results
        fid_vals = np.vstack(fid_vals)
        fid_vals = scaler.transform(fid_vals)  # Add this line
        pca_vals = pca.transform(fid_vals)

        # Plotting
        plt.figure(figsize=(10, 10))
        plt.hist2d(pca_vals[:, 0], pca_vals[:, 1], bins=(50, 50), cmap=plt.cm.jet_r, range=[[-50, 50], [-50, 50]],vmin=0, vmax=300, cmin=1)
        plt.colorbar()
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.savefig(f"demo/plots/{model_name}_histogram.png")
        plt.close()



  #tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)
  #tf.keras.utils.plot_model(self.discriminator, show_shapes=True, dpi=64)