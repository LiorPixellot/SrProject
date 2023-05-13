import display_handler
import train
from sr_resnet import SrResnet
from sr_gan import SrGan
from sr_wgan_gp import SrWganGp
import DataLoader
import model
from pathlib import Path

# Define demo_mode as a global variable
demo_mode = False

def main(image_url):
    global demo_mode
    # Load the dataset using a custom DataLoader
    data_loader = DataLoader.MyDataLoader(image_url, demo_mode)
   # display_handler.display_images(data_loader.train_dataset)
    train_sr_resnet(data_loader)

    train_sr_gan(data_loader)
    train_sr_wgan_gp(data_loader)

def train_sr_resnet(data_loader):
    global demo_mode
    sr_resnet_weights = Path("sr_resnet/weights")
    generator, step_gen = model.load_generator(sr_resnet_weights / "generator")
    sr_resnet_trainer = SrResnet(generator, None, "sr_resnet", step_gen, demo_mode)
    sr_resnet_trainer.fit(data_loader, 11000000)


def train_sr_wgan_gp(data_loader):
    global demo_mode
    sr_wgan_gp_weights = Path("sr_wgan_gp/weights")
    discriminator, step_des = model.load_discriminator(sr_wgan_gp_weights / "discriminator")
    generator, step_gen = model.load_generator(sr_wgan_gp_weights / "generator")
    sr_wgan_gp_trainer = SrWganGp(generator, discriminator, "sr_wgan_gp", step_gen, demo_mode)
    sr_wgan_gp_trainer.fit(data_loader, 1000000)


def train_sr_gan(data_loader):
    global demo_mode
    sr_gan_weights = Path("sr_gan/weights")
    discriminator, step_des = model.load_discriminator(sr_gan_weights / "discriminator")
    generator, step_gen = model.load_generator(sr_gan_weights / "generator")
    sr_wgan_gp_trainer = SrGan(generator, discriminator, "sr_gan", step_gen, demo_mode)
    sr_wgan_gp_trainer.fit(data_loader, 1000000)


if __name__ == "__main__":
    image_url = 'small' if demo_mode else "img_align_celeba"
    main(image_url)
