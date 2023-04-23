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

    train_sr_resnet(data_loader)
    train_sr_wgan_gp(data_loader)
    train_sr_gan(data_loader)


def train_sr_resnet(data_loader):
    global demo_mode
    sr_resnet_weights = Path("sr_resnet/weights")
    generator, epoch_gen = model.load_generator(sr_resnet_weights / "generator")
    sr_resnet_trainer = SrResnet(generator, None, "sr_resnet", epoch_gen, demo_mode)
    sr_resnet_trainer.fit(data_loader, 5)


def train_sr_wgan_gp(data_loader):
    global demo_mode
    sr_wgan_gp_weights = Path("sr_wgan_gp/weights")
    discriminator, epoch_des = model.load_discriminator(sr_wgan_gp_weights / "discriminator")
    generator, epoch_gen = model.load_generator(sr_wgan_gp_weights / "generator")
    sr_wgan_gp_trainer = SrWganGp(generator, discriminator, "sr_wgan_gp", epoch_gen, demo_mode)
    sr_wgan_gp_trainer.fit(data_loader, 5)


def train_sr_gan(data_loader):
    global demo_mode
    sr_gan_weights = Path("sr_gan/weights")
    discriminator, epoch_des = model.load_discriminator(sr_gan_weights / "discriminator")
    generator, epoch_gen = model.load_generator(sr_gan_weights / "generator")
    sr_wgan_gp_trainer = SrGan(generator, discriminator, "sr_gan", epoch_gen, demo_mode)
    sr_wgan_gp_trainer.fit(data_loader, 5)


if __name__ == "__main__":
    image_url = 'img_align_celeba'
    main(image_url)
