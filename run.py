import display_handler
import train
from sr_resnet import SrResnet
from sr_gan import SrGan
from sr_wgan_gp import SrWganGp
from sr_wgan import SrWgan
import DataLoader
import model
from pathlib import Path

# Define deo_mode as a global variable
demo_mode = False

def main(image_url):
    global demo_mode
    # Load the dataset using a custom DataLoader
    data_loader = DataLoader.MyDataLoader(image_url, demo_mode)
   #display_handler.display_images(data_loader.train_dataset)
    #train_sr_resnet(data_loader)

    #train_sr_gan(data_loader)
    #train_sr_wgan(data_loader)
    #train_sr_wgan_gp(data_loader)
    #train_sr_wgan_gp_2(data_loader)
    #train_sr_wgan_gp_05(data_loader)
    train_sr_cgan(data_loader)

def train_sr_resnet(data_loader):
    global demo_mode
    sr_resnet_weights = Path("sr_resnet/weights")
    generator, step_gen = model.load_generator(sr_resnet_weights / "generator")
    sr_resnet_trainer = SrResnet(generator, None, "sr_resnet", step_gen, demo_mode)
    sr_resnet_trainer.save_display_examples(data_loader.validation_dataset)
    #sr_resnet_trainer.fit(data_loader, 1200005)

def train_sr_wgan(data_loader):
    global demo_mode
    sr_wgan_gp_weights = Path("sr_wgan_clip/weights")
    discriminator, step_des = model.load_discriminator(sr_wgan_gp_weights / "discriminator")
    generator, step_gen = model.load_generator(sr_wgan_gp_weights / "generator")
    sr_wgan_gp_trainer = SrWgan(generator, discriminator, "sr_wgan_clip", step_gen, demo_mode)
    sr_wgan_gp_trainer.fit(data_loader, 3000005)

def train_sr_wgan_gp(data_loader):
    global demo_mode
    sr_wgan_gp_weights = Path("sr_wgan_gp/weights")
    discriminator, step_des = model.load_discriminator(sr_wgan_gp_weights / "discriminator")
    generator, step_gen = model.load_generator(sr_wgan_gp_weights / "generator")
    sr_wgan_gp_trainer = SrWganGp(generator, discriminator, "sr_wgan_gp", step_gen, demo_mode)
    sr_wgan_gp_trainer.fit(data_loader, 3000005)

def train_sr_wgan_gp_2(data_loader):
    global demo_mode
    sr_wgan_gp_weights = Path("sr_wgan_gp_2/weights")
    discriminator, step_des = model.load_discriminator(sr_wgan_gp_weights / "discriminator")
    generator, step_gen = model.load_generator(sr_wgan_gp_weights / "generator")
    sr_wgan_gp_trainer = SrWganGp(generator, discriminator, "sr_wgan_gp_2", step_gen, demo_mode,2)
    sr_wgan_gp_trainer.fit(data_loader, 3000005)

def train_sr_wgan_gp_05(data_loader):
    global demo_mode
    sr_wgan_gp_weights = Path("sr_wgan_gp_05/weights")
    discriminator, step_des = model.load_discriminator(sr_wgan_gp_weights / "discriminator")
    generator, step_gen = model.load_generator(sr_wgan_gp_weights / "generator")
    sr_wgan_gp_trainer = SrWganGp(generator, discriminator, "sr_wgan_gp_05", step_gen, demo_mode,0.5)
    sr_wgan_gp_trainer.fit(data_loader, 3000005)

def train_sr_gan(data_loader):
    global demo_mode
    sr_gan_weights = Path("sr_gan/weights")
    discriminator, step_des = model.load_discriminator(sr_gan_weights / "discriminator")
    generator, step_gen = model.load_generator(sr_gan_weights / "generator")
    sr_wgan_gp_trainer = SrGan(generator, discriminator, "sr_gan", step_gen, demo_mode)
    sr_wgan_gp_trainer.fit(data_loader, 1800005)

def train_sr_cgan(data_loader):
    global demo_mode
    sr_gan_weights = Path("sr_cgan_hr/weights")
    discriminator, step_des = model.load_discriminator(sr_gan_weights / "discriminator")
    generator, step_gen = model.load_generator(sr_gan_weights / "generator")
    sr_wgan_gp_trainer = SrGan(generator, discriminator, "sr_cgan_hr", step_gen, demo_mode)
    sr_wgan_gp_trainer.fit(data_loader, 1800005)


if __name__ == "__main__":
    image_url = 'small' if demo_mode else "img_align_celeba"
    main(image_url)
