import train
from sr_resnet import  SrResnet
from sr_gan import SrGan

import DataLoader
import model

#image_url = 'img_align_celeba'
demo_mode = False

image_url = 'small'
#demo_mode = True

datasets = DataLoader.MyDataLoader(image_url,demo_mode)


generator,epoch_gen = model.load_generator("sr_resnet/weights/generator")

sr_gan_trainer = SrResnet(generator,None, "sr_resnet",epoch_gen,demo_mode)
sr_gan_trainer.fit(datasets,5)

discriminator,epoch_des = model.load_discriminator("sr_resnet/weights/discriminator")
generator,epoch_gen = model.load_generator("sr_resnet/weights/generator")

sr_gan_trainer = SrGan(generator,discriminator, "sr_resnet",epoch_gen,demo_mode)
sr_gan_trainer.fit(datasets,500)