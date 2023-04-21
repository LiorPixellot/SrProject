import train

import DataLoader
import model

image_url = 'img_align_celeba'

demo_mode = True

datasets = DataLoader.MyDataLoader(image_url,demo_mode)


#generator,epoch_gen = model.load_generator("sr_resnet/weights/generator")
#generator_trainer = train.GeneratorTrainer(generator, "sr_resnet", epoch_gen)
#generator_trainer.fit(data_loader.train_dataset,test_dataset,display_dataset,sr_resnet_train_epoch)

discriminator,epoch_des = model.load_discriminator("sr_resnet/weights/discriminator")
generator,epoch_gen = model.load_generator("sr_resnet/weights/generator")

sr_gan_trainer = train.SrGanTrainer(generator,discriminator, "sr_resnet",epoch_gen,demo_mode)

sr_gan_trainer.fit(datasets,500)