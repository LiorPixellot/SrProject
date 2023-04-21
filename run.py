import train

import DataLoader
import model

image_url = 'img_align_celeba'

#TODO LB
display_dataset_size = 30
test_dataset_size = -1
sr_resnet_train_epoch = 5

data_loader = DataLoader.MyDataLoader(image_url)
display_dataset = data_loader.validation_dataset.take(display_dataset_size).cache()
test_dataset = data_loader.validation_dataset.take(test_dataset_size).cache()

#generator,epoch_gen = model.load_generator("sr_resnet/weights/generator")
#generator_trainer = train.GeneratorTrainer(generator, "sr_resnet", epoch_gen)
#generator_trainer.fit(data_loader.train_dataset,test_dataset,display_dataset,sr_resnet_train_epoch)

discriminator,epoch_des = model.load_discriminator("sr_resnet/weights/discriminator")
generator,epoch_gen = model.load_generator("sr_resnet/weights/generator")


sr_gan_trainer = train.SrGanTrainer(generator,discriminator, "sr_resnet",epoch_gen)

sr_gan_trainer.fit(data_loader.train_dataset,test_dataset,display_dataset,500)