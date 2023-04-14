import train

import DataLoader
import model

image_url = 'img_align_celeba'

#TODO LB
test_dataset_size = -1


data_loader = DataLoader.MyDataLoader(image_url)



generator,epoch_gen = model.load_generator("sr_resnet/weights/generator")
generator_trainer = train.GeneratorTrainer(generator, "sr_resnet", epoch_gen)
generator_trainer.fit(data_loader.train_dataset,data_loader.validation_dataset.take(test_dataset_size).cache(),5)

discriminator,epoch_des = model.load_discriminator("sr_resnet/weights/discriminator")
generator,epoch_gen = model.load_generator("sr_resnet/weights/generator")


sr_gan_trainer = train.SrGanTrainer(generator,discriminator, "sr_resnet",epoch_gen)

sr_gan_trainer.fit(data_loader.train_dataset,data_loader.validation_dataset.take(test_dataset_size).cache(),500)