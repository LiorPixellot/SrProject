import display_handler
import train
from sr_resnet import SrResnet
from sr_gan import SrGan
from sr_wgan_gp import SrWganGp
from sr_wgan import SrWgan
import DataLoader
import model
from pathlib import Path
from sr_cgan import  SrCGan
from srPix2Pix import SrPix2Pix
from pix2pix import pix2pix
# Define deo_mode as a global variable
demo_mode = False

def main(image_url):
    global demo_mode
    # Load the dataset using a custom DataLoader
    data_loader = DataLoader.MyDataLoader(image_url, demo_mode)

    sr_Pix2Pix_100 = SrPix2Pix("sr_Pix2Pix_01", 0.1)
    sr_Pix2Pix_100.fit(data_loader,3000000)

    sr_Pix2Pix_100 = pix2pix("real_Pix2Pix_100", 100)
    sr_Pix2Pix_100.fit(data_loader,3000000)

   # sr_Pix2Pix_1 = SrPix2Pix("sr_Pix2Pix_1",1)
   # sr_Pix2Pix_01 = SrPix2Pix("sr_Pix2Pix_01", 0.1)
   # sr_Pix2Pix_10 = SrPix2Pix("sr_Pix2Pix_10", 10)

    #models = [('sr_Pix2Pix_1', sr_Pix2Pix_1.generator), ('sr_Pix2Pix_01', sr_Pix2Pix_01.generator), ('sr_Pix2Pix_10', sr_Pix2Pix_10.generator)]
    #display_handler.get_n_images_with_most_diffrancess_fid(data_loader.validation_dataset,models,20)




if __name__ == "__main__":
    image_url = 'small' if demo_mode else "img_align_celeba"
    main(image_url)
