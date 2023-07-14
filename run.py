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
from cameraDemo import cameraDemo
from srWPix2PixHr import SrWPix2PixHr
# Define deo_mode as a global variable
demo_mode = False
hr_size=96
def main(image_url):
    global demo_mode
    # Load the dataset using a custom DataLoader
    data_loader = DataLoader.MyDataLoader(image_url, demo_mode, hr_size)

    sr_Pix2Pix_01 = SrPix2Pix("sr_Pix2Pix_01", hr_size, 0.1)
    sr_wgan = SrWgan("sr_wgan_clip", hr_size)
    no_transfer_srgan = SrGan("no_transfer_srgan", hr_size)
    srgan = SrGan("srgan", hr_size)
    models = [('no_transfer_srgan', no_transfer_srgan.generator), ('sr_Pix2Pix_01', sr_Pix2Pix_01.generator),
              ('sr_wgan', sr_wgan.generator), ('srgan', srgan.generator)]
    display_handler.plot_pca_fid(models, data_loader.validation_dataset)
    models = [('sr_Pix2Pix_01', sr_Pix2Pix_01.generator), ('sr_wgan', sr_wgan.generator),
              ('srgan', srgan.generator)]
    display_handler.get_n_images_with_most_diffrancess(data_loader, models, 20)
    display_handler.get_n_images_with_most_diffrancess(data_loader, models, 20, "ssim")
    # resnet_upsample = SrResnet("upsample_conv",hr_size)
    # resnet_upsample.fit(data_loader,1000000)

    # data_loader = DataLoader.MyDataLoader(image_url, demo_mode, hr_size,1)
    pix2pix_dis = SrPix2Pix("save_work/no_transfer_pix2pix_01/_pix2pix01", hr_size, 0.1)
    # pix2pix_dis.fit(data_loader, 1200000)
    # git addcameraDemo(pix2pix_dis.generator)

if __name__ == "__main__":
    image_url = 'small' if demo_mode else "img_align_celeba"
    main(image_url)













