import display_handler
from sr_resnet import SrResnet
from sr_gan import SrGan
from sr_wgan_gp import SrWganGp
from sr_wgan import SrWgan
import DataLoader
from srPix2Pix import SrPix2Pix
from pix2pix import pix2pix
from srWPix2PixHr import SrWPix2PixHr


demo_mode = True
hr_size=96





def display_demo(data_loader):
    srPix = SrPix2Pix("sr_Pix2Pix_01",hr_size,0.1)
    display_handler.save_display_examples("demo", srPix.generator, data_loader.validation_dataset)



def display_plots(data_loader):
    sr_Pix2Pix_01 = SrPix2Pix("sr_Pix2Pix_01", hr_size, 0.1)
    sr_wgan = SrWgan("sr_wgan_clip", hr_size)
    no_transfer_srgan = SrGan("no_transfer_srgan", hr_size)
    srgan = SrGan("srgan", hr_size)
    models = [('no_transfer_srgan', no_transfer_srgan.generator), ('sr_Pix2Pix_01', sr_Pix2Pix_01.generator),
              ('sr_wgan', sr_wgan.generator), ('srgan', srgan.generator)]
    display_handler.plot_pca_fid(models, data_loader.validation_dataset)



def display_most_improved(data_loader):
    sr_Pix2Pix_01 = SrPix2Pix("sr_Pix2Pix_01", hr_size, 0.1)
    sr_wgan = SrWgan("sr_wgan_clip", hr_size)
    srgan = SrGan("srgan", hr_size)
    models = [('sr_Pix2Pix_01', sr_Pix2Pix_01.generator), ('sr_wgan', sr_wgan.generator),
              ('srgan', srgan.generator)]
    display_handler.get_n_images_with_most_differences(data_loader, models, 20)
    display_handler.get_n_images_with_most_differences(data_loader, models, 20, "ssim")



def main(image_url):
    data_loader = DataLoader.MyDataLoader(image_url, hr_size, 4, demo_mode)
    display_demo(data_loader)
    display_plots(data_loader)
    display_most_improved(data_loader)

if __name__ == "__main__":
    image_url = 'demo/dataset' if demo_mode else "img_align_celeba"
    main(image_url)







