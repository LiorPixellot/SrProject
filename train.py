import  DataLoader
import tensorflow as tf
from tqdm import tqdm
import display_handler
from model import sr_resnet
from display_handler import display_images
# Define the URL of the dataset to download
image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'

# Create a MyDataLoader object with the desired parameters
data_loader = DataLoader.MyDataLoader(image_url)


class GeneratorTrainer:
    def __init__(self,train_dir,start_epoc = -1,optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4,beta_1=0.09),loss = tf.keras.losses.MeanSquaredError()):
        self.train_dir = train_dir
        self.loss = loss
        self.optimizer = optimizer
        if(start_epoc < 0):
            self.generator = sr_resnet()
        else:
            self.generator = tf.keras.models.load_model(self.train_dir+"/weights/gen_e_"+str(start_epoc)+'.h5')
        self.start_epoc = start_epoc + 1
    def fit(self,train_dataset,test_dataset,epochs = 50):
        for epoch in tqdm(range(epochs)):
            real_epch = epoch + self.start_epoc
            for lr,hr in train_dataset.take(-1).cache():
                self.train_step(lr,hr)
            print("epoc "+ str(real_epch))
            self.eval(test_dataset,real_epch)
            self.generator.save(self.train_dir+"/weights/gen_e_"+str(real_epch)+'.h5')


            #calc psnr # todo calc fid

    def eval(self,test_dataset,epoch):
        cont = 0
        for lr, hr in test_dataset:
            cont+=1
            display_handler.display_hr_lr(self.train_dir,self.generator,hr,lr,epoch,cont)



    @tf.function
    def train_step(self, lr, hr):
        with tf.GradientTape() as tape:

            sr = self.generator(lr, training=True)
            loss_value = self.loss(hr, sr)

        gradients = tape.gradient(loss_value, self.generator.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.generator.trainable_variables))





generator = GeneratorTrainer("sr_resnet",98)
generator.fit(data_loader.train_dataset,data_loader.validation_dataset.take(15).cache(),500)