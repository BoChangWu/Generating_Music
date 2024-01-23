import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

#### Loss Function ####
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

def discriminator_loss(real_output,fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

generator_optimizer = Adam(learning_rate=1e-9,beta_1=0.8,beta_2=0.999,amsgrad=True,epsilon=1e-8)
discriminator_optimizer = Adam(learning_rate=1e-9,beta_1=0.8,beta_2=0.999,amsgrad=True,epsilon=1e-8)


class ModelMonitor(Callback):
    def __init__(self,num_img=3,latent_dim=256):
        self.num_img = num_img
        self.latent_dim = latent_dim
        
    def on_epoch_end(self,epoch,logs=None):
        pass
        # random_latent_vectors = tf.random.uniform((self.num_img,self.latent_dim,3))
        # generated_images = self.model.generator(random_latent_vectors)
        # generated_images *= 255
        # generated_images.numpy()
        
        