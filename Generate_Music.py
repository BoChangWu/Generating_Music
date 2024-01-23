import tensorflow as tf
import random
from Transformer import TransformerGenerator,TransformerDiscriminator
from Optimize import generator_optimizer, discriminator_optimizer,generator_loss, discriminator_loss
from Midi_Edit import Midi_Encode
from GAN_Module import GAN
#### Params ####
from Params_setting import *
#### Params ####

hparams = default_hparams()

print('hparams["Heads"]: ',hparams['Heads'])
################

#### Models ####
generator = tf.keras.models.load_model('./g_model')
generator.summary()

discriminator = tf.keras.models.load_model('./d_model')
discriminator.summary()
################
Gan = GAN(generator= generator,
                discriminator= discriminator,
                datapath= 'preprocess_data.npy',
                g_loss= generator_loss,
                d_loss= discriminator_loss,
                g_opt= generator_optimizer,
                d_opt= discriminator_optimizer,
                data_length=10000)
for i in range(10):
    predict =Gan.predict(random_seed=random.randint(0,127))

    midi_encode = Midi_Encode()

    midi_encode.make_file(params=predict,name=f'./test/sample{i}.mid')