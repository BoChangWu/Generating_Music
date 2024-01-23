
from tensorflow.keras import Model,Sequential
import random
import tensorflow as tf
import numpy as np
from Midi_Edit import Midi_
from Params_setting import (ffn_hidden,num_heads,npy_path,year,max_sequence_length,d_model,time_of_set,num_layers,drop_prob,vocab_size,
note_mapping,PADDING_TOKEN,START_TOKEN,END_TOKEN,BATCH,delta,indices_dict)
from Transformer import SequenceEmbedding,Discriminator,Transformer,TransformerEncoderOnly
from GAN_Module import TransformerGAN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy,MeanSquaredError

note_cate,note_to_indx,indx_to_note = note_mapping()

print('=====INPUT=====')
noise = tf.random.normal(shape=(1,128))
print('noise',noise)
predict_table = indices_dict(1)
output = []
y = year[-1]
epochs_ = 25

# 取X
X = np.load('./unite_inputs.npy')
X  = X.reshape((X.shape[0],X.shape[1]))
X = tf.constant(X)

# 取Y
sequences = np.load(npy_path+y+'.npy')
Y = sequences
Y = Y.reshape((Y.shape[0],Y.shape[1]))
Y = tf.constant(Y)
delta_ = delta()

generator = TransformerEncoderOnly(d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,
                        vocab_size,note_to_indx,note_to_indx,START_TOKEN,END_TOKEN,PADDING_TOKEN,delta_)
discriminator = Discriminator(input_shape=(1,max_sequence_length))

# 弄一個梯度裁剪避免梯度爆炸
# 梯度裁剪閥值
clip_value = 1.0
# clipvalue=clip_value
g_opt = Adam(learning_rate=1e-4)
d_opt = Adam(learning_rate=1e-4)
# g_loss = BinaryCrossentropy()
g_loss = MeanSquaredError()
d_loss = BinaryCrossentropy()
gan = TransformerGAN(generator,discriminator)

gan.compile(g_opt,d_opt,g_loss,d_loss)

# print('inputs for GAN:',X)
with tf.device('/device:GPU:0'):
    hist = gan.fit(Y,Y,batch_size=BATCH,epochs=epochs_,verbose=1)

# gan.generator.save('./g_model')
# gan.discriminator.save('./d_model')
# gan.predict(X[0])
start = tf.zeros([1,128])
out = gan.generator(start,batch_=1)
out = out.numpy()


midi_encode = Midi_()
midi_encode.make_file(notes=out,name = f'test15_{y}_{epochs_}.mid')

print('=====OUTPUT=====')
print(out)

generator.save_weights('./model2/generator_weights',save_format='tf')

discriminator.save_weights('./model2/discriminator_weights',save_format='tf')