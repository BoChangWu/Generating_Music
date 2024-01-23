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

delta_ = delta()

# 創建新的生成器和鑑別器模型
generator = TransformerEncoderOnly(d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,
                        vocab_size,note_to_indx,note_to_indx,START_TOKEN,END_TOKEN,PADDING_TOKEN,delta_)
discriminator = Discriminator(input_shape=(1,max_sequence_length))

# 載入生成器和鑑別器的權重
generator.load_weights('./model_weights/generator_weights')
discriminator.load_weights('./model_weights/discriminator_weights')

# 創建新的 GAN 模型
gan = TransformerGAN(generator,discriminator)

# 梯度裁剪閥值
clip_value = 1.0

g_opt = Adam(learning_rate=1e-6,clipvalue=clip_value)
d_opt = Adam(learning_rate=1e-6)
g_loss = MeanSquaredError()
d_loss = BinaryCrossentropy()
gan = TransformerGAN(generator,discriminator)

gan.compile(g_opt,d_opt,g_loss,d_loss)

start = tf.zeros([1,128])
out = gan.generator(start,batch_=1)
out = out.numpy()


midi_encode = Midi_()
midi_encode.make_file(notes=out,name = f'test15_2018_300.mid')

print('=====OUTPUT=====')
print(out)

# generator.save_weights('./model2/generator_weights',save_format='tf')

# discriminator.save_weights('./model2/discriminator_weights',save_format='tf')