from warnings import filters
import math
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import InputLayer,Reshape,Input,Add, Conv1D,Embedding,Activation,Dropout,LeakyReLU,Flatten,GRU,Dense,LSTM,Bidirectional,RepeatVector,TimeDistributed
from tensorflow.keras import Model,Sequential,activations
from tensorflow.keras.layers import *
from Params_setting import *
delta_ = delta()
note_cate,note_to_indx,indx_to_note = note_mapping()
##################### Transformer ####################
# indx_table = indices_dict()
# ------------- Attention --------------------#
def scaled_dot_product(q,k,v,mask=None):
    # q,k,b = 30 x 8 x 128 x 64
    d_k = q.shape[-1] # 64
    # 計算scaled
    num_dimensions = tf.rank(k)# 獲取 tensor 的維度數 
    perm = tf.concat([tf.range(num_dimensions - 2), tf.range(num_dimensions - 1, num_dimensions - 3, -1)], axis=0)
    scaled = tf.matmul(q,tf.transpose(k,perm=perm))/math.sqrt(d_k) # 30 x 8 x 128 x 128
    
    if mask:
        # masking for decoder
        mask_ = tf.fill(scaled.shape,float('-inf'))
        mask_ = tf.experimental.numpy.triu(mask_,k=1) # k=1 對角線右邊一個
        scaled += mask_ # 30 x 8 x 128 x 128
    attention = tf.nn.softmax(scaled,axis=-1) # 30 x 8 x 128 x 128
    values = tf.matmul(attention,v) # 30 x 8 x 200 x 64
    return values,attention

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.head_dim = d_model // num_heads # e.g. 512 // 8 = 64
        self.qkv_layer = Dense(3*d_model,input_shape=(d_model,)) # 512 x 1536
        self.linear_layer = Dense(d_model,input_shape=(d_model,)) # 512 x 512
        # self.batch_size = batch_size
        # 設定trainable 
        self.trainable = False
    
    def call(self,x,mask=None): # mask 另外弄
        
        _,sequence_length,d_model = x.shape # 30 x 128 x 512
        batch_size = tf.shape(x)[0]
        # print(f"x shape: {x.shape}")
        
        qkv = self.qkv_layer(x) # 30 x 128 x 1536
        # print(f"qkv shape: {qkv.shape}")
        
        qkv = tf.reshape(qkv,[batch_size,sequence_length,self.num_heads,3*self.head_dim]) # 30 x 128 x 8 x 192(64*3)
        # print(f"qkv shape: {qkv.shape}")
        qkv = tf.transpose(qkv,perm=[0,2,1,3]) # 30 x 8 x 128 x 192
        # print(f"qkv shape: {qkv.shape}")
        q,k,v, = tf.split(qkv,3,axis=-1) # each are 30 x 8 x 128 x 64
        # print(f"q shape: {q.shape} || k shape: {k.shape} || v shape: {v.shape}")
        values,attention = scaled_dot_product(q,k,v,mask) # attention = 30 x 8 x 128 x 128, values = 30 x 8 x 128 x 64
#         print(f"values shape: {values.shape} || attention shape: {attention.shape}")
        values = tf.reshape(values,[batch_size,sequence_length,self.num_heads*self.head_dim]) # 30 x 128 x 512
#         print(f"values shape: {values.shape}")
        out = self.linear_layer(values)
#         print(f"out shape: {out.shape}")
        return out
    
class MultiHeadCrossAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.head_dim = d_model // num_heads # e.g. 512 // 8 = 64
        self.kv_layer = Dense(2*d_model,input_shape=(d_model,)) # 512 x 1024
        self.q_layer = Dense(d_model,input_shape=(d_model,)) # 512 x 512
        self.linear_layer = Dense(d_model,input_shape=(d_model,)) # 512 x 512
        # 設定trainable 
        self.trainable = False
        # self.batch_size = batch_size
    
    def call(self,x,y,mask=None): # mask 另外弄
        
        _,sequence_length,d_model = x.shape # 30 x 128 x 512
        batch_size = tf.shape(x)[0]
#         print(f"x shape: {x.shape}")
        kv = self.kv_layer(x) # 30 x 128 x 1024
#         print(f"qkv shape: {qkv.shape}")
        q =  self.q_layer(y) # 30 x 128 x 512
        kv = tf.reshape(kv,[batch_size,sequence_length,self.num_heads,2*self.head_dim]) # 30 x 128 x 8 x 128
        q = tf.reshape(q,[batch_size,sequence_length,self.num_heads,self.head_dim]) # 30 x 128 x 8 x 64
#         print(f"qkv shape: {qkv.shape}")
        kv = tf.transpose(kv,perm=[0,2,1,3]) # 30 x 8 x 128 x 128
        q = tf.transpose(q,perm=[0,2,1,3]) # 30 x 8 x 128 x 64
#         print(f"qkv shape: {qkv.shape}")
        k,v = tf.split(kv,2,axis=-1) # k: 30 x 8 x 128 x 64 v: 30 x 8 x 128 x 64
#         print(f"q shape: {q.shape} || k shape: {k.shape} || v shape: {v.shape}")
        values,attention = scaled_dot_product(q,k,v,mask) # attention = 30 x 8 x 128 x 128, values = 30 x 8 x 128 x 64
#         print(f"values shape: {values.shape} || attention shape: {attention.shape}")
        values = tf.reshape(values,[batch_size,sequence_length,d_model]) # 30 x 128 x 512
#         print(f"values shape: {values.shape}")
        out = self.linear_layer(values) # 30 x 128 x 512
#         print(f"out shape: {out.shape}")
        return out

# ------------- LayerNormalization --------------------#
class LayerNorm(tf.keras.layers.Layer):
    def __init__(self,parameter_shape,eps=1e-5):
        super().__init__()
        self.parameter_shape = parameter_shape # [512]
        self.eps = eps
        self.gamma = tf.Variable(tf.ones(parameter_shape)) # [512]
        self.beta = tf.Variable(tf.zeros(parameter_shape)) # [512]
        
    def call(self,x): 
        # input = 30 x 200 x 512
        dims = [-(i+1) for i in range(len(self.parameter_shape))] # [-1]
        mean = tf.reduce_mean(x,axis=dims,keepdims=True) # 30 x 128 x 1
#         print(f"Mean \n ({mean.shape}): \n {mean}") 
        var = tf.reduce_mean(((x-mean)**2),axis=dims,keepdims=True) # 30 x 128 x 1
        std = tf.math.sqrt(var+self.eps) # 30 x 128 x 1
#         print(f"Standard Deviation \n ({std.shape}): \n {std}")
        y = (x - mean) / std # 30 x 128 x 512
#         print(f"y \n ({y.shape}) = \n {y}")
        out = self.gamma * y + self.beta # # 30 x 128 x 512
#         print(f"out \n ({out.shape}) = \n {out}")
        return out
    
# ------------- FeedForward Network --------------------#
class PositionwiseFeedForward(tf.keras.layers.Layer):
    def __init__(self,d_model,hidden,drop_prob=0.1):
        
        super(PositionwiseFeedForward,self).__init__()
        self.linear1 = Dense(hidden,input_shape=(d_model,)) # 512 x 2048
        self.linear2 = Dense(d_model,input_shape=(hidden,)) # 2048 x 512
        self.relu = tf.keras.layers.ReLU() # 
        self.dropout = Dropout(drop_prob)
        # 設定trainable 
        self.trainable = False
        
    def call(self,x): 
        # input = 30 x 128 x 512
        x = self.linear1(x) # 30 x 128 x 2048 
#         print(f"x \n {x.shape}")
        x = self.relu(x) # 30 x 128 x 2048
#         print(f"x \n {x.shape}")
        x = self.dropout(x) # 30 x 128 x 2048
#         print(f"x \n {x.shape}")
        x = self.linear2(x) # 30 x 128 x 512
#         print(f"x \n {x.shape}")
        return x
    
# ------------- Sequence Embedding --------------------#
class SequenceEmbedding(tf.keras.layers.Layer):
    
    def __init__(self,max_sequence_length,d_model,lan_to_index,START_TOKEN=None,END_TOKEN=None,PADDING_TOKEN=None):
        super().__init__()
        time_of_set = 4
        self.vocab_size = len(lan_to_index)
        self.max_sequence_length = max_sequence_length
        self.delta = time_of_set/max_sequence_length
        self.embedding = TemporalEmbedding(max_sequence_length=max_sequence_length,d_model=d_model)
        self.lan_to_index = lan_to_index
        self.position_encoder =PositionalEncoding(d_model,max_sequence_length)
        self.dropout = Dropout(0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        
    
    def call(self,x,batch_size): # sentence
        # print('se_input',x)
        x = self.embedding(x,batch_size)
        # print('se0',x)
        pos = self.position_encoder.call()
        # print('se1',pos)
        x = self.dropout(x+pos)
        # print('se2',x)
        # x = tf.reshape(x,[x.shape[0],x.shape[1]])# 從二維轉成三維(1,128,512)
        return x

# def get_temporal_embeddings(position, d_model):
#     angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model) // 2)) / np.float32(d_model))
#     return tf.constant(position * angle_rates,dtype=tf.float32)
def get_temporal_embeddings(position, d_model):
    angle_rates = 1 / tf.pow(10000.0, (2.0 * (tf.range(d_model, dtype=tf.float32) // 2.0) / tf.cast(d_model, dtype=tf.float32)))
    return position * angle_rates
# positions = tf.zeros([max_sequence_length, 0, 1], dtype=tf.float32)


class TemporalEmbedding(tf.keras.layers.Layer):

    def __init__(self,max_sequence_length,d_model):
        super(TemporalEmbedding,self).__init__()
        
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        # print(f'TemporalEmbedding max_seq_len: {self.max_sequence_length},d_model: {self.d_model}')
        # 設定trainable 
        # self.trainable = False
        self.temporal_weights = self.add_weight(
            shape=(d_model,max_sequence_length ),
            initializer='glorot_uniform',
            trainable=True,
            name='temporal_weights'
        )
    def call(self,inputs,batch_size):
        # inputs: N x d_model(every input sequence length)
        # output: N x max_seq_length x d_model

        inputs = tf.cast(inputs,dtype=tf.float32)
        bs  = tf.shape(inputs)[0]
        # print(np.arange(self.max_sequence_length))
        # print(inputs.shape[0])
        # 需要將position 改為使用Tensor做成
        
        # position = np.array([np.arange(self.max_sequence_length)[:, np.newaxis] for _ in range(batch_size)])
        # 定義 while_loop 的條件函數
        positions = tf.zeros([max_sequence_length, 0, 1], dtype=tf.float32)
        def condition(i, positions):
            return i < bs

        # 定義 while_loop 的主體函數
        def body(i, positions):
            # 在這裡進行每個迴圈的操作
            r = tf.range(max_sequence_length, dtype=tf.float32)
            r = tf.expand_dims(r, axis=-1)
            positions = tf.concat([positions, tf.expand_dims(r, axis=1)], axis=1)
            return i + 1, positions

        # 使用 tf.while_loop 進行迴圈
        i = tf.constant(0)
        _, positions = tf.while_loop(condition, body, [i, positions],shape_invariants=[i.get_shape(), tf.TensorShape([None, None, 1])])
        # 轉換形狀
        positions = tf.reshape(positions, [bs, max_sequence_length, 1])
        # print('position size:',position.shape)
        d_model = inputs.shape[-1]
        # print('d_model:',d_model)
        print('position',positions.shape)
        time_embedding = get_temporal_embeddings(positions, d_model)
        # print('time_embedding size:',time_embedding.shape)
        inputs = inputs[:,tf.newaxis,:]
        print(inputs.shape)
        print(time_embedding.shape)
        
        combined_embedding = inputs + time_embedding
        # print('combined_embedding size:',combined_embedding.shape)
        return combined_embedding * self.temporal_weights # n*max_seq*d_model
        
# ------------- Positional Encoding --------------------#
class PositionalEncoding(tf.keras.layers.Layer):
    
    def __init__(self,d_model,max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model
        
        self.weights_var = self.add_weight(
            shape=(max_sequence_length,d_model),
            initializer='glorot_uniform',
            trainable=True,
            name='positional_encoding_weights'
        )

    def call(self):
        even_i = tf.range(start=0,limit=self.d_model,delta=2,dtype=tf.float32)
        denominator = tf.pow(10000.0,even_i/self.d_model)
        # denominator = tf
        position = tf.reshape(tf.range(self.max_sequence_length,dtype=tf.float32),[self.max_sequence_length,1])
        even_PE = tf.math.sin(position/denominator)
        odd_PE = tf.math.cos(position/denominator)
        stacked = tf.stack([even_PE,odd_PE],axis=2)
        # print(stacked)
        PE = tf.reshape(stacked, [stacked.shape[0],stacked.shape[1]*stacked.shape[2]])
        # print(PE)
        PE = tf.cast(PE,dtype=tf.float32)
        return PE * self.weights_var
# ------------- Encoder --------------------#
class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        
        super(EncoderLayer,self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1 = LayerNorm(parameter_shape=[d_model])
        self.dropout1 = Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob )
        self.norm2 = LayerNorm(parameter_shape=[d_model])
        self.dropout2 = Dropout(drop_prob)
        # 設定trainable 
        self.trainable = False
    def call(self,x,mask):
        residual_x = x # 30 x 128 x 512
        x = self.attention(x,mask=mask) # 30 x 128 x 512
        x = self.dropout1(x) # 30 x 128 x 512
        x = self.norm1(x+residual_x) # 30 x 128 x 512
        residual_x = x # 30 x 128 x 512
        x = self.ffn(x) # 30 x 128 x 512
        x = self.dropout2(x) # 30 x 128 x 512
        x= self.norm2(x+residual_x) # 30 x 128 x 512
        
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers,
                 max_sequence_length,lan_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
        super().__init__()
        self.sequence_embedding = SequenceEmbedding(max_sequence_length,d_model,lan_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.layers = Sequential()
        for _ in range(num_layers):
            self.layers.add(EncoderLayer(d_model,ffn_hidden,num_heads,drop_prob))
        
        # 設定trainable 
        self.trainable = False
        
    def call(self,x,mask,batch_size):
        x = self.sequence_embedding(x,batch_size)
        # print(x)
        x = self.layers(x,mask)
        return x
# ------------- Decoder --------------------#
class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob):
        
        super(DecoderLayer,self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads)
        self.norm1 = LayerNorm(parameter_shape=[d_model])
        self.dropout1 = Dropout(drop_prob)
        # cross attention here
        self.encoder_decoder_attention =  MultiHeadCrossAttention(d_model=d_model,num_heads=num_heads)
        self.norm2 = LayerNorm(parameter_shape=[d_model])
        self.dropout2 = Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model,hidden=ffn_hidden,drop_prob=drop_prob)
        self.norm3 = LayerNorm(parameter_shape=[d_model])
        self.dropout3 = Dropout(drop_prob)
        # 設定trainable 
        self.trainable = False

    def call(self,x,y,mask=None,cross_mask=None):
        # print('y as input of layer',y)
        _y = y # for residual # 30 x 128 x 512
        y = self.self_attention(y,mask=mask) # 30 x 128 x 512
        y = self.dropout1(y) # 30 x 128 x 512
        y = self.norm1(y+_y) # 30 x 128 x 512
        
        _y = y # for residual # 30 x 128 x 512
        y = self.ffn(y)
        # Cross attention
        y = self.encoder_decoder_attention(x,y,mask=cross_mask)
        y = self.dropout2(y)
        y = self.norm2(y+_y)
        # print('y in layers',y)
        return y

class SequentialDecoder(Sequential):
    def call(self,*inputs):
        x,y,mask,cross_mask = inputs
        for layer in self.layers:
            y = layer(x,y,mask,cross_mask) # 30 x 128 x 512
        
        return y
    
class Decoder(tf.keras.layers.Layer):
    
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers,
               max_sequence_length,lan_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN):
        
        super().__init__()
        self.sequence_embedding = SequenceEmbedding(max_sequence_length,d_model,lan_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.layers  = SequentialDecoder()
        for _ in range(num_layers):
            self.layers.add(DecoderLayer(d_model,ffn_hidden,num_heads,drop_prob))
        # 設定trainable 
        self.trainable = False


    def call(self,x,y,mask,cross_mask,AT_table,batch_):
        # AT_table is for autoregressive loop 的時候 做tensor_scatter_nd_add() 的indices 使用的
        # x: 30 x 128 x 512
        # y: 30 x 128 x 512cab_size
        # mask: 128 x 128
        batch_size = tf.shape(y)[0]
        y = self.sequence_embedding(y,batch_)
        
        num = int(y.shape[1])
        for i in range(num):
            
            # print('y delta:',y[:,:i+1,:])
            att_output = self.layers(x[:,:i+1,:],y[:,:i+1,:],mask,cross_mask)
            # print('att_output:',att_output)
            att_output = tf.reshape(att_output,[-1])
            # print('y',y)
            
            # 因為有特別製作indx_table, 這樣就不用在網路反覆運算一樣且可以重複使用的東西
            # temp = tf.constant(indx_table[str(i)])[:,:2]
            # indices 也要 跟著y[:,:i+1,:]增加
            indices = AT_table[:i+1,:,:]
            indices = tf.reshape(indices,[indices.shape[0]*indices.shape[1],indices.shape[2]])
            # print('now in loop:',i)
            # print('scatter y:',y)
            # print('scatter indices:',indices)
            # print('scatter att_output:',att_output)
            y = tf.tensor_scatter_nd_add(y,indices,att_output)
            # indices = tf.range(tf.shape[y][1])
            # y[:,i,:] += att_output[:,-1,:]
            
        # print(y)
        return y# 30 x 128 x 512
# ------------- Transformer --------------------#

class Transformer(Model):
    
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers,
                max_sequence_length,vocab_size,origin_to_index,transform_to_index,
                 START_TOKEN,END_TOKEN,PADDING_TOKEN,delta):

        super().__init__()
        self.start = START_TOKEN
        self.padding = PADDING_TOKEN
        self.end = END_TOKEN
        self.transform_to_indx = transform_to_index
        self.vocab_size = vocab_size
        self.delta = delta
        self.max_seq_len = max_sequence_length
        self.encoder = Encoder(d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,origin_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.decoder = Decoder(d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,transform_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.linear = Dense(vocab_size,input_shape=(d_model,))    
        # 設定trainable 
        self.trainable = False
        
    def call(self,inputs,y,AT_table,batch_,encoder_mask=None,decoder_mask=None,cross_mask=None,): # x, y are batch of sentence
        
        # 準備好輸出
        output = []
        
        # 生成一個空的值(with start)
        # y = [random.randint(0,131) for _ in range(self.max_seq_len)]
        
        x = inputs
        
        batch_size = tf.shape(x)[0]
        # print(batch_size)
        # print(y_)
        # y_ = tf.constant(y_,dtype=tf.float32)
        
        
        y = tf.reshape(y,[batch_size,y.shape[1]])
        y = tf.cast(y,dtype=tf.float32)
        # print(y)
        
        # print('y shape at Decoder input',y.shape)
        # y = tf.one_hot(y,self.vocab_size)
        # y = tf.reshape(y,[1,y.shape[0],y.shape[1]])
        # print('X for encoder:',x)
        x = self.encoder(x,encoder_mask,batch_)
        
        
        # print(self.max_seq_len,'/',self.delta**(-1))
        
        
        out = self.decoder(x,y,decoder_mask,cross_mask,AT_table,batch_)
        # print('decoder output',out)
        out = self.linear(out)
        # print('linear',out)
        out = tf.argmax(out,axis=2)
        # print('output',out)
        return out

class TransformerEncoderOnly(Model):
    
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob,num_layers,
                max_sequence_length,vocab_size,origin_to_index,transform_to_index,
                 START_TOKEN,END_TOKEN,PADDING_TOKEN,delta):

        super().__init__()
        self.start = START_TOKEN
        self.padding = PADDING_TOKEN
        self.end = END_TOKEN
        self.transform_to_indx = transform_to_index
        self.vocab_size = vocab_size
        self.delta = delta
        self.max_seq_len = max_sequence_length
        self.encoder = Encoder(d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,origin_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.decoder = Decoder(d_model,ffn_hidden,num_heads,drop_prob,num_layers,max_sequence_length,transform_to_index,START_TOKEN,END_TOKEN,PADDING_TOKEN)
        self.linear = Dense(vocab_size,input_shape=(d_model,))    
        # 設定trainable 
        self.trainable = False
        
    def call(self,inputs,batch_=2,encoder_mask=None,decoder_mask=None,cross_mask=None,): # x, y are batch of sentence
        
        # 準備好輸出
        output = []
        
        # 生成一個空的值(with start)
        x = inputs
        
        batch_size = tf.shape(x)[0]
        
        x = tf.reshape(x,[batch_size,x.shape[1]])
        x = tf.cast(x,dtype=tf.float32)

        out = self.encoder(x,encoder_mask,batch_)

        # print('decoder output',out)
        out = self.linear(out)
        # print('linear',out)
        out = tf.argmax(out,axis=2)
        # print('output',out)
        return out

class Discriminator(Model):

    def __init__(self,input_shape):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            InputLayer(input_shape=input_shape),
            LSTM(256,input_shape=input_shape),
            Dropout(0.2),
            LeakyReLU(0.2),
            Dense(1, activation='sigmoid')
        ])
    
    def call(self, inputs, training):
        return self.model(inputs)

