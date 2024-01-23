import tensorflow as tf
import numpy as np
# 

data_path = ['./maestro/data/2004','./maestro/data/2006','./maestro/data/2008','./maestro/data/2009',
             './maestro/data/2011','./maestro/data/2013','./maestro/data/2014','./maestro/data/2015','./maestro/data/2017','./maestro/data/2018']
npy_path = './note_2004-'
year = ['2004','2006','2008','2009','2011','2013','2014','2015','2017','2018']
BATCH= 2
time_of_set = 4
max_sequence_length = 128
vocab_size = 128
drop_prob = 0.1
num_layers=  5
num_heads = 8
ffn_hidden = 1024
d_model = 128
PADDING_TOKEN = '_'
START_TOKEN = '<START>'
END_TOKEN = '<END>'

def delta():
    return time_of_set / max_sequence_length
def note_mapping():
    note_cate = [START_TOKEN]+[str(i) for i in range(128)] + [PADDING_TOKEN,END_TOKEN]
    note_to_indx = dict([(k,i) for i,k in enumerate(note_cate)])
    indx_to_note = dict([(i,k) for i,k in enumerate(note_cate)])

    return note_cate,note_to_indx,indx_to_note

def make_sort_indx(x,batch_size):
    '''
    sort tensor 
    [[0,x,0],[1,x,0],....,[m-1,x,n],[m,x,n]] 
    to 
    [[0,x,0],[0,x,1],...[m,x,n-1],[m,x,n]]
    '''
    t1 = tf.range(batch_size)
    t2 = tf.range(d_model)
    
    g1,g2 = tf.meshgrid(t1,t2)

    r = tf.stack([g1,tf.fill(tf.shape(g1),x),g2],axis=-1)

    r_flat = tf.reshape(r,[-1,3])

    indices = []
    for i in range(batch_size):
        indices = indices + [k for k in range(i,batch_size*d_model,batch_size)]
    # print('r_flat',r_flat)
    results = tf.gather(r_flat,indices)
    # print('results',results)
    results = results.numpy()
    results = results.tolist()
    return results

def indices_dict(batch_size):

    '''
    依照[m,x,n] 中 x 的不同給予不同的index
    '''
    index_list = list()
    for i in range(max_sequence_length):
        index_list.append(make_sort_indx(i,batch_size))
    
    return np.array(index_list)