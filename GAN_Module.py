
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint
from IPython.display import clear_output
from Params_setting import *
from tensorflow.keras import Model

# 梯度裁剪閥值
clip_value = 1.0
class GAN():
    def __init__(self,generator,discriminator,datapath,g_loss,d_loss,g_opt,d_opt,data_length=0):
        self.generator = generator
        self.discriminator = discriminator
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.train_data = np.load(datapath)
        self.gloss_list=[]
        self.dloss_list=[]
        self.gloss_name=''
        self.dloss_name=''

        if data_length > 0:
            self.train_data = self.train_data[:data_length]
        # rescale 0 to 1
        self.train_data = self.train_data.reshape(-1,256,1)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(self.train_data).batch(Batch,drop_remainder=True)
        
        # print(self.train_data.shape)

    #### Train ####
    @tf.function
    def train_step(self,music):
        tf.random.set_seed(5)
        noise = tf.random.normal(shape=[Batch,Time,3],dtype=tf.float32,seed=12)
        # noise = tf.random.normal([Batch,Time,1])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_music = self.generator(noise,training=True)
            real_output = self.discriminator(music,training=True)
            fake_output = self.discriminator(generated_music,training=True)

            gen_loss = self.g_loss(fake_output)
            disc_loss = self.d_loss(real_output,fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)
        self.g_opt.apply_gradients(zip(gradients_of_generator,self.generator.trainable_variables))
        self.d_opt.apply_gradients(zip(gradients_of_discriminator,self.discriminator.trainable_variables))
        
        return gen_loss,disc_loss

    def train(self,epochs):
        total_step = 0
        G_loss = 0
        D_loss = 0

        for epoch in range(epochs):
            start = time.time()

            for i, image_batch in enumerate(self.train_dataset):
                gen_loss, disc_loss = self.train_step(image_batch)
                print(f"Step: {i} || G_loss: {gen_loss}, D_loss: {disc_loss} ||")
                G_loss += gen_loss
                D_loss += disc_loss
                total_step += 1

                if total_step%100 == 0:
                    clear_output(wait=True)
                    print(f"G_AVE_loss: {G_loss/100}")
                    print(f"D_AVE_loss: {D_loss/100}")
                    self.gloss_list.append(G_loss/100)
                    self.dloss_list.append(D_loss/100)
                    G_loss = 0
                    D_loss = 0

            print(f"Time for epoch {epoch + 1} is {time.time()-start} sec\n")
        self.generator.save('./g_model')
        self.discriminator.save('./d_model')

    def predict(self,random_seed):
        tf.random.set_seed(5)
        noise = tf.random.normal(shape=[Batch,Time,3],dtype=tf.float32,seed=random_seed)  
        predict = self.generator.predict(noise)
        outputs = predict*128
        print(outputs)
        return outputs

    def save_loss(self,gloss_name,dloss_name):

        np.save(gloss_name,np.array(self.gloss_list))
        np.save(dloss_name,np.array(self.dloss_list))
        self.gloss_name = gloss_name
        self.dloss_name = dloss_name

    def show_graph(self):
        g= np.load(self.gloss_name+'.npy')
        month = [i for i in range(len(g))]
        plt.plot(g,'s-',color='r', label= 'generator loss')

        d = np.load(self.dloss_name+'.npy')
        plt.plot(d,'s-',color='b', label='discriminator loss')
        plt.legend(loc='best',fontsize=12)
        plt.show()

class Music_GAN(Model):
    def __init__(self,generator,discriminator,*args,**kwargs):
        # Pass through arg and kwargs to base class
        super().__init__(*args,**kwargs)
        
        # Create attribute for gen and disc
        self.generator = generator
        self.discriminator = discriminator
        
    def compile(self,g_opt,d_opt,g_loss,d_loss,*args,**kwargs):
        # Compile with base class
        super().compile(*args,**kwargs)
        
        # Create attribute for Losses and optimizers
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss 
        self.g_loss_series = []
        self.d_loss_series = []
    def train_step(self,batch):

        real_music = batch
        # print(real_music.shape)
        # real_music = tf.reshape(real_music,(1,real_music.shape[0],1))
        # print(real_music.shape)
        tf.random.set_seed(5)

        # noise = tf.random.normal(shape=[Batch,Time,Time],dtype=tf.float32,seed=100)
        # one-hot noise
        noise = [randint(0,Time) for _ in range(Time)]
        noise = tf.one_hot(noise,depth=Time)
        noise = tf.reshape(noise,[Batch,Time,Time])
        # for one-hot
        # print(noise)
        # noise = np.argmax(noise,axis=1)
        # noise = tf.one_hot(np.argmax,Time)
        # noise = tf.random.normal([Batch,Time,1])
        
        with tf.GradientTape() as d_tape:
            generated_music = self.generator(noise,training=False)
            # print(generated_music.shape)
            yhat_real = self.discriminator(real_music,training=True)
            yhat_fake = self.discriminator(generated_music,training=True)
            
            
            # predict
            yhat_realfake = tf.concat([yhat_real,yhat_fake],axis=0)
            # Create labels real and fake images 
            # actual label
            y_realfake = tf.concat([tf.zeros_like(yhat_real),tf.ones_like(yhat_fake)],axis=0)
            # Add some noise to the TRUE output
            noise_real = 0.7*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.7*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real,noise_fake],axis=0)
            
            total_d_loss = self.d_loss(y_realfake,yhat_realfake)
            
            # Apply backpropagation - nn learn
            dgrad = d_tape.gradient(total_d_loss,self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad,self.discriminator.trainable_variables))
        
        # Train the Generator
        with tf.GradientTape() as g_tape:
            # Generate some new 
            tf.random.set_seed(10)
            # noise = tf.random.normal(shape=[Batch,Time,Time],dtype=tf.float32,seed=100)
            # one-hot noise
            noise = [randint(0,Time) for _ in range(Time)]
            noise = tf.one_hot(noise,depth=Time)
            noise = tf.reshape(noise,[Batch,Time,Time])
            # for one-hot
            # noise = np.argmax(noise,axis=1)
            # noise = tf.one_hot(np.argmax,Time)
            gen_music = self.generator(noise,training=True)
            # Create the predicted labels
            predicted_labels = self.discriminator(gen_music,training=False)
            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels),predicted_labels)
            ggrad = g_tape.gradient(total_g_loss,self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad,self.generator.trainable_variables))
        
        
        return {"d_loss": total_d_loss,"g_loss": total_g_loss}
            
class TransformerGAN(Model):

    def __init__(self,generator,discriminator,*args,**kwargs):
        # Pass through arg and kwargs to base class
        super().__init__(*args,**kwargs)
        
        # Create attribute for gen and disc
        self.generator = generator
        self.discriminator = discriminator
        self.table  = indices_dict(BATCH)
    def compile(self,g_opt,d_opt,g_loss,d_loss,*args,**kwargs):
        # Compile with base class
        super().compile(*args,**kwargs)
        
        # Create attribute for Losses and optimizers
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss 
        self.g_loss_series = []
        self.d_loss_series = []

    def train_step(self,data):
        batch,label = data
        batch = tf.cast(batch,dtype=tf.float32)
        print('input_shape',batch.shape)
        batch_size = tf.shape(batch)[0]
        noise = tf.random.normal(shape=(batch_size,tf.shape(batch)[1]))

        with tf.GradientTape() as d_tape:
            
            # print('batch:',batch)
            # print('noise',noise)
            
            # generated_music = self.generator(batch,noise,self.table,BATCH,training=False)
            # encoder only
            # noise 跟 batch 結合
            
            inputs = batch + noise
            generated_music = self.generator(inputs,BATCH,training=False)
            # print(generated_music)
            # print(generated_music.shape)
            # 符合LSTM 的dim
            y_real = tf.reshape(label,[batch_size,1,label.shape[1]])
            generated_music = tf.reshape(generated_music,[batch_size,1,batch.shape[1]])
            
            yhat_real = self.discriminator(y_real,training=True)
            yhat_fake = self.discriminator(generated_music,training=True)
            
            
            # predict
            # 比對時原始資料跟generator 生成的資料的shape不一致
            yhat_realfake = tf.concat([yhat_real,yhat_fake],axis=0)
            # Create labels real and fake images 
            # actual label
            y_realfake = tf.concat([tf.zeros_like(yhat_real),tf.ones_like(yhat_fake)],axis=0)
            # Add some noise to the TRUE output
            noise_real = 0.5*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.5*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real,noise_fake],axis=0)
            
            total_d_loss = self.d_loss(y_realfake,yhat_realfake)
            
            # Apply backpropagation - nn learn
            dgrad = d_tape.gradient(total_d_loss,self.discriminator.trainable_variables)
            
        self.d_opt.apply_gradients(zip(dgrad,self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            # Generate some new 
            tf.random.set_seed(100)
            
            noise = tf.random.normal(shape=(tf.shape(batch)[0],tf.shape(batch)[1]))
            # print('noise for gen training:',noise)
            # gen_music = self.generator(batch,noise,self.table,BATCH,training=True)
            # encoder only
            # noise 跟 batch結合
            inputs = batch + noise
            gen_music = self.generator(inputs,BATCH,training=True)
            gen_music = tf.reshape(gen_music,[batch_size,1,gen_music.shape[1]])
            # Create the predicted labels
            predicted_labels = self.discriminator(gen_music,training=False)
            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels),predicted_labels)
            ggrad = g_tape.gradient(total_g_loss,self.generator.trainable_variables)

            # 應用梯度之前進行梯度裁剪
            ggrad = tf.clip_by_global_norm(ggrad, clip_value)
        self.g_opt.apply_gradients(zip(ggrad,self.generator.trainable_variables))

        with tf.GradientTape() as g_tape:
            # Generate some new 
            tf.random.set_seed(100)
            
            noise = tf.random.normal(shape=(tf.shape(batch)[0],tf.shape(batch)[1]))
            # print('noise for gen training:',noise)
            # gen_music = self.generator(batch,noise,self.table,BATCH,training=True)
            # encoder only
            # noise 跟 batch結合
            inputs = batch + noise
            gen_music = self.generator(inputs,BATCH,training=True)
            gen_music = tf.reshape(gen_music,[batch_size,1,gen_music.shape[1]])
            # Create the predicted labels
            predicted_labels = self.discriminator(gen_music,training=False)
            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels),predicted_labels)
            ggrad = g_tape.gradient(total_g_loss,self.generator.trainable_variables)
        
        self.g_opt.apply_gradients(zip(ggrad,self.generator.trainable_variables))

        return {"d_loss": total_d_loss,"g_loss": total_g_loss}
    
    # def call(self,batch):
