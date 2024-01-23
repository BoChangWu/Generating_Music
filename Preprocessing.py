import os 
import time
from mido import MidiFile
from math import sqrt
import numpy as np
import tensorflow as tf

from Params_setting import data_path,npy_path,max_sequence_length,time_of_set,note_mapping,d_model,BATCH

_,note_to_indx,_ = note_mapping()
class NotesProcessing():

    def __init__(self):
        self.note_to_indx = note_to_indx
        self.paths = []
        self.songs = []
        self.sequence = []
        self.delta = time_of_set/max_sequence_length

    def open_files(self,dir):

        for r,d,f in os.walk(dir):
            '''
            r: route
            d: folder in this folder
            f: file in this folder
            '''

            # if the file is mid file then append its path to paths[]
            for file in f:
                if '.mid' in file:
                    self.paths.append(os.path.join(r,file))

        for path in self.paths:
            mid = MidiFile(path,type=1)
            self.songs.append(mid)

    def preprocessing(self,name=None,path='./note_'):


        '''
        r: route
        d: folder in this folder
        f: file in this folder
        '''

        # if the file is mid file then append its path to paths[]
        notes = []
        dataset = []
        chunk = []


        # for each in midi object in list of songs
        for i in range(len(self.songs)):
            for msg in self.songs[i]:
                # filtering out meta messages
                if not msg.is_meta:
                    # filtering out control changes
                    if (msg.type =='note_on'):
                        notes.append(msg.note)

            for note in notes:
                chunk.append(note)
                # save each 16 note chunk
        #         if (j% max_seq_len==0):
                if len(chunk) == d_model:
                    dataset.append(np.array(chunk))
                    chunk = []
            print(f"Processing {i} Song",end='\r')
            chunk=[]
            notes=[]

        # print(dataset)
        if len(dataset) % BATCH != 0:
            dataset = dataset[:len(dataset)-(len(dataset)%BATCH)]
            
        sequences = np.array(dataset)
        print(sequences.shape)
        # print(len(sequences))
        # print(np.array([len(seq) for seq in sequences]))
        if name:
            # print(path+name)
            np.save(path+name,sequences,allow_pickle=True)
        else:
            np.save(npy_path,sequences,allow_pickle=True)
        print('Preprocessing done...')
        return sequences
        