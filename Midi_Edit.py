import random
from mido import MidiFile , MidiTrack, Message


class Midi_Encode():
    def __init__(self,):
        self.midler = MidiFile()
        self.track = MidiTrack()

        self.midler.tracks.append(self.track)
        self.track.append(Message('program_change',program=2,time=0))

    def make_file(self,params,notes_nums=256,name='Midi_Sample.mid'):
        for x in range(notes_nums):
            # on_interval = random.randint(50,127)
            off_interval = random.randint(0,127)
            change_interval = random.randint(0,127)
            change_value = random.randint(0,127)
            isControl = random.randint(0,1)
    
        self.track.append(Message('note_on', channel=1, note=int(params[0][x][0]), velocity=64, time=5))

        if isControl:
            self.track.append(Message('control_change', channel=1, control=64, value=change_value, time= change_interval))

        # self.track.append(Message('note_off', channel=1, note=int(params[0][x][0]), velocity=int(params[0][x][1]), time=int(off_interval)))

        self.midler.save(name)

class Midi_():
    
    def __init__(self):
        self.mid = MidiFile()
        self.track = MidiTrack()

    def play_part(self,note, len_ ,note_bias=0,vel=1,delay=0,change=False,double=False):
    # 每個節拍的時間長度
        
        temple = 60*60*10/75
        # 大調，參考別人的做法的，我也不是很懂樂理
        major_notes = [0,2,2,1,2,2,2,1]
        # C4 - 正中間的 DO(60)
        base_note = note
        # print(base_note)
        bias = random.randint(-1,1)
        vel = round(64*vel)
        # delay = random.random()
        t_start = round(delay*temple)
        t_end =  round(temple*len_)

        # if base_note < 128 and base_note > 0:
            # base_note = base_note+bias
        

        if not double:
            self.track.append(Message("note_on",  note=base_note,velocity=vel,time=t_start))
            self.track.append(Message("note_off", note=base_note,velocity=vel,time=t_end))
        if change:
            self.track.append(Message("control_change",channel=0,control=64,value=64,time=t_start))
            self.track.append(Message("control_change",channel=0,control=64,value=0,time=t_end))
        if double:
            self.track.append(Message("program_change", channel=1,program=41 ,time = t_start))
            self.track.append(Message("note_on",channel=1, note=base_note,velocity=vel,time = t_start))
            self.track.append(Message("note_off",channel=1, note=base_note,velocity=vel,time = t_end))
            self.track.append(Message("program_change", channel=1,program=0 ,time=t_end))
        # return self.track

    def make_file(self,notes,name='new_song.mid'):
        
        for n in notes[0]:
            self.play_part(int(n),0.5)

        self.mid.tracks.append(self.track)

        self.mid.save(name)