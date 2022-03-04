import tkinter as tk
import sounddevice as sd
import wavio as wv
import numpy as np
import librosa
import pandas as pd
import pickle
import time
import threading
import warnings
import os
warnings.filterwarnings('ignore')

class window_tk(tk.Frame):
    
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        tk.Frame.update(self)
        self.root = parent
        self.Model = pickle.load(open("finalized_model.sav", 'rb'))
        self.freq = 44100
        self.recording = False
        self.record = False
        self.thread = False
        self.root.geometry("800x430")
        self.root.title("Gender Voice Predictor")
        self.root.resizable(0,0)
        self.funID = False
        root.configure(background='#272525')
        self.GenderLabel = ["Female","Male"]
        self.hours, self.minutes, self.seconds = 0, 0, 0
        self.seconds_string=0
        self.Label=tk.Label(self.root, text="Click the Button to start recording your voice !" , bg="#272525" ,font=("Arial", 11))
        self.stopwatch_label = tk.Label(text='00:00:00', font=('Arial', 80) , bg="#272525" ,foreground="white" )
        self.stopwatch_label.pack()
        self.stopwatch_label.place(rely=0.4 , anchor='center', relx=0.5)
        self.Start = tk.PhotoImage(file="Icons/Start.png")
        self.Reset = tk.PhotoImage(file="Icons/reset.png")
        self.Reset = self.Reset.subsample(2,2)
        self.Exit = tk.PhotoImage(file="Icons/Exit.png")
        self.start=tk.Button(self.root,image=self.Start ,command=self.recorder_State, height = 75, width = 95 , border = 0 )
        self.reset=tk.Button(self.root,image=self.Reset ,command=self.reset_State, height = 75, width = 95 , border = 0 )
        self.quit_ = tk.Button(self.root,image=self.Exit ,command=exit, height = 75, width = 95 , border = 0 )
        self.gender=tk.Label(self.root, text="Gender :" , bg="#272525" , font=("Arial" ,9))
        self.gender_predict=tk.Label(self.root, text="{}".format("________") , bg="#272525" , font=("Arial" ,9))#,bg="#000042")
        self.designe()

    def designe(self):
        # designe my GUI
        self.Label.configure(foreground="white" )
        self.gender.configure(foreground="white")
        self.gender_predict.configure(foreground="orange")
        self.start.configure(background="#272525")
        self.reset.configure(background="#272525")
        self.quit_.configure(background="#272525")
        self.Label.place(x=270,y=20)
        self.start.place(x=180,y=250)
        self.reset.place(x=360,y=250)
        self.quit_.place(x=520,y=250)
        self.gender.place(x=300,y=360)
        self.gender_predict.place(x=460,y=360)
        
    def recorder_State(self):
        self.recording = not self.recording
        if self.recording:
            self.recorder()
        else:
            threading.Thread(target=self.Save_Record(), args=[])

    def reset_State(self):
        self.stopwatch_label.after_cancel(self.funID)
        self.hours, self.minutes, self.seconds = 0, 0, 0
        self.recording = False
        sd.stop()
        self.gender_predict.config(text="{}".format("________"))
        self.stopwatch_label.config(text='00:00:00')

    def updater(self):
        if self.recording :
            self.seconds += 1
            if self.seconds == 60:
                self.minutes += 1
                self.seconds = 0
            if self.minutes == 60:
                self.hours += 1
                self.minutes = 0
            hours_string = f'{self.hours}' if self.hours > 9 else f'0{self.hours}'
            minutes_string = f'{self.minutes}' if self.minutes > 9 else f'0{self.minutes}'
            self.seconds_string = f'{self.seconds}' if self.seconds > 9 else f'0{self.seconds}'
            self.stopwatch_label.config(text=hours_string + ':' + minutes_string + ':' + self.seconds_string)
            if int(self.seconds_string)%5==0:
                threading.Thread(target=self.Save_Record(), args=[])
            self.funID = self.stopwatch_label.after(1000, self.updater) 
        else: return
    def recorder(self):
        duration = 100
        self.record = sd.rec(int(duration * self.freq),samplerate=self.freq, channels=2)
        threading.Thread(target=self.updater(), args=[])
        
    def Save_Record(self):
        Start = int(len(self.record)/100*(int(self.seconds_string)-5))
        Stop = int(len(self.record)/100*int(self.seconds_string))
        if int(self.seconds_string)>=5:
            record = np.array([self.record[data] for data in range(Start,Stop)])
        else :
            record = record = np.array([self.record[data] for data in range(Stop)])
        name_ = 'recorded/Snap__' + str(int(time.time()))+".wav"
        wv.write(name_, record, self.freq, sampwidth=2)
        self.gender_predict.config(text="{}".format(self._predict(name_)))

    def _predict(self,audioFilename):
        prp = self.extract_feature(audioFilename)
        Xpred = pd.DataFrame([prp])
        return self.GenderLabel[self.Model.predict(Xpred)[0]]

    def extract_feature(self,_file_name, **kwargs):
        mel = kwargs.get("mel")
        _file_name.replace('\\','/')
        X, sample_rate = librosa.core.load(_file_name)
        result = np.array([])
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
        return result

if __name__ == "__main__":
    root = tk.Tk()
    window_tk(root)
    root.mainloop()