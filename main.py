# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 10:29:32 2017

@author: jbraga
"""

#This is quite basic: The loweset level ever part of thesis.
#Eval is 100% perceptive. Synth of pitch contour is conducted.
#-------------------------------------------------------------
#OUTLINE OF THIS PYTHON SCRIPT:
#1. Load Audio
#2. Pitch Extraction
#3. Synth based EvaL
#-------------------------------------------------------------

import tradataset as td
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from aubio import pitch

if __name__=='__main__':
    
# ----> LOAD AUDIO & ANNOTATIONS
    
    ltrdataset = td.load_list()    
    fragment = ltrdataset[9]    
    audio_file = fragment + '_mono.wav'
    synth_file = fragment + '_synth.wav'
    audio, t, fs = td.load_audio(audio_file)
    
#    plt.figure(figsize=(16,5))
#    plt.plot(t, audio), plt.grid(), plt.axis('tight')    

    win=512
    hop=256
    
    f, t, Sxx = signal.spectrogram(audio, fs, window=('hann'), 
                                   nperseg=win, noverlap=win-hop, nfft=2*win, 
                                   detrend='constant', return_onesided=True, 
                                   scaling='density', axis=-1, mode='psd')
    Sxx = np.log10(Sxx)
                                      
#%%----> PITCH EXTRACTION
 
    tolerance = 0.05

    pitch_o = pitch("yin", win, hop)
    pitch_o.set_unit("Hz")
    pitch_o.set_tolerance(tolerance)
    pitch_o.set_silence(-1)

    pitches = []
    confidences = []

    # total number of frames read
    total_frames = len(audio)/hop
    for i in range(0,total_frames-1):
    
        samples = audio[i*hop:i*hop+win]
        pitch = pitch_o(samples)[0]
        
        confidence = pitch_o.get_confidence()
        if confidence < 0.8: pitch = 0.

        pitches += [pitch]
        confidences += [confidence]

    timestamps = np.arange(len(pitches)) * (hop/44100.0)

    #Flute Acoustics Knowledge
    pitches = np.array(pitches)
    melody_hz = np.copy(pitches)
    melody_hz[pitches<=200] = None
    melody_hz[pitches>2500] = None
    pitches[pitches<=200] = 0
    pitches[pitches>2500] = 0
    
#%% PLOT

    plt.figure(figsize=(16,5))
    plt.pcolormesh(t, f, Sxx, cmap='gray')
    plt.plot(timestamps, melody_hz)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.axis('tight')
    plt.ylim([200, 2000])
    plt.show()
    
#%% SYNTH
    
    import melosynth as ms
    ms.melosynth_pitch(pitches, 'melosynth.wav', fs=44100, nHarmonics=1, square=True, useneg=False) 

    from pydub import AudioSegment
    sound1 = AudioSegment.from_file(audio_file)
    sound1 = sound1.pan(+1)
    sound2 = AudioSegment.from_file("melosynth.wav")
    sound2 = sound2.apply_gain(-10)
    sound2 = sound2.pan(-1)
    
    combined = sound1.overlay(sound2)
    combined.export(synth_file, format='wav')
    
