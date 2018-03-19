import numpy as np
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.mlab import dist_point_to_segment
import os
import wave
import struct
import sys

clipdir = '/Users/CorbanSwain/Google Drive'
clipname = 'test2'

clipname += '.wav'
clipfile = os.path.join(clipdir, clipname) 

audio = wave.open(clipfile, 'r')

print('N Channels: %d' % audio.getnchannels())
print('N Frames: %d' % audio.getnframes())
print('Sample Width: %d bytes' % audio.getsampwidth())

nframes_to_read = 50
for iFrame in range(nframes_to_read):
    wave_data = audio.readframes(1)
    data = struct.unpack('<i', wave_data)
    print('%2d - %10d' % (iFrame, int(data[0])))



# def wav_to_floats(w):
#     astr = w.readframes(w.getnframes())
#     print('Length of output = %d' % len(astr))
#     print('length out / (nfames * nchanels) = %d' % (len(astr) / (w.getnframes() * w.getnchannels())))
#     # convert binary chunks to short 
#     a = struct.unpack("%iH" % (w.getnframes()* w.getnchannels()), astr)
#     a = [float(val) / pow(2, 15) for val in a]
#     return a

# # read the wav file specified as first command line arg
# signal = wav_to_floats(audio)
# print("read "+str(len(signal))+" frames")
# print("in the range "+str(min(signal))+" to "+str(min(signal)))
    
audio.close()
