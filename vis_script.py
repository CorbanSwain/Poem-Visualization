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

def wav_to_float(wavefile):
    audio = wave.open(wavefile, 'r')
    output = []
    for iFrame in range(audio.getnframes()):
        wave_data = audio.readframes(1)
        data = struct.unpack('<i', wave_data)
        output.append(data[0])
    max_val = max(output)
    output = [v / max_val for v in output]
    audio.close()
    return output

amplitude = wav_to_float(clipfile)
