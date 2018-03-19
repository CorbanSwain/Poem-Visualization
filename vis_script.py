import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    tstep = 1 / audio.getframerate()
    amplitude = []
    time = []
    frame_counter = range(audio.getnframes())
    for iFrame in frame_counter:
        wave_data = audio.readframes(1)
        data = struct.unpack('<i', wave_data)
        amplitude.append(data[0])
        if time:
            time.append(time[-1] + tstep)
        else:
            time.append(0)
    max_val = max(amplitude)
    amplitude = [v / max_val for v in amplitude]
    audio.close()
    return (time, amplitude)


t, a = np.array(wav_to_float(clipfile))

numel = len(t)
a_rect = np.multiply(a, a)
window_size = round(numel * 0.5)
smooth_fxn_1 = np.ones((window_size,)) / window_size
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
temp_x = np.linspace(-1, 1, window_size)
smooth_fxn_2 = gaussian(temp_x, 0, 0.2) + 0.25
smooth_fxn = smooth_fxn_2
a_smooth = np.convolve(a_rect, smooth_fxn, mode='same')
a_smooth = a_smooth / max(a_smooth)
new_numel = 1000
space = np.floor(numel / new_numel).astype(np.int64)
print(space)
print(type(space))
new_len = round(new_numel * space)
new_len = new_len.astype(np.int64)
print(new_len)
print(type(new_len))
# FIXME - use slicing
a_smooth_sub = a_smooth[np.arange(0, new_len, space)]
t_sub = t[np.arange(0, new_len, space)]



theta = np.linspace(0, 2 * np.pi, new_len)
r = (0.25 + a_smooth_sub) * 5

xs = np.multiply(r, np.cos(theta))
ys = np.multiply(r, np.sin(theta))

poly = mpl.patches.Polygon(np.column_stack([xs, ys]))
path = poly.get_path()

do_plot = False
if do_plot:
    plt.figure(0)
    plt.plot(t, a)
    plt.plot(t_sub, a_smooth_sub)
    plt.show()
    
    plt.figure(1)
    ax = plt.subplot(111, aspect='equal')
    ax.add_patch(poly)
    plt.show()
