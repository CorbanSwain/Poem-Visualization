import numpy as np
import matplotlib as mpl
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import os
import wave
import struct
import sys

def wav_to_float(wavefile):
    audio = wave.open(wavefile, 'r')
    frame_counter = range(audio.getnframes())
    time = np.array(frame_counter) * (1 / audio.getframerate())
    amplitude = []
    for iFrame in frame_counter:
        wave_data = audio.readframes(1)
        n_bytes = len(wave_data)
        if n_bytes is 2: fmt = 'h'  
        elif n_bytes is 4: fmt = '<i'
        data = struct.unpack(fmt, wave_data)
        amplitude.append(data[0])
    max_val = max(amplitude)
    amplitude = [v / max_val for v in amplitude]
    audio.close()
    return (time, np.array(amplitude))

clipdir = '/Users/CorbanSwain/Google Drive'
def load_wav(clipname):
    try:
        save_data = np.load(clipname + '.npy')
        t, a = (save_data[:, 0], save_data[:, 1])
    except:
        clipfile = os.path.join(clipdir, clipname + '.wav') 
        print('Beginning File Processing...')
        t, a = np.array(wav_to_float(clipfile))
        np.save(clipname, np.column_stack((t, a)))
    print('Finished file import!')
    return (t, a)

def gaussian(x, sig):
    return np.exp(-np.power(x, 2.) / (2 * np.power(sig, 2.)))

def subsample_audio(t, a):
    numel = len(t)
    a_rect = np.multiply(a, a)
    
    sample_len = 1000
    step = np.floor(numel / sample_len).astype(np.int64)
    new_len = round(sample_len * step).astype(np.int64)
    selection = np.arange(0, new_len, step)
    a_sub = []

    window_fraction = 11 / 1000
    spread = 0.5
    baseline_shift = 0
    half_conv_window = round(numel * window_fraction / 2)
    conv_window = 2 * half_conv_window
    temp_x = np.linspace(-1, 1, conv_window)
    smooth_fxn = gaussian(temp_x, spread) + (baseline_shift / conv_window) 

    print('Beginning sparse convolution ...')
    for point in selection:
        window_sel = np.arange(point - half_conv_window,
                               point + half_conv_window)
        valid = np.where(np.logical_and(window_sel >= 0, window_sel < numel))
        a_sub.append(sum(a_rect[window_sel[valid]] * smooth_fxn[valid]))
    print('Finished sparse convolution.')

    a_sub = a_sub / max(a_sub)
    t_sub = t[selection]
    return (t_sub, a_sub)

def polar_convert(t, a, flatness):
    numel = len(t)
    theta = np.linspace(0, -2 * np.pi, numel) + (np.pi / 2)
    r = np.power((a / np.log(flatness)), 1.5) + 1
    theta = np.concatenate((theta, np.flip(theta, 0)))
    r = np.concatenate((r, np.ones((numel,))))
    return (r, theta)

def plot_linear(t, a, show=True):
    plt.figure(0)
    plt.plot(t, a)
    if show: plt.show(block=False)

def plot_2_linear(t1, a1, t2, a2, show=True):
    plt.figure(1)
    plt.plot(t1, a1)
    plt.plot(t2, a2)
    if show: plt.show(block=True)

def make_poly(r, theta):
    xs = np.multiply(r, np.cos(theta))
    ys = np.multiply(r, np.sin(theta))
    poly = Polygon(np.column_stack([xs, ys]), True)
    lim = max([-min(xs), max(xs), -min(ys), max(ys)])
    return (poly, lim)

def plot_polar(r, theta, show=True):
    poly, lim = make_poly(r, theta) 
    plt.figure(1)
    ax = plt.subplot(111, aspect='equal')
    ax.add_patch(poly)
    lim = lim * 1.1
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    if show: plt.show(block=True)

def plot_polar_2(r, theta, show=True):
    poly, lim = make_poly(r, theta) 
    fig = plt.figure(2, (20, 20))
    ax = plt.subplot(111, aspect='equal')
    lim = lim * 1.1
    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))

    plt.axis('off')
    plt.tight_layout()
    poly.set_facecolor('azure')
    fig.patch.set_facecolor('xkcd:navy')
    ax.set_facecolor('xkcd:navy')
    
    ax.add_patch(poly)
    if show: plt.show(block=True)

def visualize(clipname):
    t, a = load_wav(clipname)
    t_s, a_s = subsample_audio(t, a)

    flatness = 2
    r, th = polar_convert(t_s, a_s, flatness)
    plot_2_linear(t, a, t_s, a_s, False)
    plot_polar(r, th, False)
    plot_polar_2(r, th)
    
    
