import os, sys
from os.path import join, exists

import time as systime
import psutil
import shutil

# import multiprocessing as mp
# import functools
# from multiprocessing import Process, Array

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
# from matplotlib.ticker import FormatStrFormatter

import numpy as np
from math import pi
twopi = pi*2.0
twopi2 = twopi**2.0

import h5py as h5

import scipy
from scipy import fft
# from scipy.integrate import odeint
# from scipy.signal import hilbert
# from scipy.stats import maxwell
# from scipy.signal import hilbert
# from scipy.optimize import curve_fit
# from scipy.special import jv

# import struct  # used for reading structured data, binary data?
# import copy   # used for copying colormap

# from sympy.solvers import solve
# from sympy import Symbol
# from sympy import re, im

# 
# %reset -f # clear variables in memory (keep modules, functions)

## HOME
home = os.getenv('HOME')

# customized colormaps
#from colormaps import *
#jbew = mpl.colors.LinearSegmentedColormap.from_list( #jet blue end white
#    'jbew', ['white', 'blue', 'cyan', 'lawngreen', 'gold', 'red'])
#jrew = mpl.colors.LinearSegmentedColormap.from_list( #jet red end white
#    'jrew', ['blue', 'cyan', 'lawngreen', 'gold', 'red', 'white'])
#jbew2 = mpl.colors.LinearSegmentedColormap.from_list( #jet blue end white
#'jbew', ['white', 'blue', 'red'])

# customize rcParams
def tweak_rcParams(fs=12, pyVersion=3, useHelvetica=False):
  import matplotlib.font_manager as font_manager
  
  if useHelvetica:
    # Use "Helvetica" by specifying absolute path to the font
    font_dirs = ['%s/scripts/pc/fonts'%home,]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    if pyVersion==2:
        # the following two lines will be deprecated in future versions
        font_list = font_manager.createFontList(font_files)
        font_manager.fontManager.ttflist.extend(font_list)
    elif pyVersion==3:
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file) 
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Helvetica'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.it'] = 'Helvetica:italic'

  # title, axes
  mpl.rcParams['figure.figsize'] = [4.5,4]
  mpl.rcParams['figure.titlesize'] = fs
  mpl.rcParams['axes.titlesize'] = fs
  mpl.rcParams['axes.linewidth'] = 0.5
  mpl.rcParams['axes.labelsize'] = fs
  mpl.rcParams['axes.labelweight'] = 'normal' #"light"
  mpl.rcParams['axes.unicode_minus'] = True  # minus sign

  # xtick 
  mpl.rcParams['xtick.labelsize'] = fs
  mpl.rcParams['xtick.top'] = True
  mpl.rcParams['xtick.major.size'] = 4
  mpl.rcParams['xtick.major.width'] = 0.5
  mpl.rcParams['xtick.direction'] = 'in'
  mpl.rcParams['xtick.minor.size'] = 2.5
  mpl.rcParams['xtick.minor.width'] = 0.5
  mpl.rcParams['xtick.minor.visible'] = True

  # ytick 
  mpl.rcParams['ytick.labelsize'] = fs
  mpl.rcParams['ytick.right'] = True
  mpl.rcParams['ytick.major.size'] = 4
  mpl.rcParams['ytick.major.width'] = 0.5
  mpl.rcParams['ytick.minor.size'] = 2.5
  mpl.rcParams['ytick.minor.width'] = 0.5
  mpl.rcParams['ytick.direction'] = 'in'
  mpl.rcParams['ytick.minor.visible'] = True

  # legend 
  mpl.rcParams['legend.fontsize'] = fs-2

  # image (imshow)
  mpl.rcParams["image.aspect"] = 'auto'
  mpl.rcParams["image.origin"] = 'lower'
  mpl.rcParams["image.cmap"] = 'viridis'
  mpl.rcParams["image.interpolation"] = 'gaussian'
  
  return

    
# check memory used
def mem_usage():
  process = psutil.Process(os.getpid())
  print ('Using: %.1f MB memory'%(process.memory_info().rss/1e6))
  return

# step function
# should use np.heaviside(x, 1.0) in future 
# https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html
def step(x): 
    return ( 1.0 * (x > 0) )

# used to auto-load variables of class definitions
def attrFromDict(d):
    self = d.pop('self')
    # for n, v in d.iteritems():  # python 2
    for name, value in d.items():  # python 3
        setattr(self, name, value)

# scientific notation for numbers in axis tick labels
def sci_format(limits=(-2, 3)):
    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((limits[0], limits[1]))
    return (fmt)


# 1d data smooth
def smooth1d(x, windowlen=10, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this:
            return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < windowlen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowlen < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    
    s = np.r_[x[windowlen - 1:0:-1], x, x[-2:-windowlen - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(windowlen, 'd')
    else:
        w = eval('np.' + window + '(windowlen)')
    y = np.convolve(w/w.sum(), s, mode='valid')
    # return y
    return ( y[int(windowlen/2-1):int(-windowlen/2)] )

  
## FFT of signal 'y' versus sampling points 'x'
def scipy_fft(x, y):
    F = scipy.fft.fft(y)
    dx = x[1] - x[0]
    N = len(y)
    f = np.linspace(0.0, 1.0/(2.0*dx), N//2)
    return ( f, 2.0/N*np.abs(F[0:N//2]) )

# return current directory
def pwd():
  return (os.getcwd())

# remove file
def rm(file):
  os.remove(file)

# forcefully and recursively remove directories
# equivalent to shell 'rm -rf'
def rmrf(path):
  shutil.rmtree(path)

# make directories
def mkdir(path):
  os.makedirs(path, exist_ok=True)

  
def flush():
  os.stdout.flush()

## make movies
def makeMovie(imageFolder='fields', videoFile='fields.mp4', fps=3, imageFormat='.png',codec='libx264',reverse=False):
    import moviepy.video.io.ImageSequenceClip
    imageFiles = [imageFolder+'/'+img 
                   for img in sorted(os.listdir(imageFolder), reverse=reverse) 
                       if img.endswith(imageFormat)]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(imageFiles, fps=fps)
    clip.write_videofile(imageFolder+'/'+videoFile, codec=codec)

    
# show progress bar by supplying max steps and current step
def showProgressBar(maxSteps, currStep):
    i = int(currStep/maxSteps*50)
    sys.stdout.write('\r')
    sys.stdout.write("[%-50s] %.2f%%" % ('='*i, currStep/maxSteps*100))
    sys.stdout.flush()
    if currStep==maxSteps: print('\n')

      
# for making plots with 3rd y axis
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return

# put colorbar on top
def cbar_top(ax, img, iax=[0.0, 1.02, 1, 0.1], label='', lcord=[0.5,2.3]):
  cbax = ax.inset_axes(iax)
  cb = plt.colorbar(img, cax=cbax, orientation="horizontal")
  cbax.set_xlabel(label)
  cbax.xaxis.set_label_position('top') 
  cbax.xaxis.tick_top()
#   cb.set_ticks([-4,1])
  cbax.xaxis.set_label_coords(lcord[0], lcord[1])
  return cbax, cb

# https://matplotlib.org/stable/api/axes_api.html#matplotlib-axes

## free memory of plots
# fig.clear(); plt.close(fig)

#************** using matplotlib.pyplot ******************
# plt.figure(figsize=[6,4],dpi=110)
# plt.subplot(1,2,1)
# plt.plot(x,y)
# plt.xlim(0,10)
# plt.tight_layout()

#************** using matplotlib.gridspec ******************
# fig = plt.figure(figsize=[10, 4], dpi=120)
# gs = grd.GridSpec(2, 4, height_ratios=[1,2.5], width_ratios=[2.5, 2.5,2.5,2.5], 
#                   hspace=0.1, wspace=0.1)
# ax1 = plt.subplot(gs[0])
          
#************** line plot ******************
# fig, axes = plt.subplots(1,1, figsize=[4, 3], dpi=110)
# ax1 = axes
# ax1.plot(x, d)
# ax1.set_xlim(0,)
# ax1.set_ylim(0,)
# ax1.set_xlabel(r'$x/\lambda_0$')
# ax1.set_ylabel(r'$y/\lambda_0$')
# plt.tight_layout()

# plt.show()

## super title
# fig.suptitle('title', x=0.5,y=0.85)

## set second y axis color
# ax2.yaxis.label.set_color(line.get_color())
# ax2.tick_params(axis='y', which='both', colors=line.get_color())

## scientific notation in ticker labels
# ax1.yaxis.set_major_formatter(sci_format())

## control decimal number for labelticks
# from matplotlib.ticker import FormatStrFormatter
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

## put legend on outside box
# ax1.legend(ncol=2, handlelength=0.5, fontsize=10, loc=2, frameon=False,
#                bbox_to_anchor=(0,1.4), bbox_transform=ax1.transAxes)

## dashed line control: length, space
# dashes=(10, 10)

## add text relative to axis
# ax1.text(-0.1, 1.05, "a", fontsize=16, transform = ax1.transAxes, weight='bold')
# ax1.text(0.1, 0.9, "a", fontsize=16, transform = ax1.transAxes, 
#          bbox=dict(edgecolor='w', facecolor='w', alpha=1), weight='bold', color='k')

## fill color between lines/boundaries
# line,=ax.plot(x,y)
# ax.fill_between(x_values, line2-1, line2+1, alpha=0.3, 
#                 color=line.get_color(), edgecolor='none')

#*************** imshow **********************
# fig, axes = plt.subplots(1,1, figsize=[4, 3], dpi=120)
# ax1 = axes
# img1 = ax1.imshow(d, cmap='jet', extent=[x[0], x[-1], y[0], y[-1]])
# plt.colorbar(img1, ax=ax1, pad=0)
# ax1.set_xlim(0,)
# ax1.set_ylim(0,)
# ax1.set_xlabel(r'$x/\lambda_0$')
# ax1.set_ylabel(r'$y/\lambda_0$')
# plt.tight_layout()
# plt.show()

## put colorbar on top
# cbax = ax1.inset_axes([0.0, 1.0, 1, 0.1])
# cb = plt.colorbar(img1, cax=cbax, orientation="horizontal")
# cbax.set_xlabel(r"$log(n/n_c)$")
# cbax.xaxis.set_label_position('top') 
# cbax.xaxis.tick_top()
# cb.set_ticks([-4,1])
# cbax.xaxis.set_label_coords(0.5, 1.5)

## set color to white for value smaller than the lower limit 
# # cmap = mpl.cm.get_cmap("jet").copy() # deprecated?
# cmap = copy.copy(mpl.cm.get_cmap("jet"))
# img1.cmap.set_under('w') 

#************** line 3D plot ******************
# from mpl_toolkits.mplot3d import Axes3D
# mpl.rcParams['legend.fontsize'] = 10
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)
# ax.plot(x, y, z, label='parametric curve')
# ax.legend()
