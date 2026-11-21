import sys
sys.version
import time
import os 
import happi
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.ticker import AutoMinorLocator
import h5py
import h5py as h5
from scipy import signal
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt 
import numpy as np
from scipy import signal
import matplotlib.gridspec as gridspec
import shutil
from os.path import join, exists
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

e=1.6e-19 # C
n0=4.0e6 # per meter cude
mi=1.6e-27 #kg
me=9.1e-31 #kg
epsilon=8.85e-12
B=156e-9 #Tesla
c=3.0e8 #m/s

lmin, lmax, zmax = 0.5588391918010813, 0.6411608081989186, 0.2195243103942331
M_sgn = -1; coeff = 0.41108718197029054; Re_cwr = 583.0789299377868

#lmin, lmax, zmax = 0.5388391918010813, 0.6611608081989186, 0.3195243103942331
#box=[72,384]

box=[48,256]; twopi = 2*np.pi; wce=e*B/me; wpewce=np.sqrt(n0*me/(epsilon*B**2)); de=c/wce
def disp(w,th):
   k=np.sqrt(w**2+w/(abs(np.cos(th*np.pi/180))-w)*wpewce**2)
   return k
    
def datat(f,ts,t):
    ft=f._getDataAtTime(ts[t])
    return ft

def pltmd(axt,c):
  axt.minorticks_on()
  axt.tick_params(axis='both', which='both', color=c, direction='in', top=True, right=True)

def rect(ax,x0,y0,w,h):
    rect = patches.Rectangle((x0, y0), w, h, color='gray', alpha=0.5)
    ax.add_patch(rect)

def cbar(img, ax, position, orientation,label=None,ticks=None):
    cbax = ax.inset_axes(position)
    cb=plt.colorbar(img, cax=cbax,orientation=orientation)
    cb.ax.xaxis.set_minor_locator(AutoMinorLocator())
    cb.ax.xaxis.set_minor_locator(AutoMinorLocator())
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.tick_params(direction='in',which='both')

    if ticks is not None:
        cb.set_ticks(ticks)
    if label is not None:
        cb.set_label(label)

def axex(ax):
    ax.set_xlim(0,48);
    ax.set_ylim(120,250)
    ax.set_xlabel(r'$x \Omega_e/c$')
    ax.set_xticks([5,15,25,35,45])
    pltmd(ax,'k')
    ax.axhline(128,ls='--',color='k')

def plot_fld_lines_sim(ax,l_list):
  box=[48,256]
  lmin, lmax, zmax = 0.5588391918010813, 0.6411608081989186, 0.2195243103942331
  #lmin, lmax, zmax = 0.5388391918010813, 0.6611608081989186, 0.3195243103942331
  theta = np.linspace(0, np.pi, 200)
  for l in l_list:
    x = (l*np.sin(theta)**2.-lmin)*box[0]/(lmax-lmin)
    y = l*np.sin(theta)*np.cos(theta)*0.5*box[1]/zmax+box[1]/2.
    ax.plot(x, y, 'k', linewidth=0.5)

def probe_marks(ax,l,A):
    theta_probe = np.array(A)*np.pi/180. 
    for i in range(len(theta_probe)):
        x_probe = (l*np.sin(theta_probe[i])**2.-lmin)*box[0]/(lmax-lmin)
        y_probe = l*np.sin(theta_probe[i])*np.cos(theta_probe[i])*0.5*box[1]/zmax+box[1]/2.
        ax.plot(x_probe, y_probe, 'kx',markersize=4)

def datap(tt, fe, df):
    Bx = fe['data/{:010d}/particles/elec/B/x'.format(df*tt)][:]
    By = fe['data/{:010d}/particles/elec/B/y'.format(df*tt)][:]
    Bz = fe['data/{:010d}/particles/elec/B/z'.format(df*tt)][:]
    px = fe['data/{:010d}/particles/elec/momentum/x'.format(df*tt)][:]
    py = fe['data/{:010d}/particles/elec/momentum/y'.format(df*tt)][:]
    pz = fe['data/{:010d}/particles/elec/momentum/z'.format(df*tt)][:]
    x = fe['data/{:010d}/particles/elec/position/x'.format(df*tt)][:]
    y = fe['data/{:010d}/particles/elec/position/y'.format(df*tt)][:]
    return Bx.astype(float), By.astype(float), Bz.astype(float), \
           px.astype(float), py.astype(float), pz.astype(float), \
           x.astype(float), y.astype(float)

def mag_b(x, y):
    x1 = x / Re_cwr + lmin
    y1 = y / Re_cwr - zmax
    deno = (x1**2 + y1**2)**2.0
    deno[deno == 0] = 1e-30
    B0x = M_sgn * 2 * x1 * y1 / deno * coeff
    B0y = M_sgn * (y1**2 - x1**2) / deno * coeff
    return B0x, B0y

def theta_com(B0x, B0y, B0z, Bx, By, Bz, px, py, pz):
    B0 = np.stack((B0x, B0y, B0z), axis=1)
    B = np.stack((Bx, By, Bz), axis=1)
    gamma = np.sqrt(1.0 + px**2 + py**2 + pz**2)
    p = np.stack((px, py, pz), axis=1) / gamma[:, None]
    # Unit vector
    B0_norm = np.linalg.norm(B0, axis=1)
    B0_hat = B0 / (B0_norm[:, None] + 1e-12)
    # Parallel & perpendicular
    p_para_mag = np.sum(p * B0_hat, axis=1)
    p_para = p_para_mag[:, None] * B0_hat
    p_perp = p - p_para
    p_perp_mag = np.linalg.norm(p_perp, axis=1)
    # Field decomposition
    B_para = np.sum(B * B0_hat, axis=1)[:, None] * B0_hat
    B_perp = B - B_para
    # Normalize (important)
    p_perp_hat = p_perp / (p_perp_mag[:, None] + 1e-12)
    B_perp_hat = B_perp / (np.linalg.norm(B_perp, axis=1)[:, None] + 1e-12)
    dot = np.sum(p_perp_hat * B_perp_hat, axis=1)
    cross = np.cross(p_perp_hat, B_perp_hat)
    dot_cross_B0 = np.sum(cross * B0_hat, axis=1)
    theta = np.arctan2(dot_cross_B0, dot)
    return theta, p_para_mag, p_perp_mag, gamma

# position at the dipole line with a given lambda 
def dip_pos(l, lda):
    th = (np.pi/180)*(90-lda);
    r = l*np.sin(th)
    xd = (r*np.sin(th)-lmin)*(48/(lmax-lmin)) 
    yd = r*np.cos(th)*128/zmax + 256/2
    return xd, yd

# line plot to measure the slope
def lfit(ax,xi,xf,slp,intc,color,data):
    sft=int(10*2.3);xii=int(xi*2.3);xff=int(xf*2.3);xs=np.linspace((xii+sft),(xff+sft),len(data))/2.3;
    ys=slp*xs+intc
    ax.plot(xs/100,ys,color=color,linewidth=1,ls='--')

# phase space islands 
def island(ax,s, wtr, k , zeta1, vc,color):
    zeta=np.linspace(0,2*np.pi,100000)
    with np.errstate(invalid='ignore'):
        theta = wtr*np.sqrt(2*(np.cos(zeta1) - np.cos(zeta) + s*(zeta-zeta1)))/k
    ax.plot(zeta, vc+theta,color=color)
    ax.plot(zeta, vc-theta,color=color)