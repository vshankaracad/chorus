
"""
-----------------------------------------------------------------------
BEGINNING OF THE USER NAMELIST
"""
import numpy as np
from scipy.special import erf
import math

twopi = 2*np.pi
twopi2 = (2*np.pi)**2.
# from scipy.constants import m_e, epsilon_0
m_e = 9.1093837015e-31 
epsilon_0 = 8.8541878128e-12

# ******* convert real paramters to simulation units *****
L0 = 0.6
Lmin, Lmax = L0-0.04, L0+0.04
zmax =0.21  # zmin=-zmax

Be = 0.31e5  # magnetic field at equator and L=Re
# B0 = Be/L0**3 # background B field, nT (notice this is for 3D model)
# B0 = Be/(3*L0**2)  # this is for 2D model
B0 = 156   # magnetic field at the outer boundary of the equator. 
  
n0 = 4.   # plasma density (total), per cc
de = 5.32e5/n0**0.5 # electron inertial length c/wpe, units of cm

n0b = n0*0.994  # plasma density (background), per cc
n0i = n0*0.006  # plasma density (injected), per cc
Re   = 6.378e8/de # Earth's radius, units of de
lenx = (Lmax-Lmin)*Re # plasma length in x, units of de (c/wpe)
leny = 2*zmax*Re # plasma length in y, units of de (c/wpe)

ppc_bk = 400 # particle number per cell (for both ion and electron species)
ppc_hot = 2000

inj_vel = 0.0 # injection speed of plasma, units of c

t_par = 10. # parallel electron temperature, keV
t_per = 60. # perpendicular electron temperature, keV
t_bk = 0.05 # background plasma temperature, keV

# normalize temperature to m_e * c^2
t_par /= 511.
t_per /= 511.
t_bk  /= 511.

nr = epsilon_0*(B0*1e-9)**2/m_e*1e-6  # reference density, per cc
n0_nr = n0/nr # n0 normalized to reference density
n0b_nr = n0b/nr
n0i_nr = n0i/nr
  
lenx_cwr = lenx/n0_nr**0.5  # plasma length normalized to c/wr (i.e., c/wce)
leny_cwr = leny/n0_nr**0.5
Re_cwr   = Re/n0_nr**0.5

lenx_di = lenx/1836**0.5
leny_di = leny/1836**0.5
  
# round to 2 decimals
n0_nr = round(n0_nr, 2)
n0b_nr = round(n0b_nr, 2)
n0i_nr = round(n0i_nr, 2)
lenx_cwr = round(lenx_cwr, 2)
leny_cwr = round(leny_cwr, 2)
lenx_di = round(lenx_di, 2)
leny_di = round(leny_di, 2)


#************************************************

# ---------- basic setup -------------------
box = [lenx_cwr, leny_cwr]  # in units of c/wr 
res = [32, 32]   # resolution: cell number per c/wr
number_of_patches = [256,1024]

tmax = 1000.  # max sim. time, in units of 1/wr = 1/wce

if smilei_mpi_rank==0:
    print('**************************')
    print('B0       = %s nT'%B0)
    print('n0_nr    = %s; n0b_nr=%s; n0i_nr=%s'%(n0_nr, n0b_nr, n0i_nr))
    print('de       = %s km'%round(de*1e-5,2) )
    print('wpe/wce  = ', round(n0_nr**0.5,2) )
    print('wpi/wce  = ', round((n0_nr/1836)**0.5,2) )
    print('lenx_cwr = %s; leny_cwr = %s'%(lenx_cwr, leny_cwr) )
    print('lenx_di  = %s; leny_di = %s'%(lenx_di, leny_di))
    print('before resizing: box = [%s, %s]'%(box[0],box[1]) )
    print('                 Lmin, Lmax, zmax = [%s, %s, %s]'%(Lmin, Lmax, zmax) )
  
#********** DO NOT CHANGE **********************
dr =[ 1./i for i in res ]
dt_CFL = 1./np.sqrt(sum(i**2 for i in res))  # Courant condition
rest = (1./dt_CFL)/0.98
dt =1./rest
# timesteps to finish one laser cycle
Nt = int(rest)
# sim. dimension
dim = len(box)
# adjust box size (cells) to be divided by number_of_pathces
for i in range(dim):
	fac = float(box[i])*res[i]/number_of_patches[i]
	if fac!=fac//1: fac = int(fac)+1
	box[i] = float(fac)*number_of_patches[i]/res[i]

# total number of cells
cells = list( np.array(box)*np.array(res) )
#************************************************

# since the box is resized, the L numbers should be changed accordingly 
Lmin = L0-box[0]/Re_cwr*0.5
Lmax = L0+box[0]/Re_cwr*0.5
zmax = box[1]/Re_cwr*0.5

patch_size = [cells[0]/number_of_patches[0], cells[1]/number_of_patches[1]]
if smilei_mpi_rank==0:
    print('after resizing:  box = [%s, %s]'%(box[0],box[1]) )
    print('                 Lmin, Lmax, zmax = [%s, %s, %s]'%(Lmin, Lmax, zmax) )
    print('cells = ', cells)
    print('cells per patch: ', patch_size[0], patch_size[1])
    print('**************************')

dump_freq = 10*Nt
flush_freq = 50*Nt

# ------------ global settings --------------------
Main(
  geometry = "2Dcartesian",
  interpolation_order = 2,
  grid_length  = box,
  cell_length = dr,
  # each must be power of 2; total must be greater than MPI process number
  number_of_patches = number_of_patches,
  patch_arrangement = "hilbertian",
  # patch_arrangement = "linearized_ZYX",
  timestep = dt,
  simulation_time = tmax,
  EM_boundary_conditions = [
      ['silver-muller'],
    # ["periodic"],
  ],
  solve_poisson = False,  # False to have neutralized background
  # reference_angular_frequency_SI = 3.e8/0.4e-6, #speed of light / laser wavelegth
  print_every = Nt,
  # random_seed = smilei_mpi_rank
)


# -------------- filtering ---------------
CurrentFilter(
    model = "binomial",
    passes = [10],
    kernelFIR = [0.25,0.5,0.25],
)

M_sgn = -1 # sign of earth's magnetic momentum

coeff = Lmax**2.  # B field is normalzed to the value at the outer boundary of the equator

theta=-np.pi/9.0

k0=-1414 # in Re_wcr

kx=k0*math.sin(theta)
ky=k0*math.cos(theta)

A=1.0e-6
xp=0.62 # in Re_wcr
xl=0.005 # in Re_wcr
yp=0.01 # in Re_wcr
yl=0.01 # in Re_wcr
w0=0.256 # calculated from dispersion relation for k0=1414/Re_cwr or k0=2.425

xl2 = 1.0/xl**2
yl2 = 1.0/yl**2
ky_yl2 = ky*yl/2
sqrt_pi_yl2 = np.sqrt(np.pi)*(yl/2)*np.exp(1j*ky*yp-(ky*yl/2)**2)
za = (-zmax-yp)/yl-1j*(ky*yl/2)  # Precompute za for Jz
erf_za = erf((-zmax-yp)/yl-1j*(ky*yl/2))  # Precompute erf(za)
lty=1.5*yl
ltyd=1.0/(0.5*yl)
dltkx = (kx**2+2.0*xl2)
kxxl2=4.0*kx*xl2
xl4=4.0*xl2*xl2

def Jx_profile(x, y, t):
    x1 = x/Re_cwr + Lmin
    y1 = y/Re_cwr - zmax
    if t>400:
        return 0
    else: 
     if (abs(x1 - xp) > 1.5*xl or abs(y1 - yp) >1.5*yl):
        return 0.0
     else:
       phase = w0*t + ky*y1 + kx*x1
       gaussian = math.exp(-xl2*(x1-xp)**2 - yl2*(y1-yp)**2)
       bx1 = A * math.sin(phase) * gaussian
       bz1 = A * math.cos(phase) * gaussian
       Jx = -ky*bx1 - 2*((y1-yp)*yl2)*bz1
       return Jx

def Jy_profile(x, y, t):
    x1 = x/Re_cwr + Lmin
    y1 = y/Re_cwr - zmax
    if t>400:
       return 0
    else:
     if (abs(x1 - xp) > 1.5*xl or abs(y1 - yp) > 1.5*yl):
       return 0.0 
     else:
       phase = w0*t + ky*y1 + kx*x1
       gaussian = math.exp(-xl2*(x1-xp)**2 - yl2*(y1-yp)**2)
       bx1 = A * math.sin(phase) * gaussian
       bz1 = A * math.cos(phase) * gaussian
       Jy = kx*bx1+2*((x1-xp)*xl2)*bz1
       return Jy
def Jz_profile(x, y, t):
    x1 = x/Re_cwr + Lmin
    y1 = y/Re_cwr - zmax
    if t>400:
        return 0
    else:
     if (abs(x1 - xp) > 1.5*xl or abs(y1 - yp) > 1.5*yl):
       return 0.0
     else:
       phase = w0*t + ky*y1 + kx*x1
       gaussian = math.exp(-xl2*(x1-xp)**2 - yl2*(y1-yp)**2)
       bx1 = A * math.sin(phase) * gaussian
       bz1 = A * math.cos(phase) * gaussian
       #yy = (y1-yp)/yl-1j*ky_yl2
       #pref=np.exp(1j*(w0*t+kx*x1))*sqrt_pi_yl2*(erf(yy)-erf_za)
       Jz = -ky*bz1+2.0*((y1-yp)*yl2)*bx1#+A*np.exp(-xl2*(x1-xp)**2)*((dltkx-(x1-xp)**2*xl4)*np.imag(pref)-kxxl2*(x1-xp)*np.real(pref))
       return Jz

Antenna(
    field="Jx",
    space_time_profile=Jx_profile
)

Antenna(
    field="Jy",
    space_time_profile=Jy_profile
)

Antenna(
    field="Jz",
    space_time_profile=Jz_profile
)

def B0x_profile(x,y):
    x1 = x/Re_cwr + Lmin 
    y1 = y/Re_cwr - zmax
    deno = (x1**2+y1**2)**2.
    B0x = M_sgn * 2*x1*y1/deno * coeff 
    return B0x

def B0y_profile(x,y):
    x1 = x/Re_cwr + Lmin 
    y1 = y/Re_cwr - zmax
    deno = (x1**2+y1**2)**2.
    B0y = M_sgn * (y1**2.-x1**2.)/deno* coeff
    return B0y

ExternalField(
    field = "Bx",
    profile = B0x_profile, # unit Br = me * wr/e = me * wce /e
)
ExternalField(
    field = "By",
    profile = B0y_profile, # unit Br = me * wr/e = me * wce /e
)

#--------------------- plasma -------------

# background plamsa
def den_profile_bk(x,y):
    return n0b_nr
    
Species(name = 'proton_bk', position_initialization = 'random', momentum_initialization = 'cold', temperature = [t_bk, t_bk, t_bk], particles_per_cell = ppc_bk, mass = 1836.0, charge = 1.0, number_density = den_profile_bk, boundary_conditions = [["remove"]], time_frozen=0,)

Species(name = 'elec_bk', position_initialization = 'random', momentum_initialization = 'maxwell-juettner', temperature = [t_bk, t_bk, t_bk], particles_per_cell = ppc_bk, mass = 1.0, charge = -1.0, number_density = den_profile_bk, boundary_conditions = [["reflective"]],)


def den_profile_hot(x, y):
    x1 = x/Re_cwr + Lmin 
    y1 = y/Re_cwr - zmax
    R = np.sqrt(x1**2+y1**2)
    latitude = np.arctan2(y1, x1)
    L = R/np.cos(latitude)
    Beq = Be/(3*L**2.)

    deno = (x1**2+y1**2)**2.
    B0x = M_sgn * (Be/3.) * 2*x1*y1/deno
    B0y = M_sgn * (Be/3.) * (y1**2.-x1**2.)/deno
    B = np.sqrt(B0x**2.+B0y**2.)

    zeta = 1+(t_per/t_par-1.)*(1-Beq/B)
    den = n0i_nr/zeta
    return den

def temp_perp_hot(x,y):
    x1 = x/Re_cwr + Lmin 
    y1 = y/Re_cwr - zmax
    R = np.sqrt(x1**2+y1**2)
    latitude = np.arctan2(y1, x1)
    L = R/np.cos(latitude)
    Beq = Be/(3*L**2.)

    deno = (x1**2+y1**2)**2.
    B0x = M_sgn * (Be/3.) * 2*x1*y1/deno
    B0y = M_sgn * (Be/3.) * (y1**2.-x1**2.)/deno
    B = np.sqrt(B0x**2.+B0y**2.)

    zeta = 1+(t_per/t_par-1.)*(1-Beq/B)

    temp = t_per/zeta
    return temp

Species(name = 'proton', position_initialization = 'random', momentum_initialization = 'maxwell-juettner', temperature = [t_bk, t_bk, t_bk], mean_velocity = [inj_vel,0.,0.], particles_per_cell = ppc_hot, mass = 1836.0, charge = 1.0, number_density = den_profile_hot, boundary_conditions = [["remove"]], time_frozen=0,)

Species(name = 'elec', position_initialization = 'random', momentum_initialization = 'maxwell-juettner', temperature = [temp_perp_hot, t_par, temp_perp_hot], mean_velocity = [inj_vel,0.,0.], particles_per_cell = ppc_hot, mass = 1.0, charge = -1.0, number_density = den_profile_hot, boundary_conditions = [["reflective"]],) # notice y is now the parallel direction (relative to B0)


#------------ diagnostics ----------------------
  
#---------- Scalar ---------------
DiagScalar(
  every = Nt,
  precision = 8,
)

#---------- Probe ---------------

l=0.605 # probe 0-4 
# theta_probe = np.array([78.,82.,86.])*np.pi/180. 
theta_probe = np.array([75., 78., 80., 82., 85.])*np.pi/180.
for i in range(5):
    x_probe = (l*np.sin(theta_probe[i])**2.-Lmin)*box[0]/(Lmax-Lmin)
    y_probe = l*np.sin(theta_probe[i])*np.cos(theta_probe[i])*0.5*box[1]/zmax+box[1]/2.
    DiagProbe(
      every = [0, 20],
      flush_every = flush_freq,
      origin   = [x_probe, y_probe],
      fields   = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
      # time_integral = True,
    )


l=0.61 # probe 5-9, 
# theta_probe = np.array([78.,82.,86.])*np.pi/180. 
theta_probe = np.array([75., 78., 80., 82., 85.])*np.pi/180.
for i in range(5):
    x_probe = (l*np.sin(theta_probe[i])**2.-Lmin)*box[0]/(Lmax-Lmin)
    y_probe = l*np.sin(theta_probe[i])*np.cos(theta_probe[i])*0.5*box[1]/zmax+box[1]/2.
    DiagProbe(
      every = [0, 20],
      flush_every = flush_freq,
      origin   = [x_probe, y_probe],
      fields   = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
      # time_integral = True,
    )

l=0.615 # probe 10-14
# theta_probe = np.array([78.,82.,86.])*np.pi/180. 
theta_probe = np.array([75., 78., 80., 82., 85.])*np.pi/180.
for i in range(5):
    x_probe = (l*np.sin(theta_probe[i])**2.-Lmin)*box[0]/(Lmax-Lmin)
    y_probe = l*np.sin(theta_probe[i])*np.cos(theta_probe[i])*0.5*box[1]/zmax+box[1]/2.
    DiagProbe(
      # name = "LeftBndProbe",
      every = [0, 20],
      flush_every = flush_freq,
      origin   = [x_probe, y_probe],
      fields   = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
      # time_integral = True,
    )


l=0.62 # probe 15-19
# theta_probe = np.array([78.,82.,86.])*np.pi/180. 
theta_probe = np.array([75., 78., 80., 82., 85.])*np.pi/180. 
for i in range(5):
    x_probe = (l*np.sin(theta_probe[i])**2.-Lmin)*box[0]/(Lmax-Lmin)
    y_probe = l*np.sin(theta_probe[i])*np.cos(theta_probe[i])*0.5*box[1]/zmax+box[1]/2.
    DiagProbe(
      # name = "LeftBndProbe",
      every = [0, 20],
      flush_every = flush_freq,
      origin   = [x_probe, y_probe],
      fields   = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
      # time_integral = True,
    )

l=0.625 # probe  20-24
# theta_probe = np.array([78.,82.,86.])*np.pi/180. 
theta_probe = np.array([75., 78., 80., 82., 85.])*np.pi/180.
for i in range(5):
    x_probe = (l*np.sin(theta_probe[i])**2.-Lmin)*box[0]/(Lmax-Lmin)
    y_probe = l*np.sin(theta_probe[i])*np.cos(theta_probe[i])*0.5*box[1]/zmax+box[1]/2.
    DiagProbe(
      # name = "LeftBndProbe",
      every = [0, 20],
      flush_every = flush_freq,
      origin   = [x_probe, y_probe],
      fields   = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
      # time_integral = True,
    )

l=0.63 # probe  25-29
# theta_probe = np.array([78.,82.,86.])*np.pi/180. 
theta_probe = np.array([75., 78., 80., 82., 85.])*np.pi/180.
for i in range(5):
    x_probe = (l*np.sin(theta_probe[i])**2.-Lmin)*box[0]/(Lmax-Lmin)
    y_probe = l*np.sin(theta_probe[i])*np.cos(theta_probe[i])*0.5*box[1]/zmax+box[1]/2.
    DiagProbe(
      # name = "LeftBndProbe",
      every = [0, 20],
      flush_every = flush_freq,
      origin   = [x_probe, y_probe],
      fields   = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
      # time_integral = True,
    )

l=0.64 # probe  30-34
# theta_probe = np.array([78.,82.,86.])*np.pi/180. 
theta_probe = np.array([75., 78., 80., 82., 85.])*np.pi/180.
for i in range(5):
    x_probe = (l*np.sin(theta_probe[i])**2.-Lmin)*box[0]/(Lmax-Lmin)
    y_probe = l*np.sin(theta_probe[i])*np.cos(theta_probe[i])*0.5*box[1]/zmax+box[1]/2.
    DiagProbe(
      # name = "LeftBndProbe",
      every = [0, 20],
      flush_every = flush_freq,
      origin   = [x_probe, y_probe],
      fields   = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
      # time_integral = True,
    )



#---------Field --------------

from numpy import s_

fields_list = ["Ex", 'Ey', 'Ez', 'Bx', 'By', 'Bz', "Rho_elec"]
for fld in fields_list:
    DiagFields(
    every = [0, dump_freq],
    flush_every = flush_freq,
    subgrid = s_[::4, ::4],
    fields = [fld],
    )


# ---------ParticleBinning --------------

Nx = int(box[0]) # sampling points along x
Ny = int(box[1])


# diagnose the temperature of injected electrons
#0: charge
DiagParticleBinning(deposited_quantity="weight", every=[0, dump_freq], flush_every = flush_freq, species=['elec'], axes=[ ["x", 0, box[0], Nx], ["y", 0, box[1], Ny] ],)
#1: xx pressure
DiagParticleBinning(deposited_quantity="weight_vx_px", every=[0, dump_freq], flush_every = flush_freq, species=['elec'], axes=[ ["x", 0, box[0], Nx], ["y", 0, box[1], Ny] ],)
#2: yy pressure
DiagParticleBinning(deposited_quantity="weight_vy_py", every=[0, dump_freq], flush_every = flush_freq, species=['elec'], axes=[ ["x", 0, box[0], Nx], ["y", 0, box[1], Ny] ],)
#3: zz pressure
DiagParticleBinning(deposited_quantity="weight_vz_pz", every=[0, dump_freq], flush_every = flush_freq, species=['elec'], axes=[ ["x", 0, box[0], Nx], ["y", 0, box[1], Ny] ],)

# diagnose the temperature of background electrons
#4: charge
DiagParticleBinning(deposited_quantity="weight", every=[0, dump_freq], flush_every = flush_freq, species=['elec_bk'], axes=[ ["x", 0, box[0], Nx], ["y", 0, box[1], Ny] ],)
#5: xx pressure
DiagParticleBinning(deposited_quantity="weight_vx_px", every=[0, dump_freq], flush_every = flush_freq, species=['elec_bk'], axes=[ ["x", 0, box[0], Nx], ["y", 0, box[1], Ny] ],)
#6: yy pressure
DiagParticleBinning(deposited_quantity="weight_vy_py", every=[0, dump_freq], flush_every = flush_freq, species=['elec_bk'], axes=[ ["x", 0, box[0], Nx], ["y", 0, box[1], Ny] ],)
#7: zz pressure
DiagParticleBinning(deposited_quantity="weight_vz_pz", every=[0, dump_freq], flush_every = flush_freq, species=['elec_bk'], axes=[ ["x", 0, box[0], Nx], ["y", 0, box[1], Ny] ],)
#----------------------------- track particles -------------------------------
Delta_L = Lmax - Lmin
dL = Delta_L / (32*box[0]) * 60.0  # 400 cells
def filter_func(particles):
    xprime = particles.x/Re_cwr+Lmin
    yprime = particles.y/Re_cwr-zmax
    theta = 0.5*np.pi - np.arctan2(yprime, xprime)
    L = (particles.x*Delta_L/box[0] + Lmin)/np.sin(theta)**2.0
      
    fil1=(np.abs(L-0.625)<=dL) * (np.abs(theta-(82/180)*np.pi)<=0.35/180*np.pi)\
        +(np.abs(L-0.625)<=dL) * (np.abs(theta-(85/180)*np.pi)<=0.35/180*np.pi)\
        +(np.abs(L-0.625)<=dL) * (np.abs(theta-(80/180)*np.pi)<=0.35/180*np.pi)\
        +(np.abs(L-0.625)<=dL) * (np.abs(theta-(78/180)*np.pi)<=0.35/180*np.pi)\
        +(np.abs(L-0.625)<=dL) * (np.abs(theta-(75/180)*np.pi)<=0.35/180*np.pi)


    fil=(np.abs(L-0.615)<=dL) * (np.abs(theta-(82/180)*np.pi)<=0.35/180*np.pi)\
        +(np.abs(L-0.615)<=dL) * (np.abs(theta-(85/180)*np.pi)<=0.35/180*np.pi)\
        +(np.abs(L-0.615)<=dL) * (np.abs(theta-(80/180)*np.pi)<=0.35/180*np.pi)\
        +(np.abs(L-0.615)<=dL) * (np.abs(theta-(78/180)*np.pi)<=0.35/180*np.pi)\
        +(np.abs(L-0.615)<=dL) * (np.abs(theta-(75/180)*np.pi)<=0.35/180*np.pi)
    return fil+fil1
DiagTrackParticles(species ="elec", every = [0, 5000*Nt, 10*Nt], flush_every=dump_freq, filter = filter_func, attributes = ["x", "y", "px", "py", "pz","Bx","By","Bz"] )


# --------------------------- checkpoints for restarts ------------------------
Checkpoints(
     # restart_dir = "dump1",
     # restart_number = 1, 
     #dump_minutes = 15.8*60,
     dump_step = int(0.2*Nt*(tmax-1)),
     exit_after_dump = False,
     keep_n_dumps = 2,
)


"""
    END OF THE USER NAMELIST
-----------------------------------------------------------------------
"""
