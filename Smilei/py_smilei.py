from common_modules import *
import happi

#-----------------------------------------------#
# class object to hold data
#-----------------------------------------------#
class data_out():
  pass

#-----------------------------------------------#
# open simulation 
#-----------------------------------------------#
def open_sim(path, verbose=False):
  path = join(home, path)
  sim = happi.Open(results_path=path, verbose=verbose)
  
  sim.dt = sim._timestep
  sim.dim = sim._ndim_fields
  sim.cells = sim._ncels
  sim.dr = sim._cell_length
  sim.res = 1./sim.dr
  sim.box = sim.cells*sim.dr
  sim.path = path
  if verbose: 
    print('dim, box, cells, dr, dt = ', 
          sim.dim, sim.box, sim.cells, sim.dr, sim.dt)
    print('loaded simulation at: %s'%path)
  return (sim)

#-----------------------------------------------#
# get 'field' data; only appplies 2D or 1D data arrays
#-----------------------------------------------#
def get_field(sim, fn, field, t, norm=twopi):
  obj = sim.Field(fn, field=field)
  out = data_out()
  out.field = field
  
  # use closest time step
  timesteps = obj.getTimesteps()
  ts = timesteps[np.argmin(np.abs(timesteps*sim.dt-t))]
  out.timesteps = timesteps
  out.ts = ts
  out.t = ts*sim.dt

  # extract data array
  d = obj._getDataAtTime(ts) 
  if len(d.shape) == 2: # transpose if 2D
    d = d.transpose() 
  if 'elec' in field: # normalization
    d *= -1.0  
  if 'Rho' in field: 
    d /= norm**2.
  if 'E' in field or 'B' in field: 
    d /= norm
  out.d = d
    
  # axes data
  ax=['x', 'y', 'z']
  for i in range(sim.dim):
    setattr(out, ax[i], obj.getAxis(ax[i]))

  return (out)


#-----------------------------------------------#
# get 'probe' data
#-----------------------------------------------#
def get_probe(sim, fn, field, t, norm=twopi):
  obj = sim.Probe(fn, field=field)
  out = data_out()
  out.field = field
  
  # use closest time step
  timesteps = obj.getTimesteps()
  ts = timesteps[np.argmin(np.abs(timesteps*sim.dt-t))]
  out.timesteps = timesteps
  out.ts = ts
  out.t = ts*sim.dt

  # extract data array
  d = obj._getDataAtTime(ts) 
  if len(d.shape) == 2: # transpose if 2D
    d = d.transpose() 
  if 'elec' in field: # normalization
    d *= -1.0  
  if 'Rho' in field: 
    d /= norm**2.
  if 'E' in field or 'B' in field: 
    d /= norm
  out.d = d
    
  # axes data
  ax=['x', 'y', 'z']
  for i in range(sim.dim):
    setattr(out, ax[i], obj.getAxis(ax[i]))

  return (out)


#-----------------------------------------------#
# get 'particlebinning' data
#-----------------------------------------------#
def get_partbin(sim, fn, ax, t):
  obj = sim.ParticleBinning(fn)
  out = data_out()
  
  # use closest time step
  timesteps = obj.getTimesteps()
  ts = timesteps[np.argmin(np.abs(timesteps*sim.dt-t))]
  out.timesteps = timesteps
  out.ts = ts
  out.t = ts*sim.dt
  
  # extract data array
  d = obj._getDataAtTime(ts) 
  if len(d.shape) == 2: # transpose if 2D
    d = d.transpose() 
  out.d = d
  
  # axes data
  for s in ax: 
    setattr(out, s, obj.getAxis(s))
    
  return (out)

#-----------------------------------------------#
# get 'screen' data
#-----------------------------------------------#
def get_screen(sim, fn, ax, t):
  obj = sim.Screen(fn) 
  out = data_out()
  
  # use closest time step
  timesteps = obj.getTimesteps()
  ts = timesteps[np.argmin(np.abs(timesteps*sim.dt-t))]
  out.timesteps = timesteps
  out.ts = ts
  out.t = ts*sim.dt
  
  # extract data array
  d = obj._getDataAtTime(ts)
  if len(d.shape) == 2: # transpose if 2D
    d = d.transpose() 
  out.d = d
  
  # axes data
  for s in ax: 
    setattr(out, s, obj.getAxis(s))
    
  return(out)


#***************************************************#
#************* more specific functions *************#  
#***************************************************#


#-----------------------------------------------#
# get 'field' data; only appplies 2D or 1D data arrays
#-----------------------------------------------#
# beam charge detected on screen
def get_screen_charge(sim, fn, ax, t, with_ab=True, verbose=True):
  a = get_screen(sim, fn, ax, t)
  x = getattr(a, ax[0]); dx=x[1]-x[0]
  if len(ax)==1:
    dxy =dx
  elif len(ax)==2:
    y = getattr(a, ax[1]); dy=y[1]-y[0]
    dxy = dx*dy
  else:
    raise Exception('currently only accepts two axes')
    
  if with_ab: # either 'a' or 'b' is in the axes
    charge_detected = np.sum(a.d)*dxy
  else:
    vol = sim.box[0]
    if sim.dim>1: vol *= sim.box[1]
    if sim.dim>2: vol *= sim.box[2]
    charge_detected = np.sum(a.d)*dxy*vol
  
  if verbose:
    print ('charge_detected: ', charge_detected)
  
  return (charge_detected)

#-----------------------------------------------#
# get effective density (n/gamma) evolution
# from partbin data
#-----------------------------------------------#
def get_effDenEvol_partbin(sim, fn=[0,1], tl=[0,151,2]):  
  out = data_out()
  
  obj1 = sim.ParticleBinning(fn[0]) # density
  obj2 = sim.ParticleBinning(fn[1]) # energy density
  y = obj1.getAxis("y")
  idx = int(len(y)/2)  # idx of central y lineout
  timesteps = obj1.getTimesteps()
  
  # time series
  times = np.arange(tl[0], tl[1], tl[2])
  Nt = len(times)
  
  # create arrays 
  den_max = np.zeros(Nt)
  den_gamma_max = np.zeros(Nt)
  
  # number of cells to average over y
  ave = sim.res[1]/2.
  print('averaing over %s cells'%(2*ave))
  
  # length to average over x 
  ave_x = 100 # over 1/100 laser wavelength

  # extract data
  for i in range(Nt):
    showProgressBar(Nt-1, i)
    
    if (times[i]>=60 and sim.dim==3): #or (times[i]>=70 and sim.dim==2):
      ave_x = 2
    
    ts = timesteps[np.argmin(np.abs(timesteps*sim.dt-times[i]))] 
    den      = obj1._getDataAtTime(ts).T/twopi2
    den_ekin = obj2._getDataAtTime(ts).T/twopi2

    # normal density (max)
#     den_max[i] = den[idx,:].max()
#     den_max[i] = smooth1d(den[idx,:], 2).max() # smooth in x
    den_max[i] = smooth1d(np.mean(den[idx-int(ave):idx+int(ave),:], axis=0),
                          int(sim.res[0]/ave_x)).max() # average in y
    # relativistically corrected density (max)
    gamma = den_ekin/(den + 1e-5) + 1.0
    den_gamma = den/gamma
#     den_gamma_max[i] = den_gamma[idx,:].max()
#     den_gamma_max[i] = smooth1d(den_gamma[idx,:], 2).max()
    den_gamma_max[i] = smooth1d(np.mean(den_gamma[idx-int(ave):idx+int(ave),:], axis=0), 
                                int(sim.res[0]/ave_x)).max()
  
  out.t = times
  out.den_max = den_max
  out.den_gamma_max = den_gamma_max

  return (out)