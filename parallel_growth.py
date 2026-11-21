import numpy as np
def dispersion_w(w,th,wpewce,mass,density,charge):
  # solver the cold plasma dispersion relation, find k's given w (omega)
  # see Stix (1992) Chapter 1-3
  # w in wce
  R=1.
  L=1.
  P=1.
  for i in range(len(mass)):
    #wps=wpewce*np.sqrt(density[i]*charge[i]**2/mass[i])
    wps2=wpewce**2*(density[i]*charge[i]**2/mass[i])
    ws=charge[i]/mass[i]
    #print wps, ws
    #R-=wps**2/w/(w+ws)
    #L-=wps**2/w/(w-ws)
    #P-=wps**2/w**2
    R-=wps2/w/(w+ws)
    L-=wps2/w/(w-ws)
    P-=wps2/w**2
  #print P, R, L
  S=(R+L)/2.
  D=(R-L)/2.

  th=np.pi*th/180
  A=S*np.sin(th)**2+P*np.cos(th)**2
  B=R*L*np.sin(th)**2+P*S*(1+np.cos(th)**2)
  C=P*R*L
  F=np.sqrt((R*L-P*S)**2*np.sin(th)**4+4*P**2*D**2*np.cos(th)**2)
  #print A,B,C,F
  #return n^2
  return (B+F)/2/A, (B-F)/2/A, P, R, L, S, D

def growth(w,k,p,wpewce,mass,density,charge,temp,anis):
  ''' Estimate growth rate for PARALLEL propagating EM mode, with the
  approximation gamma<<omega 
  '''
  # w -- real frequency, in |wce| 
  # k -- parallel wave number, in |wce|/c, 
  # p=1 for right hand polarization, p=-1 for left hand polarization
  # only hot components contribute to growth rate
  # cold components play very little role due to out-of-resonance

  #print 'w=',w,'k=',k
  #print 'w/k=',w/k
  #print 'wpewce=',wpewce
  g=0.
  denom=2*w
  zeta=[]
  for j in range(len(mass)):
    # growth from hot components
    #print 'j=',j
    # (omega_j/Omega_e)**2
    wj2=density[j]*charge[j]**2/mass[j]*wpewce**2
    #print 'wj2=',wj2
    # (v_j/c)
    vj=np.sqrt(temp[j]/5.11e5/mass[j])
    #print 'vj=',vj
    # Omega_j/|Omega_e|
    Wj=charge[j]/mass[j]
    #print 'Wj=',Wj
    #print 'zeta_j=',(w+p*Wj)/k/vj/np.sqrt(2)
    zeta.append((w+p*Wj)/k/vj/np.sqrt(2))
    g+=wj2/vj*(-w*anis[j]/k+p*(1.-anis[j])*Wj/k)*np.exp(-((w+p*Wj)/k/vj)**2/2)
    #print 'g=',g
    denom-=wj2*p*Wj/(w+p*Wj)**2
    #denom-=wj2*(1+anis[j])/(w+p*Wj) #conside only the real part in 1/zeta_j
  g*=np.sqrt(np.pi/2)/denom
  #print 'final g=',g
  return g,zeta
