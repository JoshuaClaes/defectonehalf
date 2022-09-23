import numpy as np
import matplotlib.pyplot as plt
import potcarsetup
import orbital
from PotcarWrapper import ReadPotcarfile
import time

def myAdd2PotcarFourier(ca,radii,Vs,Cut):
    # based on the add2POTCAR-eng.f90 fortran script
    # this is quite slow
    fourier = 0
    iniciovez = 0

    for i, r in enumerate(radii):
        if r <= 0:
            continue
            # nothting
        elif iniciovez == 0:
            fourier = fourier + (Vs[i] * np.sin(ca * r))
            iniciovez = 1
            continue
        elif r > Cut:
            fourier = fourier + (Vs[i] * np.sin(ca * r) )
            break
        else:
            fourier = fourier + (Vs[i] * np.sin(ca * r) + Vs[i - 1] * np.sin(ca * radii[i - 1])) * (
                        r - radii[i - 1]) / 2.0
    return fourier


# Create potcar setup object
atomname= 'Ctestatom'
atom= 'C'
orbitals= [1, 2]
GSorb = [orbital.orbital(n=2,l=0,occ=2.00), orbital.orbital(n=2,l=1,occ=2.00)]
ps = potcarsetup.potcarsetup('test_potcarsetup2',atomname,atom,orbitals,GSorb)

# CALCULATED VS
xi    = [0.25,0.25]
zeta  = [0.00,0.00]
CutFuncPar= {
    'Cutoff': 2.5,
    'n': 8
}

_  = ps.CalcSelfEnPot(xi,zeta)
Vs = ps.DefCalcTrimmedVs(CutFuncPar)
########################################################
# MAKE POTCAR
########################################################
potcarfile = '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/test_AW/POTCAR'
newpotcarfile = '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/Potentials/test_potcarsetup2/testpotcar'

print('start timing')
st = time.time()
newpotcar = ps.AddVs2Potcar(Vs=Vs,potcarfile=potcarfile,newpotcarfile=newpotcarfile,Cutoff=CutFuncPar['Cutoff'])
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

########################################################
# MY NEW POTCAR
########################################################
potcar,nrows,kmax,_ = ReadPotcarfile(potcarfile)
Cut = CutFuncPar['Cutoff']



st = time.time()
nk = nrows * 5  # number of kvalues in potcar file
kp = ps.beta*np.linspace(kmax/nk,kmax,nk)
mynewpotcar = potcar.copy()
# first element
I = np.arange(1,len(ps.Radii))
for i, k in enumerate(kp):
    fourier = 0
    fourier += (Vs[0] * np.sin(k * ps.Radii[0]))*(ps.Radii[0])/2.0
    # elements with R<cut
    fourier += np.sum( ( (Vs[I]*np.sin(k*ps.Radii[I]) + Vs[I-1] * np.sin(k * ps.Radii[I-1])) *
                       (ps.Radii[I] - ps.Radii[I-1]) )*(ps.Radii[I]<Cut) )/ 2.0
    # elements with R>cut
    indRcut = np.argmax(ps.Radii>=Cut)
    fourier += (Vs[indRcut] * np.sin(k * ps.Radii[indRcut])*(Cut-ps.Radii[indRcut]) )/ 2.0 # This is always 0 it seems
    # update potcar with value or fourier transform
    mynewpotcar[i] = mynewpotcar[i] + fourier/(k)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

# plot both
plt.figure()
plt.plot(potcar, label='oldpotcar')
plt.plot(newpotcar, label='fortran FFT')
plt.plot(mynewpotcar,label='numpy FFT')
plt.legend()
plt.show()

plt.figure()
plt.plot(newpotcar-mynewpotcar,label='fotran FFT - myFFT')
plt.legend()
plt.show()
