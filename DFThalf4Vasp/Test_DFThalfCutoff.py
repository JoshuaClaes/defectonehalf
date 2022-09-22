import DFThalfCutoff as Cutoff
import potcarsetup as ps
import orbital
import numpy as np
import pickle

#############################################
# Fake NV center
#############################################

#############################################
# Nitrogen self energy
#############################################
workdir  ='Examples/LDA/NV_fakevasprun'   # folder in which calculation will be done
atomname = 'Nitrogen'        # label of the atom
atom     = 'N'                       # Atom symbol
orbitals = [1, 2]                # number of core and valence eletrons
GSorb    = [orbital.orbital(n=2,l=0,occ=2.00), orbital.orbital(n=2,l=1,occ=3.00)] # Ground state orbitals
EXtype   = 'ca'                   # exchange correlation used in atom (ca=lda, pb=pbe)
N_ps     = ps.potcarsetup(workdir,atomname,atom,orbitals,GSorb)

# Vs
xi   = [0.1,0.2]    # MADE UP XI VALUES
zeta = [0.25,0.05]  # MADE UP ZETA VALUES
N_ps.CalcSelfEnPot(xi,zeta)

# Make potcars
potcarfile = 'lda'
CutFuncPar= {
    'Cutoff': list(np.linspace(0.0,4.0,41)),
    'n': 8
}
N_ps.MakePotcar(potcarfile,CutFuncPar)
file = open(N_ps.workdir + '/N_ps.PotSetup','wb')
pickle.dump(N_ps,file)
file.close()


#############################################
# Defect carbon atoms self energy
#############################################
atomname = 'Cdef'        # label of the atom
atom     = 'C'                       # Atom symbol
orbitals = [1, 2]                # number of core and valence eletrons
GSorb    = [orbital.orbital(n=2,l=0,occ=2.00), orbital.orbital(n=2,l=1,occ=2.00)] # Ground state orbitals
EXtype   = 'ca'                   # exchange correlation used in atom (ca=lda, pb=pbe)
Cdef_ps  = ps.potcarsetup(workdir,atomname,atom,orbitals,GSorb)

# Vs
xi   = [0.2,0.15]    # MADE UP XI VALUES
zeta = [0.1,0.20]    # MADE UP ZETA VALUES
Cdef_ps.CalcSelfEnPot(xi,zeta)

# Make potcars
potcarfile = 'lda'
CutFuncPar= {
    'Cutoff': list(np.linspace(0.0,4.0,41)),
    'n': 8
}
Cdef_ps.MakePotcar(potcarfile,CutFuncPar)

file = open(Cdef_ps.workdir + '/Cdef_ps.PotSetup','wb')
pickle.dump(Cdef_ps,file)
file.close()

#############################################
# Find cutoff parameters
#############################################
# we choose the difference between the up bands 1022 and 1023
unoccband   = [1022,2]
occband     = [1023,2]
bulkpotcarloc = '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/Potentials/Examples/LDA/Cbulk/Cbulk_sp0.25/POTCAR_DFThalf/POTCAR_rc_2.4_n_8'
typevasprun = 'cp ../../EIGENVAL EIGENVAL'
AtomSelfEnPots = [N_ps, Cdef_ps]
PotcarLoc = ['/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/Potentials/Examples/LDA/NV_fakevasprun/POTCAR_C','/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/Potentials/Examples/LDA/NV_fakevasprun/POTCAR_N']
NVcutoff = Cutoff.DFThalfCutoff(AtomSelfEnPots,PotcarLoc,unoccband,occband,typevasprun=typevasprun, bulkpotcarloc=bulkpotcarloc)

#
rb = 0.0
rf = 4.0
nsteps_list = [9,11]
NVcutoff.FindCutoff(rb,rf,nsteps_list,CutFuncPar)