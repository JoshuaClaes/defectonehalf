import DFThalfCutoff as Cutoff
import numpy as np
import pickle

#############################################
# Fake NV center 2 Run with pre generated potcar setup objects
#############################################

# load potcar setup objects
N_file_loc = '/mnt/extradata/DFThalf/SelfEnergyPot_Auto/Potentials/Examples/LDA/NV_fakevasprun/N_ps.PotSetup'
with open(N_file_loc,'rb') as file:
    N_ps = pickle.load(file)

Cdef_file_loc = '/mnt/extradata/DFThalf/SelfEnergyPot_Auto/Potentials/Examples/LDA/NV_fakevasprun/Cdef_ps.PotSetup'
with open(Cdef_file_loc,'rb') as file:
    Cdef_ps = pickle.load(file)

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
nsteps_list = [9,11,11]
CutFuncPar= {
    'Cutoff': list(np.linspace(0.0,4.0,41)),
    'n': 8
}
NVcutoff.FindCutoff(rb,rf,nsteps_list,CutFuncPar)