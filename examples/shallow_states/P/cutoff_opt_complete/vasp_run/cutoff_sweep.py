import os
import shutil
import sys
import pickle
import defectonehalf.DFThalfCutoff as Cutoff
from defectonehalf.VaspWrapperAse import VaspWrapperAse

# input
sc = 4 # supercell size set to 2x2x2 for testing and 4x4x4 for actual calculation
my_vasp_cmd = 'srun vasp_gam'

# set path to potcar files for ase vasp wrapper. defectonehalf does not use these potcars but ase requires it.
os.environ['VASP_PP_PATH'] = ''

# Set POSCAR file
shutil.copyfile(f'./POSCAR_{sc}', 'POSCAR')

##################################################
# load potcar setup objects
##################################################
file_loc = '../Phosporus_ps.PotSetup'
with open(file_loc,'rb') as file:
    P_ps = pickle.load(file)
P_ps.workdir = '../'

##################################################
# Cutoff sweep
##################################################\
# Determine occupied and unoccupied band
nelect = 4*8*(sc**3) # number of electrons in pristine supercell. This is used to set ferwe and ferdo in vasp
filled_upband = nelect/2 + 1 # number of filled bands in doped supercell

# INPUT DFThalfCutoff
kwargs = {
    "AtomSelfEnPots": [P_ps],
    "PotcarLoc"     : ['../../POTCAR_P'],
    # We choose the difference between the highest occupied and the lowest unoccupied band for spin up channel
    "occband"       : [int(filled_upband -1), 'up', 'all'], # band index, spin, kpoints
    "unoccband"     : [int(filled_upband)   , 'up', 'all'], # band index, spin, kpoints
    # python script to run the two step vasp calculation instead of the default vasp call
    "typevasprun"   : f'python ../../../Eb_run.py --sc {sc} --vasp-cmd "{my_vasp_cmd}"',
    "bulkpotcarloc" : '../../POTCAR_Sibulk',
    "save_eigenval" : True,
    "save_doscar"   : False,
    "extrema_type"  : "local max", # We want to find a local max in the occupied band. This procedure for find the extrema is the most rebust.
    "vasp_wrapper"  : VaspWrapperAse() # We use the ase vasp wrapper
}

kwargs_fc = {
    "rb": 0.0, # starting cutoff
    "rf": 4.0, # final cutoff
    # number of steps in the cutoff range. For the initial sweep we use a coarse grid with 9 steps. After we found a
    # minimum we do a finer sweep with 11 steps around the minimum. Resulting precisio is about 0.1 a0.
    "nsteps_list"   : [9,11],
    "cut_func_par"  : {'n': 8, 'Cutoff': 0.0}
}

# Initialize class
Eb = Cutoff.DFThalfCutoff(**kwargs)
Eb.foldervasprun = os.getcwd() # folder where the vasp calculations are run

# Find cutoff
Eb.find_cutoff(**kwargs_fc)