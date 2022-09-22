import numpy as np

import potcarsetup
import orbital


# Create potcar setup object
atomname= 'Ctestatom'
atom= 'C'
orbitals= [1, 2]
GSorb = [orbital.orbital(n=2,l=0,occ=2.00), orbital.orbital(n=2,l=1,occ=2.00)]
ps = potcarsetup.potcarsetup('test_potcarsetup2',atomname,atom,orbitals,GSorb)

# Calculate self energy potential
xi    = [0.25,0.25]
zeta  = [0.00,0.00]
Vs = ps.CalcSelfEnPot(xi,zeta)
print(Vs)
# Make new potcar file
potcarfile = '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/test_AW/POTCAR'
CutFuncPar= {
    'Cutoff': 2.5,
    'n': 8
}

ps.MakePotcar(potcarfile,CutFuncPar)


potcarfile = '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/test_AW/POTCAR'
CutFuncPar= {
    'Cutoff': list(np.linspace(0.0,4.0,41)),
    'n': 8
}
ps.MakePotcar(potcarfile,CutFuncPar)