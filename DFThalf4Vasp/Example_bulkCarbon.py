import DFThalf4Vasp.potcarsetup as ps
import DFThalf4Vasp.orbital as orbital
import numpy as np
import pickle

#############################################
# Bulk V(2s,2p)-V(1.75s,1.75p)
#############################################
workdir  ='Examples/LDA/Cbulk'   # folder in which calculation will be done
atomname = 'Cbulk_sp0.25'        # label of the atom
atom     = 'C'                       # Atom symbol
orbitals = [1, 2]                # number of core and valence eletrons
GSorb    = [orbital.Orbital(n=2, l=0, occ=2.00), orbital.Orbital(n=2, l=1, occ=2.00)] # Ground state orbitals
EXtype   = 'ca'                   # exchange correlation used in atom (ca=lda, pb=pbe)
Cbulk_ps = ps.PotcarSetup(workdir, atomname, atom, orbitals, GSorb)

# Vs
xi   = [0.25,0.25]
zeta = [0.0,0.0]
Cbulk_ps.calc_self_En_pot(xi, zeta)

# Make potcars
potcarfile = 'lda'
CutFuncPar= {
    'Cutoff': list(np.linspace(0.0,4.0,41)),
    'n': 8
}
Cbulk_ps.make_potcar(potcarfile, CutFuncPar)

file = open(Cbulk_ps.workdir + '/Cbulk_ps.PotSetup','wb')
pickle.dump(Cbulk_ps,file)
file.close()

#############################################
# Bulk V(2s,2p)-V(1.75s,1.75p)
#############################################
workdir  ='Examples/PBE/Cbulk'   # folder in which calculation will be done
atomname = 'Cbulk_sp0.25'        # label of the atom
atom     = 'C'                       # Atom symbol
orbitals = [1, 2]                # number of core and valence eletrons
GSorb    = [orbital.Orbital(n=2, l=0, occ=2.00), orbital.Orbital(n=2, l=1, occ=2.00)] # Ground state orbitals
EXtype   = 'pb'                   # exchange correlation used in atom (ca=lda, pb=pbe)
Cbulk_ps = ps.PotcarSetup(workdir, atomname, atom, orbitals, GSorb)

# Vs
xi   = [0.25,0.25]
zeta = [0.0,0.0]
Cbulk_ps.calc_self_En_pot(xi, zeta)

# Make potcars
potcarfile = 'pbe'
CutFuncPar= {
    'Cutoff': list(np.linspace(0.0,4.0,41)),
    'n': 8
}
Cbulk_ps.make_potcar(potcarfile, CutFuncPar)

file = open(Cbulk_ps.workdir + '/Cbulk_ps.PotSetup','wb')
pickle.dump(Cbulk_ps,file)
file.close()