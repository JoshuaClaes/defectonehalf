import pymatgen.io.vasp as pmg
from pymatgen.core import Structure
from pymatgen.electronic_structure.core import Spin
from DFThalf4Vasp.preprocessing import print_band_characters, calc_electron_fraction

folder = '/mnt/extradata/VASP_calculations/LDA/NVcenter/ZPL/NV0/excited_states/Up/scf'
iocc      = 1021
iunocc    = [1022, 1023]
atominds  = [507,509,508,510]
spin      = Spin.up

# load structure
structure = Structure.from_file(folder + "/POSCAR")

# load project eigenvalues
run   = pmg.Vasprun(folder+'/vasprun.xml', parse_potcar_file=False,
                     parse_eigen=True, parse_projected_eigen=True, separate_spins=True)
Peign = run.projected_eigenvalues[spin]

# Check bands
print_band_characters([iocc, iunocc],atominds, Peign, structure)

# Calculate electron fractions
Xi, Zeta = calc_electron_fraction(Peign=Peign, iocc=iocc, iunocc=iunocc, atominds=atominds)
print(Xi)
print(Zeta)