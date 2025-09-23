"""
Example input file for a P donor in Si, using PBE functional and s-orbital only. This example was used to determine
the binding energy of the P donor in Si in the paper: https://arxiv.org/abs/2508.14738

This example uses the preprocessing module to set up the calculation. The module will create a folder calculate the
self-energy potential by doing two ATOM calculations. One for the neutral atom and one for the half-ionized atom. The
self-energy potential is then calculated by subtracting the two potentials. The module will also create a PAW potential
for VASP using the self-energy potential for each given Cutoff value in cutfuncpar. All files from the calculation will
be stored in the workdir folder.
"""

import defectonehalf.orbital as orbital
from defectonehalf.preprocessing import setup_calculation

# General settings
workdir = './P_selfenergypot'  # Folder where all files will be stored. If it does not exist it will be created.

# Define atom and orbitals
atomnames = ['Phosporus']   # Just a label for the atom. Can be anything.
atoms     = ['P']           # Atomic symbol from periodic table
orbitals  = [[3,2]]         # Number of core and valence orbitals. This is an input for the AE ATOM code.
# List of lists of orbital objects. Each list corresponds to one of the valence orbitals in 'orbitals'.
# Here there is only one atom and thus one list of orbitals.
GSorbs    = [[orbital.Orbital(n=3, l=0, occ=2.00),
             orbital.Orbital(n=3, l=1, occ=3.00)]]

# Electron fractions to be removed from each atom in atoms. In this case we only have P and we remove 0.5e from the s-orbital.
xi   = [[0.50, 0.00]] # Remove 0.5e from the s-orbital and 0e from the p-orbital
# Electron fractions to be added to each atom in atoms. In this case we do not add any electrons as the self-energy
# of the silicon conduction band is negligible.
zeta = [[0.00, 0.00]]

# Calculation parameters
EXtype  = 'pbr'     # Exchange correlation type used in ATOM.
potcarfile = 'pbe'  # Exchange correlation type used in VASP.
cutfuncpar = {
    'Cutoff': [0],  # We only create one PAW potential here. The cutoff optimizer can create additional potentials on the fly.
    'n': 8          # Exponent of the cutting function. Recommended value is 8.
}

# Set up calculation-
setup_calculation(atomnames, atoms, orbitals, GSorbs, xi, zeta, workdir,
                  EXtype, potcarfile, cutfuncpar, fullworkdirpath=True)