import os
import pandas as pd
import numpy as np
import ase
from ase.calculators.vasp import Vasp
from ase.dft.bandgap import bandgap

from DFThalf4Vasp import VaspWrapper

class VaspWrapperAse(VaspWrapper.VaspWrapper):
    """
    A vasp wrapper which uses the ASE frame library
    """
    def __init__(self):
        super().__init__()

    def run_vasp(self, foldervasprun, typevasprun):
        # Go vasp run directory
        oldpath = os.getcwd()
        os.chdir(foldervasprun)

        # Run vasp
        if typevasprun == 'vasp_std' or typevasprun == 'std':
            os.system('srun vasp_std >> vasp.out')
        elif typevasprun == 'vasp_gam' or typevasprun == 'gam':
            os.system('srun vasp_gam >> vasp.out')
        elif typevasprun == 'vasp_ncl' or typevasprun == 'ncl':
            os.system('srun vasp_ncl >> vasp.out')
        else:
            # incase another type is given we try to run the given string
            os.system(typevasprun)
        # Go back to original path
        os.chdir(oldpath)

    def calculate_bandgap(self):
        # Get information from previous Vasp run
        calc = Vasp(directory='', restart=True, xc='lda')
        calc.read_results()

        # Calculate band gap
        bg = bandgap(calc, output=None)[0]  # The actual gap is found at position 0 in the array
        return bg

    def calculate_gap(self, bands, spins, vaspfolder='./', kpoints=0):
        """
        Calculalate gap will calculate the gap between 2 different bands.
        :param bands: list 2 indices [index lowest band, index highest band]
        :param spins: list containing the spins of the bands. 1=up and 2=down or use strings 'up' and 'down'.
        Example: [1, 2], ['up','down']
        If the input is not a list but an int or str, calculate gap will assume that both bands have the same spin.
        :param vaspfolder:
        :param kpoints: list of 2 indices with the kpoints for which the gap should be calculated. If a single interger
        is give calculate_gap will calculate the gap a the same kpoints.
        Altenatively: kpoints can be set to None or all which will calculate the indirect gap between the 2 given bands
        :return: float
        """

        # format input
        if not(isinstance(spins,list)):
            spins = [spins, spins]       # convert spins to a list

        # Convert spins from strings to integers and make integers ase compatible
        for i, spin in enumerate(spins):
            if spin == 'up':
                spins[i] = 0
            elif spin == 'down':
                spins[i] = 1
            elif spin == 1:
                spins[i] = 0 # Convert index to ase format
            elif spin == 2:
                spins[i] = 1 # convert index to ase format
            else:
                raise Exception('Invalid spin was given to calculate gap!\nSpin can either be a str containing up or' +
                                ' down or an integert with 1=up and 2=down')

        if not(isinstance(kpoints,list)):
            kpoints = [kpoints, kpoints] # convert kpoints to a list

        # Get information from previous Vasp run
        calc = Vasp(directory='', restart=True, xc='lda')
        calc.read_results()

        # Get eigenvalues from vasp calculation
        bs = calc.band_structure()

        if kpoints[0] is None or kpoints[0] == 'all':
            # get energies for all kpoints
            en_low  = bs.energies[0,:,bands[0]]
            en_high = bs.energies[0,:,bands[1]]

            gap = np.min(en_high) - np.max(en_low) # calculate inderect gap between bands
        else:
            en_low  = bs.energies[0,kpoints[0],bands[0]]
            en_high = bs.energies[0,kpoints[1],bands[1]]

            gap = en_high - en_low

        return gap