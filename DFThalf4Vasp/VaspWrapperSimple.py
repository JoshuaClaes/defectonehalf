import os
import pandas as pd
from DFThalf4Vasp import VaspWrapper

class VaspWrapperSimple(VaspWrapper):
    """
    A child of the VaspWrapper class which will interact with vasp using only bacis python functionalities and libraries.
    """
    def __init__(self):
        VaspWrapper.__init__(self)

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
        return 0

    def calculate_gap(self,bands,spins,vaspfolder='./',kpoints=0):
        """

        :param bands: list with the indices of the lower en higher band, [lower band, higher band]
        :param spins: list containing the spind of the bands. 1=up and 2=down or use strings 'up' and 'down'.
        Example: [1, 2], ['up','down']
        If the input is not a list calculate gap will assume
        that both bands have the same spin.
        :param vaspfolder: path to folder were vasp calculation ran.
        :param kpoints: list with kpoints, VaspWrapperSimple can currently only deal with gamma point calculation so
        this should always be set to 0.
        :return:
        """

        if kpoints is not 0:
            raise Exception('kpoints is not equal to 0!\nWhile VaspWrapperSimple only works for gamma point calculations!')

        if not(isinstance(spins,list)):
            spins = [spins, spins] # convert spins to a list

        """Convert spins from strings to intergers"""
        for i, spin in enumerate(spins):
            if spin == 'up':
                spins[i] = 1
            elif spin == 'down':
                spins[i] = 2

        # Read eigenvalues
        eign = pd.read_csv(vaspfolder + '/EIGENVAL', delim_whitespace=True, skiprows=8, header=None)
        # Calculate gap
        gap = eign.iloc[bands[0], spins[0]] - eign.iloc[bands[1], spins[1]]
        return gap

def calculate_bandgap(self):
    raise Exception('calculate bandgap is not implemeneted in VaspWrapperSimple!\nUse calculate gap instead!')
    return 0