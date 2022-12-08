import os
import numpy as np
from DFThalf4Vasp import VaspWrapper

class VaspWrapperSimple(VaspWrapper.VaspWrapper):
    """
    A child of the VaspWrapper class which will interact with vasp using only bacis python functionalities and libraries.
    """
    def __init__(self):
        super().__init__()
        #VaspWrapper.__init__(self)

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

    def calculate_gap(self,bands,spins,foldervasprun='./',kpoints=0):
        """

        :param bands: list with the indices of the lower en higher band, [lower band, higher band]
        :param spins: list containing the spins of the bands. 1=up and 2=down or use strings 'up' and 'down'.
        Example: [1, 2], ['up','down']
        If the input is not a list calculate gap will assume
        that both bands have the same spin.
        :param vaspfolder: path to folder were vasp calculation ran.
        :param kpoints: list of 2 indices with the kpoints for which the gap should be calculated. If a single interger
        is give calculate_gap will calculate the gap a the same kpoints.
        Altenatively: kpoints can be set to None or all which will calculate the indirect gap between the 2 given bands
        :return:
        """

        # format input
        if not (isinstance(spins, list)):
            spins = [spins, spins]  # convert spins to a list

        # Convert spins from strings to integers and make integers ase compatible
        for i, spin in enumerate(spins):
            if spin == 'up':
                spins[i] = 0
            elif spin == 'down':
                spins[i] = 1
            elif spin == 1:
                spins[i] = 0  # Convert index to correct format
            elif spin == 2:
                spins[i] = 1  # convert index to correct format
            else:
                raise Exception('Invalid spin was given to calculate gap!\nSpin can either be a str containing up or' +
                                ' down or an integert with 1=up and 2=down')

        if not (isinstance(kpoints, list)):
            kpoints = [kpoints, kpoints]  # convert kpoints to a list

        # Read eigenvalues
        _, eign = self._read_eigenvalues(foldervasprun + '/EIGENVAL')
        # Calculate gap
        if kpoints[0] is None or kpoints[0] == 'all':
            # get energies for all kpoints
            en_low  = eign[:, bands[0], spins[0]]
            en_high = eign[:, bands[1], spins[1]]
            gap = np.min(en_high) - np.max(en_low)  # calculate inderect gap between bands
        else:
            gap = eign[kpoints[1], bands[1], spins[1]] - eign[kpoints[0], bands[0], spins[0]]
        return gap

    def calculate_bandgap(self, foldervasprun=None):
        raise Exception('calculate bandgap is not implemeneted in VaspWrapperSimple!\nUse calculate gap instead!')

    def _read_eigenvalues(self,filename):
        # This code was written with assistance of openai chatGPT
        # open the EIGENVAL file
        with open(filename, "r") as eigenval_file:
            # skip the first 5 lines of the file (header)
            for i in range(5):
                eigenval_file.readline()

            # read the next line to get the number of electrons, k-points, and bands
            line = eigenval_file.readline()
            num_electrons, num_kpoints, num_bands = line.split()
            # num_electrons = int(num_electrons)
            num_kpoints = int(num_kpoints)
            num_bands = int(num_bands)

            # initialize empty lists to store the k-point coordinates and eigenvalues
            kpoint_coords = []
            eigenvalues = []

            # read the remaining lines of the file
            for i in range(num_kpoints):
                # skip the empty line between eigenvalue blocks
                eigenval_file.readline()

                # read the next line to get the k-point coordinates
                line = eigenval_file.readline()
                kpoint_coords.append([float(v) for v in line.split()])

                # initialize an empty list to store the eigenvalues for this k-point
                kpoint_eigenvalues = []

                # read the eigenvalue lines for this k-point
                for j in range(num_bands):
                    line = eigenval_file.readline()
                    kpoint_eigenvalues.append([float(v) for v in line.split()[1:]])

                # store the eigenvalues for this k-point in the eigenvalues list
                eigenvalues.append(kpoint_eigenvalues)

        # convert the k-point coordinates and eigenvalues to NumPy arrays
        kpoint_coords = np.array(kpoint_coords)
        eigenvalues = np.array(eigenvalues)

        # return the k-point coordinates and eigenvalues
        return kpoint_coords, eigenvalues
