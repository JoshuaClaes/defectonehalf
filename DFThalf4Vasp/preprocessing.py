import pickle
import os
import shutil
import numpy as np
import pymatgen.io.vasp as pmg
from pymatgen.core import Structure

import DFThalf4Vasp.potcarsetup as ps

def print_band_characters(bandind, atomind, Peign, structure):
    if not (isinstance(atomind, list)):
        atomind = [atomind]
    for bi in bandind:
        print('\nBand:',bi)
        print('Orbitals  : \t [s p d]')
        if not( isinstance(bi,list)):
            bi = [bi]
        for ia in atomind:
            #print(Peign[0, bi, ia])
            print('\t\t', np.average(Peign[0, bi, ia], 0), structure[ia])


def calc_electron_fraction(Achar=None, mlt=None, Peign=None, iocc=None,iunocc=None,atominds=None,numdec=2):
    """
    This function calculates the fraction of an electron (Xi and Zeta) that needs to be
    substracted of each orbital of each atom for a defect dft-1/2 calculation

    Achar:    A list of list with each list containing the character of each
              atom ie [[Cs,Cp,Cd],[Ns,Np,Nd]]
    mlt:      A list containing the multiplicity of each atom
    """

    if not(isinstance(Achar,type(None)) ):
        Efrac = calc_electron_fraction_basic(Achar, mlt=mlt)
    elif not(isinstance(Peign,type(None))):
        Efrac = calc_electron_fraction_fullinput(Peign,iocc,iunocc,atominds,mlt)
    Efrac = np.round(Efrac,numdec)
    return Efrac

def full_band_character_analysis(folder, iocc, iunocc, atominds, spin, print_band_chars=True, print_xi_zeta=False):
    # load structure
    structure = Structure.from_file(folder + "/POSCAR")

    # load project eigenvalues
    run = pmg.Vasprun(folder + '/vasprun.xml', parse_potcar_file=False,
                      parse_eigen=True, parse_projected_eigen=True, separate_spins=True)
    Peign = run.projected_eigenvalues[spin]

    # Check bands
    if print_band_chars:
        print_band_characters([iocc, iunocc],atominds, Peign, structure)

    # Calculate electron fractions
    Xi, Zeta = calc_electron_fraction(Peign=Peign, iocc=iocc, iunocc=iunocc, atominds=atominds)

    if print_xi_zeta:
        print('Xi:\n', Xi)
        print('Zeta:\n', Zeta)

    return Xi, Zeta

def setup_calculation(atomnames, atoms, orbitals, GSorbs, Xi, Zeta, workdir, EXtype, potcarfile, cutfuncpar, vaspfiles = [] ):
    """
    This function setups a folder to prefrom a cutoff sweep from DFThalfCutoff
    :param atomnames: list with atom labels
    :param atoms: list with atomic symbols
    :param orbitals: list of list containing the number of core and valence electrons
    [ [#core e atom1, #val e atom1], [#core e atom2,#val e atom2], ..., [#core e atomn,#val e atomn]]
    :param GSorbs: list orbital object containing the ground state configuration
    :param Xi: list of xi for each orbtial of each atom [[xi_s a1, xi_p a1, xi_d a1],...]
    :param Zeta: list of zeta for each orbtial of each atom [[zeta_s a1, zeta_p a1, zeta_d a1],...]
    :param workdir: name/path to working directory
    :param EXtype: exchange correclation type for ATOM
    :param potcarfile: string with potcar file type 'lda' or 'pbe'
    :param cutfuncpar:  dict with cutoff function parameters
    :param vaspfiles: list of vasp files location which will be copied to the vasp_run file
    :return:
    """
    for i, atom in enumerate(atoms):
        # Calc Vs
        Vs = ps.PotcarSetup(workdir, atomnames[i], atom, orbitals[i], GSorbs[i], ExCorrAE=EXtype)
        Vs.calc_self_En_pot(Xi[i], Zeta[i])

        # Make potcars
        Vs.make_potcar(potcarfile, cutfuncpar)

        # Safe potcar setup object
        file = open(Vs.workdir + '/' +atomnames[i] + '_ps.PotSetup', 'wb')
        pickle.dump(Vs, file)
        file.close()

        # make vasp run folder
        if i == 0 and not(os.path.isdir(Vs.workdir + '/vasp_run')):
            # setup vasp calcualtion
            os.mkdir(Vs.workdir + '/vasp_run')
    # copy vasp files
    for file in vaspfiles:
        shutil.copy(file, workdir + '/vasp_run/' )
    return 0

#################################
# HELPER FUNCTIONS
#################################


def calc_electron_fraction_basic(Achar, mlt=None):
    """
    This function calculates the fraction of an electron (Xi and Zeta) that needs to be
    substracted of each orbital of each atom for a defect dft-1/2 calculation

    Achar:    A list of list with each list containing the character of each
              atom ie [[Cs,Cp,Cd],[Ns,Np,Nd]]
    mlt:      A list containing the multiplicity of each atom
    """

    if isinstance(mlt,type(None)):
        mlt = np.ones(len(Achar))

    # Calculate norm such that Sum char = 1/2
    norm = 0
    for c in range(0, len(Achar)):
        norm += mlt[c] * np.sum(Achar[c])
    norm = 2 * norm  # extra factor 2 such that the sums is equal to 1/2

    # Calculate electron fraction
    Efrac = []  # new list with normalised characters
    for ch in Achar:
        Efrac.append(ch / norm)

    return Efrac

def calc_electron_fraction_fullinput(Peign, iocc,iunocc,atominds, mlt=None):
    """
    Allows to calculate the electron fraction a more lazy manner using more inputs.
    :param Peign: projected eigenvalue object from pymatgen
    :param iocc: index occupied orbital
    :param iunocc: index unoccupied orbital
    :param atominds: indeces of all atoms involved
    :param mlt: list with multiplicity of each atoms. If none is give the multiplicty of each atom is assumed to be 1
    :return:
    """

    if not(isinstance(iocc,list)):
        iocc = [iocc]
    if not(isinstance(iunocc,list)):
        iunocc = [iunocc]

    char_occ   = []
    char_unocc = []
    for i, ai in enumerate(atominds):
        co = np.average(Peign[0, iocc  , ai], 0)
        char_occ.append(co)
        cu = np.average(Peign[0, iunocc, ai],0)
        char_unocc.append(cu)

    Xi = calc_electron_fraction_basic(char_occ  , mlt=mlt)
    Ze = calc_electron_fraction_basic(char_unocc, mlt=mlt)
    Efrac = [Xi, Ze]

    return  Efrac

