import numpy as np
import pymatgen.io.vasp as pmg
from pymatgen.core import Structure

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

