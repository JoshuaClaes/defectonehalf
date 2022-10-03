import numpy as np

def print_band_characters(bandind, atomind, Peign, structure):
    if not (isinstance(bandind, list)):
        bandind = [bandind]
    for bi in bandind:
        print('\nBand:',bi)
        print('Orbitals  : \t [s p d]')
        for ia in atomind:
            print('\t\t', np.average(Peign[0, bandind, ia], 0), structure[ia])


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

    char_occ   = []
    char_unocc = []
    for ai in enumerate(atominds):
        co = np.average(Peign[0, iocc  , ai], 0)
        char_occ.append(co)
        cu = np.average(Peign[0, iunocc, ai],0)
        char_unocc.append(cu)

    Xi = calc_electron_fraction_basic(char_occ  , mlt=mlt)
    Ze = calc_electron_fraction_basic(char_unocc, mlt=mlt)
    Efrac = [Xi, Ze]

    return  Efrac

