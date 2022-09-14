import numpy as np
import pandas as pd
import linecache

def FindKmax(potcarfile):
    with open(potcarfile) as pfile:
        for i,line in enumerate(pfile):
            if ' local part\n' in line:
                linekmax = i+2;
                break

    kmax = float(linecache.getline(potcarfile, linekmax))  # read kmax
    return kmax,linekmax


def FindNrows(potcarfile):
    with open(potcarfile) as pfile:
        firstline = None
        lastline = None
        for i, line in enumerate(pfile):
            if 'local part\n' in line:
                firstline = i + 2
            elif 'gradient corrections used for XC\n' in line:
                lastline = i
        if firstline == None:
            raise Exception('Line containing "Local part" was not found in potcar file!')
        elif lastline == None:
            raise Exception('Last containing "  gradient corrections used for XC" was not found in potcar file!')
        else:
            nrows = lastline - firstline

    return nrows


def ReadPotcarfile(file, skiprows=None, nrows=None):
    """
    Read potcar file and return a numpy array with the local part
    :param self:
    :param file:
    :param skiprows:
    :param nrows:
    :return:
    """
    if skiprows==None:
        kmax, skiprows = FindKmax(file)
    if nrows == None:
        nrows = FindNrows(file)

    potcar = pd.read_csv(file, nrows=nrows, delim_whitespace=True, skiprows=skiprows, header=None)
    potcar = np.concatenate(potcar.to_numpy())    # convert to numpy and concatenate to single array

    return potcar, nrows, kmax, skiprows
