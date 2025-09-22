import numpy as np
import pandas as pd
import linecache

def find_kmax(potcarfile):
    with open(potcarfile) as pfile:
        for i,line in enumerate(pfile):
            if ' local part\n' in line:
                linekmax = i+2;
                break

    kmax = float(linecache.getline(potcarfile, linekmax))  # read kmax
    return kmax,linekmax


def find_nrows(potcarfile):
    with open(potcarfile) as pfile:
        firstline = None
        lastline = None
        for i, line in enumerate(pfile):
            if 'local part\n' in line:
                firstline = i + 2
            elif ('gradient corrections used for XC' in line) or ('core charge-density (partial)' in line) or (' atomic pseudo charge-density' in line):
                lastline = i
                break
        if firstline == None:
            raise Exception('Line containing "Local part" was not found in potcar file!')
        elif lastline == None:
            raise Exception('Line containing "gradient corrections used for XC" or "core charge-density (partial)" was not found in potcar file!')
        else:
            nrows = lastline - firstline

    return nrows


def read_potcar_file(file, skiprows=None, nrows=None):
    """
    Read potcar file and return a numpy array with the local part
    :param file: filename potcar file
    :param skiprows: number of rows that need to be skipped. Can be given for debugging or edge cases
    :param nrows: number of rows that need to be read. Can be given for debugging or edge cases
    :return: np.array with potcar values
    """
    kmax = None
    if skiprows==None:
        kmax, skiprows = find_kmax(file)
    if nrows == None:
        nrows = find_nrows(file)

    potcar = pd.read_csv(file, nrows=nrows, delim_whitespace=True, skiprows=skiprows, header=None)
    potcar = np.concatenate(potcar.to_numpy())    # convert to numpy and concatenate to single array

    return potcar, nrows, kmax, skiprows
