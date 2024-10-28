import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_bandE(folder,cuts,n,bandinds):
    """
    Reads the all eigenvalues in folder for the given cuts and return numpy
    array with the energies given of the bands given by bandinds
    :param folder:
    :param cuts:
    :param n:
    :param bandinds:
    :return:
    """
    bandE = []
    for cut in cuts:
        # Read eigenvalue file
        eign  = pd.read_csv( folder + '/EIGENVALS/EIGENVAL_rc_'
                            +str(cut)+'_n_'+str(n),
                            delim_whitespace=True,skiprows=8,header=None)
        bandE.append(eign.iloc[bandinds,[1,2]].to_numpy())
    bandE = np.array(bandE)
    return bandE


def cutoff_effect_plot(folder, cuts, n, bandinds, title='', ylim=None, ax=None, colors=None, labels=None,
                       addlegend=True):
    # Get energy of each band
    BandEn = get_bandE(folder, cuts, n, bandinds)

    # make figure object if needed
    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(1, 1, figsize=[8, 7])

    for i in range(BandEn.shape[1] - 1, -1, -1):
        if isinstance(colors, type(None)):
            color = None
        else:
            color = colors[i]

        if isinstance(labels, type(None)):
            label = None
        else:
            label = labels[i]
        # up
        ax.scatter(cuts, BandEn[:, i, 0], color=color, marker='^', s=100)
        ax.plot(cuts   , BandEn[:, i, 0], color=color, label=label)
        # down
        ax.scatter(cuts, BandEn[:, i, 1], color=color, marker='v', s=100)
        ax.plot(cuts   , BandEn[:, i, 1], color=color)

    ax.set_xlim([cuts[0], cuts[-1]])
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='both', labelsize=25)
    ax.set_xlabel(r'Cutoff radius ($a_0$)', fontsize=25)
    ax.set_ylabel(r'Energy eigenvalue at $\Gamma$ point', fontsize=25)
    ax.set_title(title, fontsize=30)

    if addlegend:
        ax.legend(fontsize=25, loc='upper right')

def find_lue(eign, tol=5e-3):
    # find index lue up
    im_up = eign.iloc[:, 3].idxmin()
    # Check the occupation of the level above the lue this might practically 0
    while eign.iloc[im_up-1,3] < 1-tol:
        im_up -= 1

    # find index lue down
    im_do = eign.iloc[:, 4].idxmin()
    while eign.iloc[im_do-1,4] < 1-tol:
        im_do -= 1

    if eign.iloc[im_up,1] > eign.iloc[im_do,2]:
        lue = eign.iloc[im_do,2]
        ind_lue = im_do
        spin = 1
    else:
        lue = eign.iloc[im_up,1]
        ind_lue = im_up
        spin = 2

    return lue, ind_lue, spin



def find_optimal_cutoff(folder, atomnames=None, print_output=False, cutoff_filename='CutoffOpt.csv', extrema_type='extrema'):
    """
    Looks for the optimal cut parameters as well as maximum gaps for a set of atoms within a defect
    :param folder: Folder containing the data
    :param atomnames: List of atom names or None to auto-detect
    :param print_output: Whether to print the output
    :param cutoff_filename: Filename of the cutoff data
    :param extrema_type: Type of extrema to find ('extrema', 'maximum', 'minimum')
    :return: list of rc, list of maximum gaps, list of all dataframes
    """
    if atomnames is None:
        # List all subfolders in the given folder
        subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]
        # Filter subfolders to match the format <integer>_<symbol_atom>_<other_integer>
        pattern = re.compile(r'^\d+_[A-Za-z]+_\d+$')
        atomnames = sorted([name for name in subfolders if pattern.match(name)], key=lambda x: int(x.split('_')[0]))

    rc_list = []
    max_gap_list = []
    gapdf_list = []
    for name in atomnames:
        # read csv files with gap data
        gap = pd.read_csv(os.path.join(folder, name, cutoff_filename))  # read csv file with gap as a function of rc
        # find extrema
        rcext, ext_gap, indext = find_extrema_gap(gap, extrema_type)
        # save data in appropriate structure
        rc_list.append(rcext)
        max_gap_list.append(ext_gap)
        gapdf_list.append(gap)
        if print_output:
            print('The extreme gap of ', name, 'is ', ext_gap, 'eV and is found at r_c = ', rcext, 'a.u.')

    return rc_list, max_gap_list, gapdf_list

def find_extrema_gap(rc_cutoff_df,extrema_type='extrema'):
    if extrema_type == 'extrema' or extrema_type == 'ext':
        return _find_extrema_gap(rc_cutoff_df)
    elif extrema_type == 'maximum' or extrema_type == 'max':
        indext  = rc_cutoff_df.iloc[:, 1].idxmax()
        rcext   = rc_cutoff_df.iloc[indext, 0]
        ext_gap = rc_cutoff_df.iloc[indext, 1]
        return rcext, ext_gap, indext
    elif extrema_type == 'minimum' or extrema_type == 'min':
        indext  = rc_cutoff_df.iloc[:, 1].idxmin()
        rcext   = rc_cutoff_df.iloc[indext, 0]
        ext_gap = rc_cutoff_df.iloc[indext, 1]
        return rcext, ext_gap, indext
    elif extrema_type == 'local maximum' or extrema_type == 'local max' or extrema_type == 'localmaximum':
        return find_local_max_gap(rc_cutoff_df)
    else:
        print('Unknown extrema type was given! Extrema type extrema was used!')
        return _find_extrema_gap(rc_cutoff_df)

def _find_extrema_gap(rc_cutoff_df):
    """
    finds the extremal gap in rc_cutoff_df and return the rc, gap and index of this extremum
    :param rc_cutoff_df:
    :return:
    """
    # Find max
    indmax = rc_cutoff_df.iloc[:, 1].idxmax()
    rcmax = rc_cutoff_df.iloc[indmax, 0]
    max_gap = rc_cutoff_df.iloc[indmax, 1]

    # Find min
    indmin = rc_cutoff_df.iloc[:, 1].idxmin()
    rcmin = rc_cutoff_df.iloc[indmin, 0]
    min_gap = rc_cutoff_df.iloc[indmin, 1]

    # rc proprties
    largest_rc = rc_cutoff_df.iloc[:, 0].max()
    smallest_rc = rc_cutoff_df.iloc[:, 0].min()

    # Find extrema
    if rcmax > smallest_rc and rcmax < largest_rc:
        # if rcmax is not at the edge we found the extrema
        return rcmax, max_gap, indmax

    elif rcmax == largest_rc and rcmin == 0:
        raise Warning('rcmin was found at 0 and rcmax was found at largest rc! Rc max is likely to small!')
    else:
        # if rc max is at the edges we return the minimum
        return rcmin, min_gap, indmin


def find_local_max_gap(self, rc_cutoff_df):
    """
    Finds the local maximum gap in rc_cutoff_df and returns the rc, gap and index of this maximum. This is done by
    first sorting the dataframe. Then the function will loop over the dataframe and check if the gap is larger than
    the previous and next gap. If this is the case the function will return the rc, gap and index of this maximum.
    If no local maximum is found the function will return the global maximum which should be located at the largest
    rc value.
    :param rc_cutoff_df:
    :return:
    """
    # Sort dataframe
    rc_cutoff_df = rc_cutoff_df.sort_values('Cutoff', axis=0)
    rc_cutoff_df = rc_cutoff_df.reset_index(drop=True) # reset index to make sure we can loop over the dataframe
    # Find local maximum
    for i in range(1, len(rc_cutoff_df)-1):
        # Check if the gap is larger than the previous and next gap
        logging.debug(f'i: {i}, rc: {rc_cutoff_df.iloc[i, 0]}, gap: {rc_cutoff_df.iloc[i, 1]}, gap_prev: {rc_cutoff_df.iloc[i-1, 1]}, gap_next: {rc_cutoff_df.iloc[i+1, 1]}')
        if rc_cutoff_df.iloc[i, 1] > rc_cutoff_df.iloc[i-1, 1] and rc_cutoff_df.iloc[i, 1] > rc_cutoff_df.iloc[i+1, 1]:
            logging.debug('Local maximum found at index %d' % i)
            # If this is the case we return the rc, gap and index of this maximum
            return rc_cutoff_df.iloc[i, 0], rc_cutoff_df.iloc[i, 1], i

    # If no local maximum is found we return the global maximum
    logging.debug('No local maximum found, returning global maximum')
    indmax = rc_cutoff_df.iloc[:, 1].idxmax()
    rcmax = rc_cutoff_df.iloc[indmax, 0]
    max_gap = rc_cutoff_df.iloc[indmax, 1]
    return rcmax, max_gap, indmax