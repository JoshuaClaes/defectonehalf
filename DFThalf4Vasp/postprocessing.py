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

def find_optimal_cutoff(folder, atomnames, print_output=False, cutoff_filename = 'CutoffOpt.csv'):
    """
    Looks for the optimal cut parameters aswell as maximum gaps for a set of atoms within a defect
    :param folder:
    :param atomnames:
    :param print_output:
    :return: list of rc, list of maximum gaps, list of all dataframes
    """
    rc_list     = []
    max_gap_list = []
    gapdf_list  = []
    for name in atomnames:
        gap = pd.read_csv(folder + '/' + name + '/' + cutoff_filename)  # read csv file with gap as a function of rc
        indmax = gap.iloc[:, 1].idxmax()  # find maximum cutoff
        rc = gap.iloc[indmax, 0]  # cutoff radius
        max_gap = gap.iloc[indmax, 1]  # maximum gap
        rc_list.append(rc)
        max_gap_list.append(max_gap)
        gapdf_list.append(gap)
        if print_output:
            print('The maximum gap of ', name, 'is ', max_gap, 'eV and is found at r_c = ',
                  rc, 'a.u.')

    return rc_list, max_gap_list, gapdf_list
