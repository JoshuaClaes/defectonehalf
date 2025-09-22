from defectonehalf.postprocessing import find_optimal_cutoff, get_atom_names
import matplotlib.pyplot as plt

def plot_cutoff_sweep(df_cutsweep=None, ax = None, folder=None, atomnames=None,
                      labels=None, colors=None, title=None, cutoff_filename='CutoffOpt.csv'):

    if atomnames is None and folder is not None:
        atomnames = get_atom_names(folder)

    if folder is not None and df_cutsweep is None:
        _, _, df_cutsweep = find_optimal_cutoff(folder,atomnames, cutoff_filename==cutoff_filename)

    if labels is None:
        labels=atomnames

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[8, 7])

    for i, df in enumerate(df_cutsweep):
        color = None
        if colors is not(None):
            color = colors[i]
        df = df.sort_values('Cutoff',axis=0)
        ax.plot(df.iloc[:,0], df.iloc[:,1], 'o-', label=labels[i], color=color)

    xlim = [df_cutsweep[0].iloc[:,0].min(),df_cutsweep[0].iloc[:,0].max()]
    ax.set_title(title, fontsize=20);
    ax.tick_params(which='both', labelsize=15);
    ax.set_ylabel('$E_{gap}$ (eV)', fontsize=25)
    ax.set_xlabel('Cutoff ($a_0$)', fontsize=25)
    ax.set_xlim(xlim)
    ax.legend(fontsize=15)


def decoupled_analysis(folder, atomnames, E_bg, atom_symbols=None, colors=None, print_output=True,
                       title_xi='occ to CBM', title_zeta='VBM to unocc', extrema_type='extrema', make_plot=True):
    """
    Makes a plot of the energy gap for xi and zeta.

    Parameters
    ----------
    folder : str
        Folder containing decoupled run.
    atomnames : list of str
        List of defect atoms in the system.
    E_bg : float,
        Bandgap energy in eV. Preferably this is obtained from a separate DFT-1/2 bulk calculation
    atom_symbols : list of str
        Symbol of atoms. This is used as the label and is visible in the legend of the figures. Default is None
    colors : list of str, optional
        Colors for the data in the plots.
    print_output : bool, optional
        Whether to print the output. Default is True.
    title_xi : str, optional
        Title for the xi plot. Default is 'occ to CBM'.
    title_zeta : str, optional
        Title for the zeta plot. Default is 'VBM to unocc'.
    extrema_type : str, optional
        Type of extrema to use for the analysis. Options are 'extrema', 'maximum' or 'minimum'. Default is 'extrema'.
    make_plot: bool, optional
        If true(default) this function will plot the curve op the gap optimization. It is generally recommended to look
        at these curves.

    Returns
    -------
    Egap : float
        Gap energy in eV.
    gap_occ_CBM : float
        Gap between the highest occupied state and the conduction band minimum.
    gap_unocc_VBM : float
        Gap between the valence band maximum and the lowest unoccupied state.
    """

    # Plot xi
    if print_output:
        print('Xi')
    # Find the optimal cutoff for xi and get a dataframe with the values
    rc, max_gap, df_cutsweep = find_optimal_cutoff(folder + '/xi', atomnames, print_output=print_output,
                                                   extrema_type=extrema_type)
    # Store sweep data in dictionary
    sweepdata = {'xi': {'rc': rc, 'max_gap': max_gap, 'df_cutsweep': df_cutsweep}}


    if make_plot:
        # Create a figure with two subplots, one for xi and one for zeta
        fig, ax = plt.subplots(1, 2, figsize=[2 * 8, 1 * 8])
        # Plot the values in the dataframe
        plot_cutoff_sweep(df_cutsweep=df_cutsweep, atomnames=atomnames, title=title_xi, colors=colors, labels=atom_symbols, ax=ax[0])

    # Get the maximum gap for xi
    gap_occ_CBM = max_gap[-1]

    # Plot zeta
    if print_output:
        print('\nZeta')
    # Find the optimal cutoff for zeta and get a dataframe with the values
    rc, max_gap, df_cutsweep = find_optimal_cutoff(folder + '/zeta', atomnames, print_output=print_output,
                                                   extrema_type=extrema_type)
    # Store sweep data in dictionary
    sweepdata['zeta'] = {'rc': rc, 'max_gap': max_gap, 'df_cutsweep': df_cutsweep}

    if make_plot:
        # Plot the values in the dataframe
        plot_cutoff_sweep(df_cutsweep=df_cutsweep, atomnames=atomnames, title=title_zeta, colors=colors, labels=atom_symbols, ax=ax[1])
        # Adjust the spacing between the subplots
        plt.tight_layout()

    # Get the maximum gap for zeta
    gap_unocc_VBM = max_gap[-1]

    # Print Egap
    Egap = gap_unocc_VBM + gap_occ_CBM - E_bg
    if print_output:
        print('\nEgap =', Egap)

    # Return the values of Egap, gap_occ_CBM, and gap_unocc_VBM
    return Egap, gap_occ_CBM, gap_unocc_VBM, sweepdata

