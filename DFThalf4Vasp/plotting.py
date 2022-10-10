from DFThalf4Vasp.postprocessing import find_optimal_cutoff
import matplotlib.pyplot as plt

def plot_cutoff_sweep(df_cutsweep=None, ax = None, folder=None, atomnames=None,
                      labels=None, colors=None, title=None):
    if folder is not None and atomnames is not None:
        _, _, df_cutsweep = find_optimal_cutoff(folder,atomnames)

    if labels is None:
        labels=atomnames

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=[8, 7])

    for i, df in enumerate(df_cutsweep):
        if colors is None:
            color = None
        else:
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