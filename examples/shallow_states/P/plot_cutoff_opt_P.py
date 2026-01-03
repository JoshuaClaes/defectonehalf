import matplotlib.pyplot as plt
from defectonehalf.postprocessing import find_optimal_cutoff
from defectonehalf.plotting import plot_cutoff_sweep

# Find optimal cutoff for P in Si. Currently this reads the data from a previous run in ./cutoff_opt_complete folder
# change this to cutoff_opt to compare.
rc, max_gap, df_cutsweep = find_optimal_cutoff(f'./cutoff_opt_complete', ['Phosporus'],
                                               print_output=True, extrema_type='local max')
# Print cutoff sweep data
print(df_cutsweep)

# Plot cutoff info
plot_cutoff_sweep(df_cutsweep = df_cutsweep, title='P donor in Si', labels=['P'])
plt.tight_layout()
plt.savefig('./P_cutoff_sweep.pdf')