"""
Run a two-step VASP simulation for use with DFThalfCutoff.

This script is intended to be passed as the `typevasprun` argument of the
DFThalfCutoff object. It automates the two-step VASP execution workflow:
first running with `INCAR1`, then repeating with `INCAR2`. The appropriate
VASP binary (e.g. `vasp_std` or `vasp_gam`) is selected automatically based
on the supercell size, unless overridden by the user. By encapsulating these
steps here, DFThalfCutoff can remain independent of the underlying VASP run
logic.
"""

import os
import shutil
import subprocess
import logging
import click

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@click.command()
@click.option('--sc', default=2, show_default=True, help='Supercell size (sc_size)')
@click.option('--vasp-cmd', default=None, help='Optional override for VASP binary (e.g., vasp_std, vasp_gam)')
def main(sc, vasp_cmd):
    logging.info(f'Starting simulation with sc_size = {sc}')

    # Determine which VASP binary to use
    if vasp_cmd:
        vasp_binary = vasp_cmd
        logging.info(f'Overriding VASP binary with user-provided value: {vasp_binary}')
    else:
        vasp_binary = "srun vasp_gam >> vasp.out" if sc > 3 else "srun vasp_std >> vasp.out"
        logging.info(f'Using default logic to select VASP binary: {vasp_binary}')

    # Step 1
    logging.info('Step 1: Running VASP with INCAR1')
    shutil.copyfile('INCAR1', 'INCAR')
    #subprocess.run([vasp_binary], check=True)
    os.system(vasp_binary)

    # Step 2
    logging.info('Step 2: Running VASP with INCAR2')
    shutil.copyfile('INCAR2', 'INCAR')
    #subprocess.run([vasp_binary], check=True)
    os.system(vasp_binary)
    logging.info('Simulation completed.')

if __name__ == '__main__':
    main()
