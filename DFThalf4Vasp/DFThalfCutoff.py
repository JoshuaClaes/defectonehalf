import os
import shutil
import numpy as np
import pandas as pd

class DFThalfCutoff:
    def __init__(self,AtomSelfEnPots,PotcarLoc,occband,unoccband,typevasprun='vasp_std',
                 bulkpotcarloc='',save_eigenval=True ,save_doscar=False, run_in_ps_workdir=False,
                 save_to_workdir=True):
        # DFT-1/2 VARIABLES
        # list with potcarsetup objects of all the diffrent atoms.
        self.atoms_self_En_pots = AtomSelfEnPots
        self.potcar_loc = PotcarLoc     # list of potcar location corresponding to each atom
        self.bulk_potcar_loc = bulkpotcarloc # Bulk potcar location, this parameter is only required for defect runs and should remain onchanged for bulk

        self.unoccband = unoccband  # list [index unoccupied band, spin] (up=1, down=2)
        self.occband   = occband    # list [index occupied band  , spin] (up=1, down=2)

        # VASP VARIABLES
        self.typevasprun   = typevasprun
        self.foldervasprun = None
        self.run_in_ps_workdir = run_in_ps_workdir # if true vasp will run in potcarsetup.workdir

        # EXTRA VARIABLES
        self.potcar_command_begin = bulkpotcarloc
        self.save_to_workdir = save_to_workdir
        self.save_eigenval = save_eigenval
        self.save_doscar   = save_doscar



    def find_cutoff(self, rb, rf, nsteps_list, cut_func_par, numdecCut=3, extra_unaltered_pot=''):
        if not(isinstance(nsteps_list,type([])) ):
            nsteps_list = [nsteps_list]
        for i,pot_setup in enumerate(self.atoms_self_En_pots):
            print('Starting cutoff optimisation for ' + pot_setup.atomname, flush=True)
            cutoff_df = pd.DataFrame(columns=['Cutoff', 'Gap'])
            rb_atom = rb
            rf_atom = rf
            unalteredpotcars = ' '.join(self.potcar_loc[(i+1):]) + ' ' + extra_unaltered_pot
            for j, nsteps in enumerate(nsteps_list):
                if j != 0:
                    # Set begin and final radius for next loop
                    if rcmax != 0:
                        rb_atom = RC[indmax - 1]
                        rf_atom = RC[indmax + 1]
                    elif rcmax == 0:
                        rb_atom = 0
                        rf_atom = RC[indmax + 1]
                    else:
                        # this should never happen
                        raise Exception('An unexpected maximum cutoff was found')
                    print('Current maximum gap for ', pot_setup.atomname, ' is ', np.round(gapmax,4), 'eV and was found at rc', rcmax, ' a0',
                          flush=True)
                # Run single sweep
                RC = np.round(np.linspace(rb_atom, rf_atom, nsteps),numdecCut)
                new_cut_func_par = {
                                'Cutoff': RC,
                                'n': cut_func_par['n']
                }
                potcarfile = self.potcar_loc[i]
                newcutoff_df,rcmax , gapmax,indmax, RC = self.single_cutoff_sweep(pot_setup, potcarfile, new_cut_func_par, unalteredpotcars, cutoff_df=cutoff_df, numdecCut=numdecCut)

                # Update cutoff_df
                # cutoff_df.append(newcutoff_df)

                # Save dataframe to csv file
                csvfileloc =  pot_setup.workdir + '/' +pot_setup.atomname + '/CutoffOpt.csv'
                cutoff_df.to_csv(csvfileloc)
            # print maximal gap
            print('Maximum gap for ', pot_setup.atomname, ' is ', np.round(gapmax,4),  'eV and was found at rc', rcmax, ' a0',flush=True)
            # Copy potcar of maximum gap
            oldpotcaroptloc = pot_setup.workdir + '/' +pot_setup.atomname + '/POTCAR_DFThalf' + '/POTCAR_rc_' + str(np.round(rcmax,numdecCut)) + '_n_' +  str(cut_func_par['n'])
            newpotcaroptloc = pot_setup.workdir + '/' +pot_setup.atomname + '/POTCAR_opt'
            shutil.copy(oldpotcaroptloc, newpotcaroptloc)
            self.potcar_command_begin += ' ' + newpotcaroptloc

    def single_cutoff_sweep(self, Vs_potsetup, potcarfile, CutFuncPar, unalterpotcars, cutoff_df=None, numdecCut=3):
        if self.foldervasprun == None or self.run_in_ps_workdir:
            self.foldervasprun = Vs_potsetup.workdir + '/' + Vs_potsetup.atomname + '/Vasp_run'
        if not(os.path.isdir(self.foldervasprun)):
            os.makedirs(self.foldervasprun)
        if isinstance(cutoff_df,type(None)):
            cutoff_df = pd.DataFrame(columns=['Cutoff', 'Gap'])

        # Get cutoff vector
        RC = CutFuncPar['Cutoff']
        # Make potcars
        Vs_potsetup.make_potcar(potcarfile, CutFuncPar)

        for rc in RC:
            # If the current rc value is already present we will skip it
            if cutoff_df['Cutoff'].isin([rc]).any():
                continue

            # Run vasp
            newpotcarfileloc = Vs_potsetup.workdir + '/' +Vs_potsetup.atomname + '/POTCAR_DFThalf' + '/POTCAR_rc_' + str(np.round(rc,numdecCut)) + '_n_' +  str(CutFuncPar['n'])
            self.run_vasp(newpotcarfileloc, unalterpotcars)
            # Calculate gap
            eigenval_loc = self.foldervasprun + '/EIGENVAL'
            gap = self.calculate_gap(eigenval_loc)
            # print result
            print('Rc: ', rc, ' Gap: ', np.round(gap,4), flush=True)
            # save result
            current_cutoff = pd.DataFrame([[rc,gap]],columns=['Cutoff','Gap'])
            cutoff_df = pd.concat([cutoff_df,current_cutoff])
            # Save files
            self.save_vasp_output_files(Vs_potsetup,rc,numdecCut,CutFuncPar)

        # Find max
        indmax = cutoff_df.iloc[:, 1].idxmax()
        rcmax  = cutoff_df.iloc[indmax, 0]
        Gapmax = cutoff_df.iloc[indmax, 1]
        return cutoff_df,rcmax,Gapmax,indmax,RC

    def run_vasp(self, currentpotcar, unalteredpotcars):
        # Go vasp run directory
        oldpath = os.getcwd()
        os.chdir(self.foldervasprun)
        # Make POTCAR for vasp run
        self.make_vasp_run_potcar(currentpotcar, unalteredpotcars)
        # Copy INCAR, KPOINTS and POSCAR file

        # Run vasp
        if self.typevasprun == 'vasp_std' or self.typevasprun =='std':
            os.system('srun vasp_std')
        elif self.typevasprun == 'vasp_gam' or self.typevasprun =='gam':
            os.system('srun vasp_gam')
        elif self.typevasprun == 'vasp_ncl' or self.typevasprun =='ncl':
            os.system('srun vasp_ncl')
        else:
            # incase another type is given we try to run the given string
            os.system(self.typevasprun)
        # Go back to original path
        os.chdir(oldpath)

    def make_vasp_run_potcar(self, currentpotcar, unalteredpotcars):
        # Makes the potcar for the actual vasp by concatenating DFT-1/2 potcars
        Makepotcarcommand = 'cat ' + self.potcar_command_begin + ' ' + currentpotcar + ' ' + unalteredpotcars + ' > POTCAR'
        os.system(Makepotcarcommand)

    def calculate_gap(self, EIGENVALfileloc):
        # spinlb: Spin lowest band (up=1, down=2)
        # spinhb: Spin highest band (up=1, down=2)

        ilb     = self.occband[0] # index lowest band
        spinlb  = self.occband[1] # Spin lowest band (up=1, down=2)
        ihb     = self.unoccband[0]   # index highest band
        spinhb  = self.unoccband[1]   # Spin highest band (up=1, down=2)

        # Read eigenvalues
        eign = pd.read_csv(EIGENVALfileloc, delim_whitespace=True, skiprows=8, header=None)
        # Calculate gap
        Gap = eign.iloc[ihb, spinhb] - eign.iloc[ilb, spinlb];
        return Gap

    def save_vasp_output_files(self,Vs_potsetup,rc,numdecCut,CutFuncPar):
        if self.save_to_workdir:
            save_folder = Vs_potsetup.workdir
        else:
            save_folder = self.foldervasprun
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        if self.save_eigenval:
            shutil.copy(save_folder + '/EIGENVAL',
                        'EIGENVALS/EIGENVAL' + '_rc_' + str(np.round(rc, numdecCut)) + '_n_' + str(CutFuncPar['n']))
        if self.save_doscar:
            shutil.copy(save_folder + '/DOSCAR',
                        'DOSCARS/DOSCAR' + '_rc_' + str(np.round(rc, numdecCut)) + '_n_' + str(CutFuncPar['n']))
