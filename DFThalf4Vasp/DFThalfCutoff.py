import os
import shutil
import numpy as np
import pandas as pd

class DFThalfCutoff:
    def __init__(self,AtomSelfEnPots,PotcarLoc,occband,unoccband,typevasprun='vasp_std', bulkpotcarloc=''):
        # DFT-1/2 VARIABLES
        # list with potcarsetup objects of all the diffrent atoms.
        self.AtomSelfEnPots = AtomSelfEnPots
        self.PotcarLoc = PotcarLoc     # list of potcar location corresponding to each atom
        self.BulkPotcarLoc = bulkpotcarloc # Bulk potcar location, this parameter is only required for defect runs and should remain onchanged for bulk

        self.unoccband = unoccband  # list [index unoccupied band, spin] (up=1, down=2)
        self.occband   = occband    # list [index occupied band  , spin] (up=1, down=2)

        # VASP VARIABLES
        self.typevasprun   = typevasprun
        self.foldervasprun = None

        # EXTRA VARIABLES
        self.PotcarCommandBegin = bulkpotcarloc


    def FindCutoff(self,rb,rf,nsteps_list,CutFuncPar, numdecCut=3, extraunaltpot=''):
        if not(isinstance(nsteps_list,type([])) ):
            nsteps_list = [nsteps_list]
        for i,Potsetup in enumerate(self.AtomSelfEnPots):
            print('Starting cutoff optimisation for ' + Potsetup.atomname, flush=True)
            cutoff_df = pd.DataFrame(columns=['Cutoff', 'Gap'])
            rb_atom = rb
            rf_atom = rf
            unalteredpotcars = ''.join(self.PotcarLoc[i:-1]) + extraunaltpot
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
                    print('Current maximum gap for ', Potsetup.atomname, ' is ', np.round(Gapmax,4), 'eV and was found at rc', rcmax, ' a0',
                          flush=True)
                # Run single sweep
                RC = np.round(np.linspace(rb_atom, rf_atom, nsteps),numdecCut)
                newCutFuncPar = {
                                'Cutoff': RC,
                                'n': CutFuncPar['n']
                }
                potcarfile = self.PotcarLoc[i]
                newcutoff_df,rcmax , Gapmax,indmax, RC = self.SingleCutoffSweep(Potsetup,potcarfile, newCutFuncPar, unalteredpotcars, cutoff_df=cutoff_df, numdecCut=numdecCut)

                # Update cutoff_df
                # cutoff_df.append(newcutoff_df)

                # Save dataframe to csv file
                csvfileloc =  Potsetup.workdir + '/' +Potsetup.atomname + '/CutoffOpt.csv'
                cutoff_df.to_csv(csvfileloc)
            # print maximal gap
            print('Maximum gap for ', Potsetup.atomname, ' is ', np.round(Gapmax,4),  'eV and was found at rc', rcmax, ' a0',flush=True)
            # Copy potcar of maximum gap
            oldpotcaroptloc = Potsetup.workdir + '/' +Potsetup.atomname + '/POTCAR_DFThalf' + '/POTCAR_rc_' + str(np.round(rcmax,numdecCut)) + '_n_' +  str(CutFuncPar['n'])
            newpotcaroptloc = Potsetup.workdir + '/' +Potsetup.atomname + '/POTCAR_opt'
            shutil.copy(oldpotcaroptloc, newpotcaroptloc)
            self.PotcarCommandBegin += ' ' + newpotcaroptloc

    def SingleCutoffSweep(self, Vs_potsetup, potcarfile, CutFuncPar, unalterpotcars,cutoff_df=None, numdecCut=3):
        if self.foldervasprun == None:
            self.foldervasprun = Vs_potsetup.workdir + '/' + Vs_potsetup.atomname + '/Vasp_run'
        if not(os.path.isdir(self.foldervasprun)):
            os.makedirs(self.foldervasprun)
        if isinstance(cutoff_df,type(None)):
            cutoff_df = pd.DataFrame(columns=['Cutoff', 'Gap'])

        # Get cutoff vector
        RC = CutFuncPar['Cutoff']
        # Make potcars
        Vs_potsetup.MakePotcar(potcarfile, CutFuncPar)

        for rc in RC:
            # If the current rc value is already present we will skip it
            if cutoff_df['Cutoff'].isin([rc]).any():
                continue

            # Run vasp
            newpotcarfileloc = Vs_potsetup.workdir + '/' +Vs_potsetup.atomname + '/POTCAR_DFThalf' + '/POTCAR_rc_' + str(np.round(rc,numdecCut)) + '_n_' +  str(CutFuncPar['n'])
            self.RunVasp(newpotcarfileloc,unalterpotcars)
            # Calculate gap
            EIGENVALloc = self.foldervasprun + '/EIGENVAL'
            Gap = self.CalculateGap(EIGENVALloc)
            # print result
            print('Rc: ', rc, ' Gap: ', np.round(Gap,4), flush=True)
            # save result
            current_cutoff = pd.DataFrame([[rc,Gap]],columns=['Cutoff','Gap'])
            cutoff_df = pd.concat([cutoff_df,current_cutoff])

        # Find max
        indmax = cutoff_df.iloc[:, 1].idxmax()
        rcmax  = cutoff_df.iloc[indmax, 0]
        Gapmax = cutoff_df.iloc[indmax, 1]
        return cutoff_df,rcmax,Gapmax,indmax,RC

    def RunVasp(self,currentpotcar,unalteredpotcars):
        # Go vasp run directory
        oldpath = os.getcwd()
        os.chdir(self.foldervasprun)
        # Make POTCAR for vasp run
        self.MakeVaspRunPotcar(currentpotcar,unalteredpotcars)
        # Copy INCAR, KPOINTS and POSCAR file

        # Run vasp
        if self.typevasprun == 'vasp_std' or self.typevasprun =='std':
            os.system('vasp_std')
        elif self.typevasprun == 'vasp_gam' or self.typevasprun =='gam':
            os.system('vasp_gam')
        elif self.typevasprun == 'vasp_ncl' or self.typevasprun =='ncl':
            os.system('vasp_ncl')
        else:
            # incase another type is given we try to run the given string
            os.system(self.typevasprun)
        # Go back to original path
        os.chdir(oldpath)

    def MakeVaspRunPotcar(self,currentpotcar,unalteredpotcars):
        # Makes the potcar for the actual vasp by concatenating DFT-1/2 potcars
        Makepotcarcommand = 'cat ' + self.PotcarCommandBegin + ' ' + currentpotcar + ' '  +unalteredpotcars + ' > POTCAR'
        os.system(Makepotcarcommand)

    def CalculateGap(self,EIGENVALfileloc):
        # spinlb: Spin lowest band (up=1, down=2)
        # spinhb: Spin highest band (up=1, down=2)

        ilb     = self.unoccband[0] # index lowest band
        spinlb  = self.unoccband[1] # Spin lowest band (up=1, down=2)
        ihb     = self.occband[0]   # index highest band
        spinhb  = self.occband[1]   # Spin highest band (up=1, down=2)

        # Read eigenvalues
        eign = pd.read_csv(EIGENVALfileloc, delim_whitespace=True, skiprows=8, header=None)
        # Calculate gap
        Gap = eign.iloc[ihb, spinhb] - eign.iloc[ilb, spinlb];
        return Gap

