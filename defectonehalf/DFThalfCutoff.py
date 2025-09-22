import logging
import os
import shutil
import numpy as np
import pandas as pd
from DFThalf4Vasp import VaspWrapperSimple
from DFThalf4Vasp.postprocessing import find_local_max_gap


class DFThalfCutoff:
    def __init__(self, AtomSelfEnPots, PotcarLoc, occband, unoccband, typevasprun='vasp_std',
                 bulkpotcarloc='', save_eigenval=True, save_doscar=False, run_in_ps_workdir=False,
                 save_to_workdir=True, is_bulk_calc=False, extrema_type='localmaximum', vasp_wrapper=None,
                 save_other_outputs=None):
        """
        Constructor of the the DFThalfCutoff class
        :param AtomSelfEnPots: list of potcarsetup objects
        :param PotcarLoc: list of potcar file location corresponding to potcarsetup object
        :param occband: list [index occupied band, spin] (up=1, down=2)
        :param unoccband: list [index unoccupied band, spin] (up=1, down=2)
        :param typevasprun: string with the bash command to run vasp
        :param bulkpotcarloc: string with location for bulk potcar. This is only needed for defect calculation an not bulk
        :param save_eigenval: if true eigenval files will be saved
        :param save_doscar:  if true doscar files will be saved
        :param run_in_ps_workdir: if true vasp will run in the workdir of potcarsetup
        :param save_to_workdir: if true files are saved in workdir/atomname of the current potcarsetup
        :param is_bulk_calc: if set to true, DFThalfCutoff will optimise the band gap instead of the gaps given by
        occband and unocc band.
        :param extrema_type: string with the type of extrema we're looking for. Option: extrema(default) or ext,
         maximum or max, minimum or min
         :param vasp_wrapper: vasp_wrapper object to interact with vasp on the system. There are some example
         vasp_wrappers in this project but you might need to make your own. Default VaspWrapperSimple, ase: VaspWrapperAse
         :param save_other_outputs: list containing the names of other files that need to be saved
        """
        # DFT-1/2 VARIABLES
        # list with potcarsetup objects of all the diffrent atoms.
        self.atoms_self_En_pots = AtomSelfEnPots
        self.potcar_loc = PotcarLoc # list of potcar location corresponding to each atom

        # Bulk potcar location, this parameter is only required for defect runs and should remain onchanged for bulk
        self.bulk_potcar_loc = bulkpotcarloc

        self.unoccband = unoccband      # list [index unoccupied band, spin, kpoint(optional)] (up=1, down=2)
        self.occband   = occband        # list [index occupied band  , spin, kpoint(optional)] (up=1, down=2)
        self.is_bulk_calc = is_bulk_calc  # if this is set to true the band gap will be calculated using pymatgen

        self.extrema_type = extrema_type
        #self.extrema_largest_rc_threshold = 3.5

        # VASP VARIABLES
        if vasp_wrapper == None:
            vasp_wrapper = VaspWrapperSimple.VaspWrapperSimple()
        self.vasp_wrapper = vasp_wrapper
        self.typevasprun   = typevasprun
        self.foldervasprun = None
        self.run_in_ps_workdir = run_in_ps_workdir # if true vasp will run in potcarsetup.workdir (this is not implemented)

        # EXTRA VARIABLES
        self.potcar_command_begin = bulkpotcarloc
        self.save_to_workdir = save_to_workdir
        self.save_eigenval = save_eigenval
        self.save_doscar   = save_doscar
        if save_other_outputs is None:
            save_other_outputs = []
        self.save_other_outputs = save_other_outputs



    def find_cutoff(self, rb, rf, nsteps_list, cut_func_par, numdecCut=3, extra_unaltered_pot=''):
        if not(isinstance(nsteps_list,type([])) ):
            nsteps_list = [nsteps_list]
        for i, pot_setup in enumerate(self.atoms_self_En_pots):
            print('Starting cutoff optimisation for ' + pot_setup.atomname, flush=True)
            # load dataframe from previous run if it exists
            csvfileloc = pot_setup.workdir + '/' + pot_setup.atomname + '/CutoffOpt.csv'
            # Check if csv file exists from previous run
            if os.path.isfile(csvfileloc):
                logging.debug('Loading previous cutoff data from ' + csvfileloc)
                cutoff_df = pd.read_csv(csvfileloc)
            else:
                cutoff_df = pd.DataFrame(columns=['Cutoff', 'Gap'])
            rb_atom = rb
            rf_atom = rf
            unalteredpotcars = ' '.join(self.potcar_loc[(i+1):]) + ' ' + extra_unaltered_pot
            for j, nsteps in enumerate(nsteps_list):
                if j != 0:
                    # Set begin and final radius for next loop
                    if rcext != 0:
                        rb_atom = RC[indext - 1]
                        rf_atom = RC[indext + 1]
                    elif rcext == 0:
                        rb_atom = 0
                        rf_atom = RC[indext + 1]
                    elif rcext == rf_atom:
                        raise Exception('The extreme gap is found at rc max!\n Increase rcext to get a proper gap')
                    else:
                        # this can happen when _find_extrema_gap does not find a gap.
                        raise Exception('An unexpected extremal cutoff was found')
                    print('Current extreme gap for ', pot_setup.atomname, ' is ', np.round(ext_gap,4), 'eV and was found at rc', rcext, ' a0',
                          flush=True)
                # Run single sweep
                RC = np.round(np.linspace(rb_atom, rf_atom, nsteps),numdecCut)
                new_cut_func_par = {
                                'Cutoff': RC,
                                'n': cut_func_par['n']
                }
                potcarfile = self.potcar_loc[i]
                cutoff_df, rcext , ext_gap, indext, RC = self.single_cutoff_sweep(pot_setup, potcarfile, new_cut_func_par, unalteredpotcars, cutoff_df=cutoff_df, numdecCut=numdecCut)

                # Save dataframe to csv file
                cutoff_df.to_csv(csvfileloc,index=False)
            # print maximal gap
            print('Extreme gap for ', pot_setup.atomname, ' is ', np.round(ext_gap,4),  'eV and was found at rc', rcext, ' a0',flush=True)
            # Copy potcar of maximum gap
            oldpotcaroptloc = pot_setup.workdir + '/' +pot_setup.atomname + '/POTCAR_DFThalf' + '/POTCAR_rc_' + str(np.round(rcext,numdecCut)) + '_n_' +  str(cut_func_par['n'])
            newpotcaroptloc = pot_setup.workdir + '/' +pot_setup.atomname + '/POTCAR_opt'
            shutil.copy(oldpotcaroptloc, newpotcaroptloc)
            self.potcar_command_begin += ' ' + newpotcaroptloc

    def single_cutoff_sweep(self, Vs_potsetup, potcarfile, CutFuncPar, unalterpotcars, cutoff_df=None, numdecCut=3):
        logging.debug('Starting single cutoff sweep')
        # Set up vasp run folder
        if self.foldervasprun is None or self.run_in_ps_workdir:
            logging.debug('Setting up vasp run folder')
            self.foldervasprun = Vs_potsetup.workdir + '/' + Vs_potsetup.atomname + '/Vasp_run'

        # Check if folder exists
        if not(os.path.isdir(self.foldervasprun)):
            logging.debug('Creating vasp run folder')
            os.makedirs(self.foldervasprun)

        # Check if dataframe is given
        if isinstance(cutoff_df,type(None)):
            logging.debug('Creating new dataframe')
            cutoff_df = pd.DataFrame(columns=['Cutoff', 'Gap'])

        # Get cutoff vector
        RC = CutFuncPar['Cutoff']
        # Make potcars
        Vs_potsetup.make_potcar(potcarfile, CutFuncPar)

        for rc in RC:
            # If the current rc value is already present we will skip it
            if cutoff_df['Cutoff'].isin([rc]).any():
                logging.debug('Skipping rc value: %f as it is already present in the dataframe' % rc)
                continue

            # Run vasp
            logging.debug('Running vasp for rc: %f' % rc)
            newpotcarfileloc = Vs_potsetup.workdir + '/' +Vs_potsetup.atomname + '/POTCAR_DFThalf' + '/POTCAR_rc_' + str(np.round(rc,numdecCut)) + '_n_' +  str(CutFuncPar['n'])
            self._run_vasp(newpotcarfileloc, unalterpotcars)
            # Calculate gap
            gap = self._calculate_gap()
            # print result
            print('Rc: ', rc, ' Gap: ', np.round(gap,4), flush=True)
            # save result
            current_cutoff = pd.DataFrame([[rc,gap]],columns=['Cutoff','Gap'])
            cutoff_df = pd.concat([cutoff_df,current_cutoff], ignore_index=True)
            # Save files
            logging.debug('Saving vasp output files')
            self.save_vasp_output_files(Vs_potsetup,rc,numdecCut,CutFuncPar)


        # rc_cutoff_df makes sure the given data frame only contains gaps corresponding to the current RC.
        # Without this we might find a maximum gap that does not correspond to our RC which will mess up the next step.
        # This only makes a difference when starting from a previous run
        rc_cutoff_df = cutoff_df[cutoff_df['Cutoff'].isin(RC)]
        rc_cutoff_df = rc_cutoff_df.sort_values('Cutoff',axis=0) # sort to maintain the order of RC
        rc_cutoff_df = rc_cutoff_df.reset_index(drop=True)
        # Find extrema
        try:
            rcext, ext_gap, indext = self.get_rext_gap(rc_cutoff_df)
        except Warning:
            logging.warning('No extrema was found! rcext ext_gap and index are set to None.')
            rcext = None
            ext_gap = None
            indext = None

        return cutoff_df, rcext, ext_gap, indext, RC

    def _run_vasp(self, currentpotcar, unalteredpotcars):
        # Make POTCAR for vasp run
        self._make_vasp_run_potcar(currentpotcar, unalteredpotcars)
        # Run vasp
        self.vasp_wrapper.run_vasp(self.foldervasprun, self.typevasprun)


    def _make_vasp_run_potcar(self, currentpotcar, unalteredpotcars):
        # Makes the potcar for the actual vasp by concatenating DFT-1/2 potcars
        Makepotcarcommand = 'cat ' + self.potcar_command_begin + ' ' + currentpotcar + ' ' + unalteredpotcars + ' > POTCAR'
        os.system(Makepotcarcommand)

    def _calculate_gap(self):
        if self.is_bulk_calc:
            gap = self.vasp_wrapper.calculate_bandgap(foldervasprun=self.foldervasprun)
        else:
            bands = [self.occband[0], self.unoccband[0]]
            spins = [self.occband[1], self.unoccband[1]]
            if len(self.occband) > 2 and len(self.unoccband) > 2:
                kpoints = [self.occband[2], self.unoccband[2]] # Make kpoints list if given by user
            else:
                kpoints = [0, 0] # default: Gamma point/ kp=0 gap

            gap = self.vasp_wrapper.calculate_gap(bands,spins, kpoints=kpoints, foldervasprun=self.foldervasprun)
        return gap

    def save_vasp_output_files(self,Vs_potsetup,rc,numdecCut,CutFuncPar):
        if self.save_to_workdir:
            save_folder = Vs_potsetup.workdir + '/' + Vs_potsetup.atomname
        else:
            save_folder = self.foldervasprun
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        if self.save_eigenval:
            # Check if folder exist
            if not os.path.isdir(save_folder+ '/EIGENVALS'):
                os.makedirs(save_folder+ '/EIGENVALS')
            # copy file
            shutil.copy(self.foldervasprun + '/EIGENVAL',
                        save_folder +'/EIGENVALS/EIGENVAL' + '_rc_' + str(np.round(rc, numdecCut)) + '_n_' + str(CutFuncPar['n']))
        if self.save_doscar:
            if not os.path.isdir(save_folder + '/DOSCARS'):
                os.makedirs(save_folder + '/DOSCARS')
            shutil.copy(self.foldervasprun + '/DOSCAR',
                        save_folder + '/DOSCARS/DOSCAR' + '_rc_' + str(np.round(rc, numdecCut)) + '_n_' + str(CutFuncPar['n']))

        for file in self.save_other_outputs:
            if not os.path.isdir(save_folder + '/other_outputs'):
                os.makedirs(save_folder + '/other_outputs')
            shutil.copy(self.foldervasprun + '/' + file,
                        save_folder + '/other_outputs/' + file + '_rc_' + str(np.round(rc, numdecCut)) + '_n_' + str(CutFuncPar['n']))

    def get_rext_gap(self, rc_cutoff_df):
        if self.extrema_type == 'extrema' or self.extrema_type == 'ext':
            return self._find_extrema_gap(rc_cutoff_df)
        elif self.extrema_type == 'maximum' or self.extrema_type == 'max':
            indext = rc_cutoff_df.iloc[:, 1].idxmax()
            rcext = rc_cutoff_df.iloc[indext, 0]
            ext_gap = rc_cutoff_df.iloc[indext, 1]
            return rcext, ext_gap, indext
        elif self.extrema_type == 'local max' or self.extrema_type == 'local maximum' or self.extrema_type == 'localmaximum':
            return self._find_local_max_gap(rc_cutoff_df)
        elif self.extrema_type == 'minimum' or self.extrema_type == 'min':
            indext = rc_cutoff_df.iloc[:, 1].idxmin()
            rcext  = rc_cutoff_df.iloc[indext, 0]
            ext_gap = rc_cutoff_df.iloc[indext, 1]
            return rcext, ext_gap, indext
        else:
            raise Warning('Unknown extrema_type was given!\nextrema_type was set to local max!')
            self.extrema_type = 'local max'
            return self._find_local_max_gap(rc_cutoff_df)


    def _find_extrema_gap(self, rc_cutoff_df):
        """
        finds the extremal gap in rc_cutoff_df and return the rc, gap and index of this extremum
        :param rc_cutoff_df:
        :return:
        """
        # Find max
        indmax  = rc_cutoff_df.iloc[:, 1].idxmax()
        rcmax   = rc_cutoff_df.iloc[indmax, 0]
        max_gap = rc_cutoff_df.iloc[indmax, 1]

        # Find min
        indmin  = rc_cutoff_df.iloc[:, 1].idxmin()
        rcmin   = rc_cutoff_df.iloc[indmin, 0]
        min_gap = rc_cutoff_df.iloc[indmin, 1]

        # rc proprties
        largest_rc  = rc_cutoff_df.iloc[:, 0].max()
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

    def _find_local_max_gap(self, rc_cutoff_df):
        return find_local_max_gap(rc_cutoff_df)

