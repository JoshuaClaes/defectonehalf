from typing import List
import pickle
import logging
import shutil
import numpy as np
import pymatgen.io.vasp as pmg
from pymatgen.core import Structure
from pymatgen.electronic_structure.core import Spin
from DFThalf4Vasp.preprocessing import get_largest_contributors, calc_electron_fraction
from DFThalf4Vasp.preprocessing import setup_calculation
from DFThalf4Vasp.preprocessing import make_defect_poscar
from DFThalf4Vasp.DFThalfCutoff import DFThalfCutoff
import DFThalf4Vasp.potcarsetup as ps
from DFThalf4Vasp.orbital import Orbital

class Orb_info:
    def __init__(self, element, core_val_orbs, GSocc):
        """
        element: string with chemical symbol of element
        core_val: list with number of core and valence electrons [core_electons, valence_electrons]
        GS_occ: List with Orbital objects for each of the valence_electron
        """
        # Some simple checks on the input parameters
        if isinstance(element,str) and isinstance(core_val_orbs, list) and isinstance(GSocc, list):
            self.element = element
            self.core_val= core_val_orbs
            self.GSocc   = GSocc
        else:
            raise Exception('Input of Orb_info is not correct!')


def analysis_defect_setup_calc(folder: str, def_bands, vbm_ind: int, cbm_ind: int,
                               orb_info_sc: List[Orb_info], workdir_self_en: str, threshold_defect_atoms: float = 0.005,
                               efrac_threshold: float = None,
                               decoupled_run: bool = False, EXtype: str = 'ca', typepotcarfile: str = 'lda',
                               cutfuncpar=None,
                               bulk_potcar: str = '../POTCAR_bulk ', typevasprun: str = 'vasp_gam',
                               save_eigenval: bool = True,
                               save_doscar: bool = False, rb: float = 0.0, rf: float = 4.0, nsteps: List[int] = [9, 11],
                               job_script_header: str = '', job_script_footer: str = '',
                               job_script_name: str = 'job_script.slurm', set_num_groups=None, print_output=True,
                               incar_loc=None, kpoints_loc=None, potcar_loc_base=None, dry_run = False,
                               fullworkdirpath = False) -> None:
    """
    Function to perform analysis of defect setup calculations.

    Parameters:
    folder (str): Folder with DFT of bulk DFT-1/2 calculation.
    def_bands (List[List[int, str]]): List with list of defect bands. Example: [[[1024],'up'],[[1024],'down']]
    vbm_ind (int): Index of the VBM in the bulk calculation.
    cbm_ind (int): Index of the CBM in the bulk calculation.
    orb_info_sc (List[Orb_info]): List of Orb_info objects for each element in the unit cell.
    workdir_self_en (str): Working directory for self energy.

    Optional parameters
    # Defect atoms parameters
    threshold_defect_atoms (float): Threshold for determining defect atoms. Default is 0.005.
    set_num_groups (int): the number of groups you want returned. This is for a second run where you want less
    groups than in the default case.
    dry_run (bool): if True no calculations will be performed i.e. no potcars will be generated only the analysis is
    performed. Default is False
    efrac_threshold (float): Threshold for electron fraction. Default is None.
    fullworkdirpath (bool): if True the full path to the workdir will be used. Default is False.

    # type of DFT-1/2 run
    decoupled_run (bool): if True a decoupled calculation will be setup, if false a conventional calculation will be
    setup. Default is False

    # Self energy parameter
    EXtype (str): Type of self energy. Default is 'ca'.
    typepotcarfile (str): Name of the POTCAR file. Default is 'lda'.
    cutfuncpar (dict): Allows to pregenerate more potcar file by changing the Cutoff list or change the power of the
    trimming function n.

    # DFThalfCutoff parameters
    bulk_potcar (str): String with all POTCARs which are not altered in the cutoff optimization. Default is '../POTCAR_bulk '.
    typevasprun (str): Type of VASP run. Default is 'vasp_gam'.
    save_eigenval (bool): Whether to save the eigenvalues file. Default is True.
    save_doscar (bool): Whether to save the DOS file. Default is False.
    rb (float): Cutoff begin. Default is 0.0.
    rf (float): Cutoff final. Default is 4.0.
    nsteps (List[int]): List of number of points for each sweep. Default is [9, 11].

    # Job script parameters
    job_script_header (str): Header for the job script. Default is an empty string.
    job_script_footer (str): Footer for the job script. Default is an empty string.
    job_script_name (str): Name of the job script. Default is 'job_script.slurm'.

    # Extra parameters
    print_output (bool): if True print some intermediate results
    incar_loc (string): location of INCAR file for calculation. Defeault None. If None INCAR will be copied from folder
    kpoints_loc (string): location of KPOINTS file for calculation. Defeault None. If None KPOINTS will be copied from folder

    Returns:
    None
    """

    if cutfuncpar is None:
        cutfuncpar = {'Cutoff': [0], 'n': 8}

    #####################
    # Load calculation
    #####################

    structure = Structure.from_file(folder + '/POSCAR')
    try:
        vasprun = pmg.Vasprun(folder + '/vasprun.xml', parse_potcar_file=False, parse_projected_eigen=True)
        projected_eign = vasprun.projected_eigenvalues
    except:
        print('Something went wrong while reading vasprun.xml. Make sure LORBIT=10 is in the INCAR file!')

    #####################
    # Find defect atoms
    #####################
    '''
    We'll start by only considering the 2 largest groups. If the second group has a xi or zeta < 0.01 they'll only the first group will be
    considered as defect atoms. If this is not the case we'll add another group until we find a group n for which xi and zeta<0.01
    When we have this group we keep the n-1 groups with a xi and zeta >= 0.01
    '''
    # Occupied bands
    band_ind = def_bands[0][0]
    if def_bands[0][1] == 'up':
        band_spin = Spin.up
    elif def_bands[0][1] == 'down':
        band_spin = Spin.down

    logging.debug('Finding defect groups for xi')
    defect_groups_xi, xi, elem_xi = _find_def_atoms(projected_eign, band_ind, band_spin, structure,
                                                    threshold_int=threshold_defect_atoms,
                                                    set_num_groups=set_num_groups)

    # Unoccupied bands
    band_ind = def_bands[1][0]
    if def_bands[1][1] == 'up':
        band_spin = Spin.up
    elif def_bands[1][1] == 'down':
        band_spin = Spin.down

    logging.debug('Finding defect groups for zeta')
    defect_groups_zeta, zeta, elem_zeta = _find_def_atoms(projected_eign, band_ind, band_spin, structure,
                                                          threshold_int=threshold_defect_atoms,
                                                          set_num_groups=set_num_groups)

    # We should now combine the groups of xi and zeta. Since there not necessarily the same
    all_defect_groups = defect_groups_xi.copy()
    xi_all_groups = xi.copy()
    elem_all_groups = elem_xi.copy()

    zeta_all_groups = np.zeros([len(all_defect_groups), 3])

    for i, group_zeta in enumerate(defect_groups_zeta):
        group_zeta_not_found = True
        # check if group zeta is in all_defect_groups
        for j, group in enumerate(all_defect_groups):
            if group == group_zeta:
                zeta_all_groups[j] = zeta[i]  # add zeta at the index were group is found
                group_zeta_not_found = False  # We found group_zeta in all groups
                # Look for next group_zeta since we found this one
                break

        if group_zeta_not_found:
            # In this case we need to add this group to all_defect_groups
            all_defect_groups.append(group_zeta)
            xi_all_groups = np.concatenate((xi_all_groups, [np.zeros(3)]))
            zeta_all_groups = np.concatenate((zeta_all_groups, [zeta[i]]))
            elem_all_groups.append(elem_zeta[i])

    if efrac_threshold is not None:
        xi_all_groups, zeta_all_groups, all_defect_groups, elem_all_groups = _apply_efrac_threshold(
        xi_all_groups, zeta_all_groups, all_defect_groups, elem_all_groups, efrac_threshold
        )

    # Check if the selected number of groups was found
    if set_num_groups is not None:
        if len(all_defect_groups) < set_num_groups:
            logging.warning(
                f'Only {len(all_defect_groups)} groups were found by analysis_defect_setup_calculation. '
                f'Expected {set_num_groups} groups. Consider raising the threshold_defect_atoms parameter.'
                f' Continuing with {len(all_defect_groups)} groups.')
        elif len(all_defect_groups) > set_num_groups:
            logging.warning(
                f'More than {set_num_groups} groups were found by analysis_defect_setup_calculation. '
                f'The {set_num_groups} with the highest contribution will be used for further calculations.'
                f'This warning can occur when the {set_num_groups} largest contribution groups found by _find_def_atoms'
                f' are different for xi and zeta!')
            # log warning with old xi and zeta
            logging.warning(f'Old groups: {all_defect_groups}\nOld xi: {xi_all_groups} \nOld zeta: {zeta_all_groups}')
            # Add xi and zeta contributions and sort by total contribution then take the first set_num_groups
            kwargs = {'xi_all_groups': xi_all_groups, 'zeta_all_groups': zeta_all_groups,
                      'all_defect_groups': all_defect_groups, 'elem_all_groups': elem_all_groups,
                      'set_num_groups': set_num_groups}
            xi_all_groups, zeta_all_groups, all_defect_groups, elem_all_groups = _select_top_groups(**kwargs)

    # As extra information we print the total electron fraction removed. This can somewhat deviate from 0.50 because of rounding errors
    # Calculate multiplicity for each group
    multiplicity = np.array([len(group) for group in all_defect_groups])[:, np.newaxis]

    # Calculate total contributions considering multiplicity
    xi_sum = np.sum(xi_all_groups * multiplicity)
    zeta_sum = np.sum(zeta_all_groups * multiplicity)

    logging.debug(f'Sum of removed electron fractions:\nxi:\t{xi_sum}\nzeta:\t{zeta_sum}')
    if np.abs(0.5-xi_sum) > 0.05 or np.abs(0.5-zeta_sum) > 0.05:
        logging.warning(f'Electron fraction of xi or zeta deviates significantly(more than 0.05) from 0.5.\nCheck your output carefully!')
        # Try to renormalize xi and zeta
        logging.warning('Renormalizing xi and zeta')
        xi_all_groups, zeta_all_groups = renormalize_efracs(xi_all_groups, zeta_all_groups,all_defect_groups)
        xi_sum = np.sum(xi_all_groups * multiplicity)
        zeta_sum = np.sum(zeta_all_groups * multiplicity)
        logging.warning(f'New sum of removed electron fractions:\nxi:\t{xi_sum}\nzeta:\t{zeta_sum}')


    if print_output:
        print('======================\nInfo xi\n======================')
        print(f'Elements defect groups:\n{elem_all_groups} \nIndices defect atoms:\n{all_defect_groups} \nxi\n{xi_all_groups}')
        print('\n======================\nInfo zeta\n======================')
        print(f'Elements defect groups:\n{elem_all_groups} \nIndices defect atoms:\n{all_defect_groups} \nzeta\n{zeta_all_groups}')

    if not (dry_run):
        #####################
        # The self energy
        #####################
        group_names = []
        orbitals = []
        GSorbs = []
        # These are extra input parameters given at the beginning
        # EXtype  = 'ca'
        # potcarfile = 'lda'

        for i, group in enumerate(all_defect_groups):
            # group name indexofgroup_element_numberofelement. Index of group is such that all created folders are ordered
            element = elem_all_groups[i]
            group_name = str(i + 1) + '_' + element + '_' + str(len(group))
            group_names.append(group_name)  # Group gets name of element it contains

            # Get orbitals and GSorbs which is required for setup_calculation
            # This is somewhat tricky because this depends from atoms to atoms and on the specific potcar that is being use and
            # thus the user should supply this information for each atom in the unit cell

            # We loop over all elements until we find a matching element
            for o, orb in enumerate(orb_info_sc):
                if element == orb.element:
                    orbitals.append(orb.core_val)
                    GSorbs.append(orb.GSocc)
                    break
                if o == (len(orb_info_sc) - 1):
                    raise Exception(
                        f'Some defect atoms are {element} which is an element which is not present in orb_info_sc!')


        if decoupled_run:
            _setup_decoupled_runs(folder, workdir_self_en, xi_all_groups, zeta_all_groups, group_names, elem_all_groups, orbitals,
                                  GSorbs, EXtype, typepotcarfile, cutfuncpar, all_defect_groups, def_bands, vbm_ind, cbm_ind,
                                  typevasprun, bulk_potcar, save_eigenval, save_doscar, rb, rf, nsteps, job_script_name,
                                  job_script_header, job_script_footer, incar_loc=incar_loc, kpoints_loc=kpoints_loc,
                                  potcar_loc_base=potcar_loc_base, fullworkdirpath=fullworkdirpath)
        else:
            _setup_conventional_run(folder, workdir_self_en, xi_all_groups, zeta_all_groups, group_names, elem_all_groups, orbitals,
                                    GSorbs, EXtype, typepotcarfile, cutfuncpar, all_defect_groups, def_bands,
                                    typevasprun, bulk_potcar, save_eigenval, save_doscar, rb, rf, nsteps, job_script_name,
                                    job_script_header, job_script_footer, incar_loc=incar_loc, kpoints_loc=kpoints_loc,
                                    potcar_loc_base=potcar_loc_base, fullworkdirpath=fullworkdirpath)


#################################
# HELPER FUNCTIONS
#################################
def _setup_conventional_run(folder, workdir_self_en, xi_all_groups, zeta_all_groups, group_names, elem_all_groups, orbitals,
                            GSorbs, EXtype, typepotcarfile, cutfuncpar, all_defect_groups, def_bands,
                            typevasprun, bulk_potcar, save_eigenval, save_doscar, rb, rf, nsteps, job_script_name,
                            job_script_header, job_script_footer, incar_loc = None, kpoints_loc = None,
                            potcar_loc_base = None, fullworkdirpath = False):
    """
    This function sets up a conventional run for the DFT-1/2 method.

    Parameters:
    folder (str): The directory containing the DFT of bulk DFT-1/2 calculation.
    workdir_self_en (str): The working directory for self energy.
    xi_all_groups (np.array): The electron fractions to be removed for each group of atoms contributing to the defect orbitals.
    zeta_all_groups (np.array): The electron fractions to be added for each group of atoms contributing to the defect orbitals.
    group_names (list): The names of the groups of atoms contributing to the defect orbitals.
    elem_all_groups (list): The elements of the groups of atoms contributing to the defect orbitals.
    orbitals (list): The orbitals for each element in the unit cell.
    GSorbs (list): The ground state occupation of the valence orbitals for each element in the unit cell.
    EXtype (str): The type of exchange correlation used in atom.
    typepotcarfile (str): The name of the POTCAR file.
    cutfuncpar (dict): The for the cutoff function in DFT-1/2 method.
    all_defect_groups (list): The indices of the atoms in the defect groups.
    def_bands (list): The indices of the defect bands.
    typevasprun (str): The type of VASP run.
    bulk_potcar (str): The location of the POTCAR file for the bulk calculation.
    save_eigenval (bool): Whether to save the eigenvalues file.
    save_doscar (bool): Whether to save the DOS file.
    rb (float): The beginning of the cutoff.
    rf (float): The end of the cutoff.
    nsteps (list): List with number of points for each step.
    job_script_name (str): The name of the job script.
    job_script_header (str): The header for the job script.
    job_script_footer (str): The footer for the job script.
    incar_loc (str, optional): The location of the INCAR file for the calculation. If None, the INCAR file will be copied from the folder.
    kpoints_loc (str, optional): The location of the KPOINTS file for the calculation. If None, the KPOINTS file will be copied from the folder.
    potcar_loc_base (str, optional): The base location of the POTCAR file.

    Returns:
    None
    """
    Vs = setup_calculation(group_names, elem_all_groups, orbitals, GSorbs, xi_all_groups, zeta_all_groups,
                           workdir_self_en, EXtype, typepotcarfile, cutfuncpar, fullworkdirpath = fullworkdirpath)


    #####################
    # Defect poscar
    #####################
    old_poscar_loc = folder + '/POSCAR'
    new_poscar_loc = Vs[0].workdir + '/vasp_run/POSCAR'
    make_defect_poscar(old_poscar_loc, new_poscar_loc, all_defect_groups)

    #####################
    # Other input files (Incar & kpoints)
    #####################
    if incar_loc is None:
        # copy incar from original folder
        shutil.copyfile(folder + '/INCAR', Vs[0].workdir + '/vasp_run/INCAR')
    else:
        # copy incar given by user
        shutil.copyfile(incar_loc, Vs[0].workdir + '/vasp_run/INCAR')

    if kpoints_loc is None:
        # copy kpoints from original folder
        shutil.copyfile(folder + '/KPOINTS', Vs[0].workdir + '/vasp_run/KPOINTS')
    else:
        # copy kpoints given by user
        shutil.copyfile(kpoints_loc, Vs[0].workdir + '/vasp_run/KPOINTS')

    #####################
    # DFThalfCutoff object
    #####################
    # locations of pristine potcar files of each element
    # These are usually dumped on level above the vasp run file
    Potcar_loc = []
    for element in elem_all_groups:
        if potcar_loc_base is None:
            Potcar_loc.append(f'../POTCAR_{element}')
        else:
            Potcar_loc.append(f'{potcar_loc_base}/POTCAR_{element}')

    '''
    string with all potcars which are not altered in the cutoff optimatisation this could be '../POTCAR_Zn ../POTCAR_O
     for a  defect in ZnO or '../POTCAR_DFThalf_bulk_Zn ../POTCAR_DFThalf_bulk_O' incase we've also want to apply a 
     bulk correction which is usually the case. bulkpotcarloc = potcar_bulk # given input
    '''

    # The occupied defect band(s)
    band_ind = def_bands[0][0][0]
    band_spin = _get_band_spin(def_bands[0])
    occband = [band_ind, band_spin]

    # The unoccupied defect band(s)
    band_ind = def_bands[1][0][0]
    band_spin = _get_band_spin(def_bands[1])
    unoccband = [band_ind, band_spin]

    # Change working directory
    '''
    Since the calculation will not run on the same system we need to change the working directory of the potcarstup 
    object form an absolute to a relative path. Since the calculation is assumbe to be run in the vasp_run folder with 
    the workdir folders one level above that.
    '''
    workdir = Vs[0].workdir
    for vs in Vs:
        vs.workdir = '../'


    # DFThalfcutoff object for xi
    # self energies
    AtomSelfEnPots = Vs

    # Make cutoff object
    cutoff_opt = DFThalfCutoff(AtomSelfEnPots, Potcar_loc, occband, unoccband, typevasprun=typevasprun,
                               bulkpotcarloc=bulk_potcar,
                               save_eigenval=save_eigenval, save_doscar=save_doscar)

    # pickle cutoff object
    file = open(workdir + '/DFThalfCutoff.p', 'wb')
    pickle.dump(cutoff_opt, file)
    file.close()

    #####################
    # Make job scripts
    #####################
    # python script
    ps_string = '#importing libraries\nimport os\nimport pickle\n\n'  # import libraries
    ps_string += '#Setup DFThalfCutoff object\n'
    ps_string += 'with open("../DFThalfCutoff.p","rb") as file: \n\tcutoff_opt = pickle.load(file)\n'  # load DFThalf cutofoptimiser
    # set additional parameters for cutoff optimisation
    ps_string += 'cutoff_opt.foldervasprun = os.getcwd()\n'
    # currently DFThalfCutoff needs this input but this should be handelded more elegant in the future
    ps_string += '''cut_func_par = {'n': ''' +  f'{cutfuncpar["n"]}' + ''', 'Cutoff' : [0.0]}\n\n#run calculation\n'''
    ps_string += f'cutoff_opt.find_cutoff({str(rb)}, {str(rf)}, {str(nsteps)}, cut_func_par)'

    # Make python file
    with open(workdir + '/vasp_run/find_cutoff.py', 'w') as python_script:
        python_script.write(ps_string)

    # Make job script
    with open(workdir + '/vasp_run/' + job_script_name, 'w') as job_script_file:
        job_script_file.write(job_script_header)
        job_script_file.write('\n\npython find_cutoff.py\n ')
        job_script_file.write(job_script_footer)


def _setup_decoupled_runs(folder, workdir_self_en, xi_all_groups, zeta_all_groups, group_names, elem_all_groups, orbitals,
                          GSorbs, EXtype, typepotcarfile, cutfuncpar, all_defect_groups, def_bands, vbm_ind, cbm_ind,
                          typevasprun, bulk_potcar, save_eigenval, save_doscar, rb, rf, nsteps, job_script_name,
                          job_script_header, job_script_footer, incar_loc=None, kpoints_loc=None, potcar_loc_base=None,
                          fullworkdirpath = False):
    # Check potcar_loc_base. If this is None set it to ../../ such that potcar files can be shared between the xi and
    # zeta calculations
    if potcar_loc_base is None:
        potcar_loc_base = '../../'

    # xi
    workdir = workdir_self_en + '/xi'
    xi = xi_all_groups
    zeta = np.array(xi_all_groups) * 0
    # Set defect bands
    def_bands_xi = [def_bands[0], [[cbm_ind], def_bands[0][1]]]

    _setup_conventional_run(folder, workdir, xi, zeta, group_names, elem_all_groups, orbitals,
                          GSorbs, EXtype, typepotcarfile, cutfuncpar, all_defect_groups, def_bands_xi,
                          typevasprun, bulk_potcar, save_eigenval, save_doscar, rb, rf, nsteps, job_script_name,
                          job_script_header, job_script_footer, incar_loc=incar_loc, kpoints_loc=kpoints_loc,
                            potcar_loc_base=potcar_loc_base, fullworkdirpath = fullworkdirpath)

    # zeta
    workdir = workdir_self_en + '/zeta'
    xi = np.array(xi_all_groups) * 0
    zeta = zeta_all_groups
    # Set defect bands
    def_bands_zeta = [[[vbm_ind], def_bands[1][1]], def_bands[1]]
    _setup_conventional_run(folder, workdir, xi, zeta, group_names, elem_all_groups, orbitals,
                          GSorbs, EXtype, typepotcarfile, cutfuncpar, all_defect_groups, def_bands_zeta,
                          typevasprun, bulk_potcar, save_eigenval, save_doscar, rb, rf, nsteps, job_script_name,
                          job_script_header, job_script_footer, incar_loc=incar_loc, kpoints_loc=kpoints_loc,
                            potcar_loc_base=potcar_loc_base, fullworkdirpath = fullworkdirpath)



def _calc_efrac_ngroups(cag_s, ocg_s, n=2):
    """
    This function will return all the elecron fractions xi and zeta of the first groups ins cag_s and ocg_s
    cag_s: a list of list with the indices of a group of atoms contributinig to the defect orbitals.
    Usually this list should be sorted. cag stands for contributing atom groups
    ocg_s: the orbital contribution of each group
    n: the number of groups for which an electron fraction needs to be calculated
    """
    # Calc the multiplicity of each orbital character i.e. the number of atoms in each groups
    n_cag_s = cag_s[0:n]
    multiplicity = list(map(len, n_cag_s))
    # The orbital characters of relevant group
    n_ocg_s = ocg_s[0:n]
    # Calculate electron fraction
    efrac = calc_electron_fraction(Achar=n_ocg_s, mlt=multiplicity)
    return efrac

def _find_def_atoms_from_groups(cag_s, ocg_s, n=2):
    """
    Finds the k groups contributiong more than 0.01 to the defect bands.
    We'll start by only considering the 2 largest groups. If the second group has a xi or zeta < 0.01 then only the first group will be
    considered as defect atoms. If this is not the case we'll add another group until we find a group n for which xi and zeta<0.01
    When we have this group we keep the n-1 groups with a xi and zeta >= 0.01
    """
    # calculate electron fractions
    efrac = _calc_efrac_ngroups(cag_s, ocg_s, n=n)
    if np.sum(efrac[-1]) < 0.01:
        # return n-1 electron fraction and the number of contributing groups (n-1)
        return efrac[0:-1], (n - 1)
    elif len(efrac) == len(cag_s):
        # return all efracs and n the size of the given groups
        # -> this means to few groups were given and one should lower the threshold to find defect atoms
        logging.warning(
            'Not enough groups were given to find_def_atom! Def atoms should be rerun with a large amount of groups')
        # We still return this output which other function can use to know a large amount of groups should be submitted
        return efrac, n
    else:
        # In case where we do not have an electron fraction < 0.01 we need to include an extra group
        return _find_def_atoms_from_groups(cag_s, ocg_s, n=(n + 1))


def _find_def_atoms(projected_eign, band_ind, band_spin, structure, threshold_int=0.005, min_threshold=1e-5,
                    set_num_groups=None):
    """
    Finds the defect atoms contibuting to a defect bands from the spd projection of each atom on each band.
    projected_eign: the spd projection of each atoms for each band. Obtained from Vasprun object property .projected_eigenvalues
    band_ind: index of band in projected_eign
    band_spin: pymatgen Spin object
    structure: pymatgen structure object. Mainly used to get the element of each atom
    set_num_groups: integer with the number of groups you want returned. This is for a second run where you want less
    groups than in the default case.
    """
    # Get contribution for default parameters
    cag, ocg, eg = get_largest_contributors(projected_eign, band_ind, band_spin, structure, threshold=threshold_int)
    # Find defect groups and electron fractions
    efrac, num_groups = _find_def_atoms_from_groups(cag, ocg)
    # Check results if all groups where found. This is done by requiring that the number of groups found by
    # get_largest_contributors is larger than the number of groups found by _find_def_atoms_from_groups. If this is not
    # the case it is possible that there are more group that _find_def_atoms_from_groups could find but did not because
    # they were not included by get_largest_contributors. This means the threshold of get_largest_contributors should be
    # lowered
    if num_groups < len(cag):
        # Check if user wants a certain number of groups. If not use all groups found
        if set_num_groups is None:
            defect_atoms = cag[0:num_groups]  # get n groups with largest contribution
            defect_elem = eg[0:num_groups]
            return defect_atoms, efrac, defect_elem
        else:
            if num_groups < set_num_groups:
                # Although it is possible this was only a problem for xi or zeta thus it should not stop the program.
                logging.warning(f'_find_def_atoms only found {num_groups} groups. Expected {set_num_groups} groups.'
                                'Program continues with the groups found.')
                defect_atoms = cag[0:num_groups]  # get n groups with largest contribution
                defect_elem = eg[0:num_groups]
                return defect_atoms, efrac, defect_elem
                #raise Exception('set_num_groups was chosen too large there are not enough defect groups!')
            # Return predetermined number of groups
            defect_atoms = cag[0:set_num_groups]  # get n groups with largest contribution
            defect_elem  = eg[0:set_num_groups]
            efrac = _calc_efrac_ngroups(cag, ocg, n=set_num_groups)
            return defect_atoms, efrac, defect_elem

    else:
        # In this case the groups from get_largest_contributors did not contain all defect groups
        new_threshold = threshold_int/10 # we lower our threshold to increase the amount of group
        # To prevent infinite loops we introduce a minimum threshold
        if new_threshold < min_threshold:
            # check if old threshold is equal to min threshold if not do a run with min threshold
            if threshold_int==min_threshold:
                raise Exception('Threshold find_def_atoms reached minimum threshold!')
            else:
                return _find_def_atoms(projected_eign, band_ind, band_spin, structure, threshold_int=min_threshold,
                                       min_threshold=min_threshold, set_num_groups=set_num_groups)
        else:
            # Do a run with new threshold
            return _find_def_atoms(projected_eign, band_ind, band_spin, structure, threshold_int=new_threshold,
                                   min_threshold=min_threshold, set_num_groups=set_num_groups)

def _get_band_spin(def_band):
    if def_band[1] == 'up':
        return 1
    elif def_band[1] == 'down':
        return 2
    else:
        raise ValueError(f"Unexpected spin value: {def_band[1]}")

def renormalize_efracs(xi_all_groups, zeta_all_groups, all_defect_groups):
    # Calculate multiplicity for each group
    multiplicity = np.array([len(group) for group in all_defect_groups])[:, np.newaxis]

    # Calculate total contributions considering multiplicity
    xi_sum = np.sum(xi_all_groups * multiplicity)
    zeta_sum = np.sum(zeta_all_groups * multiplicity)

    # Renormalize xi and zeta
    xi_all_groups = xi_all_groups * (0.5 / xi_sum)
    zeta_all_groups = zeta_all_groups * (0.5 / zeta_sum)

    # Round xi and zeta to two decimals
    xi_all_groups = np.round(xi_all_groups, 2)
    zeta_all_groups = np.round(zeta_all_groups, 2)

    logging.debug(f'New renormalized xi and zeta values:\nxi: {xi_all_groups}\nzeta: {zeta_all_groups}')

    return xi_all_groups, zeta_all_groups

def _select_top_groups(xi_all_groups, zeta_all_groups, all_defect_groups, elem_all_groups, set_num_groups):
    # Add xi and zeta contributions and sort by total contribution then take the first set_num_groups
    total_contributions = np.sum(xi_all_groups, axis=1) + np.sum(zeta_all_groups, axis=1)
    sorted_indices = np.argsort(total_contributions)[::-1]
    all_defect_groups = [all_defect_groups[i] for i in sorted_indices[:set_num_groups]]
    xi_all_groups = xi_all_groups[sorted_indices[:set_num_groups]]
    zeta_all_groups = zeta_all_groups[sorted_indices[:set_num_groups]]
    elem_all_groups = [elem_all_groups[i] for i in sorted_indices[:set_num_groups]]

    # Renormalize xi and zeta
    xi_all_groups, zeta_all_groups = renormalize_efracs(xi_all_groups, zeta_all_groups, all_defect_groups)

    # Add warning that this case has not been properly tested
    logging.warning('_select_top_groups has not been properly tested xi and zeta values should be checked by the user!')

    return xi_all_groups, zeta_all_groups, all_defect_groups, elem_all_groups

def _apply_efrac_threshold(xi_all_groups, zeta_all_groups, all_defect_groups, elem_all_groups, efrac_threshold):
    logging.warning('Function _apply_efrac_threshold has not been properly tested yet!')
    while len(xi_all_groups) > 0:
        # Check if all xi and zetas in last group are below threshold
        if np.all(xi_all_groups[-1] < efrac_threshold) and np.all(zeta_all_groups[-1] < efrac_threshold):
            # Select all but one group from xi and zeta
            logging.debug(f'Xi and zeta values found below efrac_threshold {efrac_threshold}. Current number of groups {len(xi_all_groups)}')
            xi_all_groups, zeta_all_groups, all_defect_groups, elem_all_groups = _select_top_groups(
                xi_all_groups, zeta_all_groups, all_defect_groups, elem_all_groups, len(xi_all_groups) - 1)
        else:
            break
    logging.debug(f'New xi and zeta values are {xi_all_groups}, {zeta_all_groups}')
    return np.array(xi_all_groups), np.array(zeta_all_groups), all_defect_groups, elem_all_groups