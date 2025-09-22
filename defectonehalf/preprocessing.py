import pickle
import os
import shutil
import numpy as np
import pymatgen.io.vasp as pmg
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

import defectonehalf.potcarsetup as ps

def print_band_characters(bandind, atomind, Peign, structure, kp=0):
    if not (isinstance(atomind, list)):
        atomind = [atomind]
    for bi in bandind:
        print('\nBand:',bi)
        print('Orbitals  : \t [s p d]')
        if not( isinstance(bi,list)):
            bi = [bi]
        for ia in atomind:
            #print(Peign[0, bi, ia])
            print('\t\t', np.average(Peign[kp, bi, ia], 0), structure[ia])


def calc_electron_fraction(Achar=None, mlt=None, Peign=None, iocc=None,iunocc=None,atominds=None,numdec=2, kp=0):
    """
    This function calculates the fraction of an electron (Xi and Zeta) that needs to be
    substracted of each orbital of each atom for a defect dft-1/2 calculation

    Achar:    A list of list with each list containing the character of each
              atom ie [[Cs,Cp,Cd],[Ns,Np,Nd]]
    mlt:      A list containing the multiplicity of each atom
    Peign:    A projected eigenvalue object from pymatgen
    iocc:     index occupied orbital
    iunocc:   index unoccupied orbital
    atominds: indeces of all atoms involved
    numdec:   number of decimals to round to
    kp:       kpoint index. Default 0 this is usually the gamma point
    """

    if not(isinstance(Achar,type(None)) ):
        Efrac = calc_electron_fraction_basic(Achar, mlt=mlt)
    elif not(isinstance(Peign,type(None))):
        Efrac = calc_electron_fraction_fullinput(Peign,iocc,iunocc,atominds,mlt, kp=kp)
    Efrac = np.round(Efrac,numdec)
    return Efrac

def full_band_character_analysis(folder, iocc, iunocc, atominds, spin, print_band_chars=True, print_xi_zeta=False, kp=0):
    # load structure
    structure = Structure.from_file(folder + "/POSCAR")

    # load project eigenvalues
    run = pmg.Vasprun(folder + '/vasprun.xml', parse_potcar_file=False,
                      parse_eigen=True, parse_projected_eigen=True, separate_spins=True)
    Peign = run.projected_eigenvalues[spin]

    # Check bands
    if print_band_chars:
        print_band_characters([iocc, iunocc],atominds, Peign, structure)

    # Calculate electron fractions
    Xi, Zeta = calc_electron_fraction(Peign=Peign, iocc=iocc, iunocc=iunocc, atominds=atominds, kp=kp)

    if print_xi_zeta:
        print('{:<20} {:<20}'.format('Xi', 'Zeta'))
        for i in range(len(Xi)):
            print('{:<20} {:<20}'.format(str(Xi[i]), str(Zeta[i])))

    return Xi, Zeta

def setup_calculation(atomnames, atoms, orbitals, GSorbs, Xi, Zeta, workdir, EXtype, potcarfile, cutfuncpar,
                      vaspfiles=None, fullworkdirpath=False):
    """
    This function setups a folder to prefrom a cutoff sweep from DFThalfCutoff
    :param atomnames: list with atom labels
    :param atoms: list with atomic symbols
    :param orbitals: list of list containing the number of core and valence electrons
    [ [#core e atom1, #val e atom1], [#core e atom2,#val e atom2], ..., [#core e atomn,#val e atomn]]
    :param GSorbs: list orbital object containing the ground state configuration
    :param Xi: list of xi for each orbtial of each atom [[xi_s a1, xi_p a1, xi_d a1],...]
    :param Zeta: list of zeta for each orbtial of each atom [[zeta_s a1, zeta_p a1, zeta_d a1],...]
    :param workdir: name/path to working directory
    :param EXtype: exchange correclation type for ATOM
    :param potcarfile: string with potcar file type 'lda' or 'pbe'
    :param cutfuncpar:  dict with cutoff function parameters
    :param vaspfiles: list of vasp files location which will be copied to the vasp_run file
    :param fullworkdirpath: if True workdir is the full path to the workdir. If False workdir is the name of the workdir
    :return:
    """
    # Set vaspfiles to empty list if None
    if vaspfiles is None:
        vaspfiles = []

    Vs_list = [] # list of all self energies
    for i, atom in enumerate(atoms):
        # Calc Vs
        Vs = ps.PotcarSetup(workdir, atomnames[i], atom, orbitals[i], GSorbs[i], ExCorrAE=EXtype,
                            isfullpath=fullworkdirpath)
        Vs.calc_self_En_pot(Xi[i], Zeta[i])

        # Make potcars
        Vs.make_potcar(potcarfile, cutfuncpar)
        Vs_list.append(Vs)

        # Safe potcar setup object
        file = open(Vs.workdir + '/' +atomnames[i] + '_ps.PotSetup', 'wb')
        pickle.dump(Vs, file)
        file.close()

        # make vasp run folder
        if i == 0 and not(os.path.isdir(Vs.workdir + '/vasp_run')):
            # setup vasp calcualtion
            os.makedirs(Vs.workdir + '/vasp_run')
    # copy vasp files
    for file in vaspfiles:
        shutil.copy(file, f'{Vs_list[0].workdir}/vasp_run/' )
    return Vs_list

def get_largest_contributors(projected_eign, bandind, spin, structure, threshold=0.005, group_sym_tol=0.001,
                             kpoint_index=0):
    """
    Returns the indices of the largest contributors

    Parameters:
    - projectd_eign: A 4D array of eigenvalues for the projected density of states (PDOS). The first index corresponds to the spin (up or down),
    the second index corresponds to the band index, the third index corresponds to the site index, and the fourth index corresponds to the orbital index.
    - bandind: An integer or list of integers indicating the band index to use.
    - spin: An integer indicating the spin to use (0 for up, 1 for down).
    - threshold: A float indicating the threshold for the sum of the orbital characters.
    - kpoint_index: An integer indicating the kpoint index to use. This parameter has a default value of 0 i.e. the Gamma point
    Only sites with a sum above this threshold will be printed. This parameter has a default value of 0.005.
    This values means that any atom with a xi or zeta of 0.01 (the precision of atom our AE code, 0.01 because DFT-1/2
    normalised the characters to a sum of 0.5) will be included.
    :param group_sym_tol: The maximum absolute differnece between orbital chararcters of the same group.
    """
    # Normalize the eigenvalues for the given spin and band index/indices
    if isinstance(bandind, list):
        projected_eign_norm = projected_eign[spin][kpoint_index, bandind[0], :, :].copy()
        for bi in bandind[1:]:
            projected_eign_norm += projected_eign[spin][kpoint_index, bi, :, :]
        projected_eign_norm = projected_eign_norm / np.sum(projected_eign_norm)
    else:
        projected_eign_norm = projected_eign[spin][kpoint_index, bandind, :, :].copy() / np.sum(projected_eign[spin][kpoint_index, bandind, :, :])

    # make empty list to save the group of atoms
    contributing_atom_groups = []
    orb_char_groups = []
    element_groups  = []
    # Iterate through the sites in the structure
    for i, site in enumerate(structure):
        # Get the orbital characters for the current site and spin
        orbital_characters = projected_eign_norm[i,:]

        # Check if any oribtal character(s,p,d) of the current site is above the threshold
        if any(list( map(lambda orb_char: orb_char >= threshold,  orbital_characters))):
            #contributing_atom_groups.append(i) # add atom index to contributing atoms

            # Check if there is a group with the same characters and atoms type
            no_group_found = True
            for g_ind, char_group in enumerate(orb_char_groups):
                if  np.sum( np.abs(char_group - orbital_characters)) < group_sym_tol and site.specie.symbol == element_groups[g_ind]:
                    # if we find such a group add index to this group
                    contributing_atom_groups[g_ind].append(i)
                    no_group_found = False
                    break
            # If the current site does not belong to a group but has a large enough contribution, a new group will be
            # created
            if no_group_found:
                contributing_atom_groups.append([i])
                orb_char_groups.append(orbital_characters)
                element_groups.append(site.specie.symbol)

    # Sort from largest to lowest contribution
    total_char_groups = np.sum(orb_char_groups, axis=1)  # Total spd contribution
    sorted_indices = np.flip(np.argsort(total_char_groups))  # indices to sort from highest to lowest

    # Make new list in this order
    cag_sorted = []  # contributing atom group sorted
    ocg_sorted = []  # orbital charaters of each group sorted
    eg_sorted = []  # chemical element of each group
    for i in sorted_indices:
        cag_sorted.append(contributing_atom_groups[i])
        ocg_sorted.append(orb_char_groups[i])
        eg_sorted.append(element_groups[i])

    return cag_sorted, ocg_sorted, eg_sorted



def print_largest_contributors(projected_eign, bandind, spin, structure, threshold=0.005, group_sym_tol=0.001, kpoint_index=0):
    """
    Prints the index, position, atom, and orbital characters for all sites in the structure that have a sum of orbital characters above the given threshold.

    Parameters:
    - projectd_eign: A 4D array of eigenvalues for the projected density of states (PDOS). The first index corresponds to the spin (up or down),
    the second index corresponds to the band index, the third index corresponds to the site index, and the fourth index corresponds to the orbital index.
    - bandind: An integer indicating the band index to use.
    - spin: An integer indicating the spin to use (0 for up, 1 for down).
    - threshold: A float indicating the threshold for the sum of the orbital characters.
    - kpoint_index: An integer indicating the kpoint index to use. This parameter has a default value of 0 i.e. the Gamma point
    Only sites with a sum above this threshold will be printed. This parameter has a default value of 0.005.
    """
    # We use get_largest_contributors to find the indices of the largest contributors
    contributing_atom_groups, orb_char_groups, element_groups = get_largest_contributors(projected_eign, bandind, spin,
                                                                                         structure, threshold=threshold,
                                                                                         group_sym_tol=group_sym_tol,
                                                                                         kpoint_index=kpoint_index)

    # Loop over all groups and print the result of each group
    for ig, group in enumerate(contributing_atom_groups):
        print('{:<5} {:<5} {:<35} {:<12} {:<35}'.format('Group', 'Atom', 'SPD','Group size','Group contribution to bands'))
        length_group = len(group)
        percentage_cont = np.round(np.sum(orb_char_groups[ig])*100*length_group,2)
        print('{:<5} {:<5} {:<35} {:<12} {:<35}'.format(str(ig+1), str(element_groups[ig]), str(orb_char_groups[ig]),
                                                str(length_group),
                                                str(percentage_cont) + '%' ))
        print('{:<5} {:<35}'.format('Index', 'Position'))
        for index_element in group:
            print('{:<5} {:<35}'.format(index_element, str(structure[index_element])))
        print()

    # Normalize the eigenvalues for the given spin and band index
    projected_eign_norm = projected_eign[spin][0,bandind, :, :]/np.sum(projected_eign[spin][0,bandind, :, :])



def make_defect_poscar(poscar_loc, defect_poscar_loc,atom_groups, defect_atom_names=None):
    """
    Makes a new poscar file with the defect atoms at the bottom. This way the DFT-1/2 potcar file can easily be
    generated.
    :param poscar_loc: Path to poscar file
    :param defect_poscar_loc: The location where the new poscar will be stored
    :param atom_groups: list of list with each list containing the indices of the defect atoms belonging to a certain
    group.
    :param
    :return: list with names of defect atoms. Default None, the name will be the atoms specie
    """
    # load poscar
    defect_structure = Structure.from_file(poscar_loc)

    # check if defect atoms names where given
    if defect_atom_names is None:
        defect_atom_names = []
        # loop over all groups to fill defect_atom_names
        for a_group in atom_groups:
            # defect atom name will be the specie of the first atom in each group. All atoms should have the same
            # species though.
            defect_atom_names.append(str(defect_structure[a_group[0]].species))

    # Variables used during the rearrangement of the poscar
    removed_atom_inds = []
    poscar_comment  = 'Poscar made by make_defect_poscar: '
    # line 6 in the poscar, contains all element and the order in which there given. We start with adding the host
    # material. This only works for mono atomic materials.
    bulk_poscar_elements = [] # list with all non defect (or element for which no correction will be applied) elements in the poscar
    bulk_poscar_elements_occurences = [] # list with the number of times each element is present in the poscar

    for i in range(len(defect_structure.species)):
        # Check if i is not defect atom. If not we found a a bulk atom
        if not(_check_index_in_sublist(i, atom_groups)):
            #poscar_elements = str(defect_structure[i].specie.symbol) + ' '
            # Check if element is already in the list if not a new bulk species is added
            if str(defect_structure[i].specie.symbol) not in bulk_poscar_elements:
                bulk_poscar_elements.append(str(defect_structure[i].specie.symbol))
                # set occurences to 1
                bulk_poscar_elements_occurences.append(1)
            else:
                # update occurences
                bulk_poscar_elements_occurences[bulk_poscar_elements.index(str(defect_structure[i].specie.symbol))] += 1

    poscar_elements = ' '.join(bulk_poscar_elements) + ' ' # add bulk elements to poscar elements
    # line 7 in the poscar containts the number of times each element from line 6 will be present
    number_atoms_line = ' '.join(map(str,bulk_poscar_elements_occurences)) + ' ' # number of bulk atoms
    #number_atoms_line = str(int( len(defect_structure.species) - sum(map(len,atom_groups)) )) + ' ' # number of bulk atoms

    # Rewrite poscar. We loop over each group of atoms remove them from the poscar and adding them at the end.
    for i, a_group in enumerate(atom_groups):
        ag_sort = np.flip(np.sort(a_group)) # sort list from highest to lowest, such that removing atoms is easier
        for j, atom_ind in enumerate(ag_sort):
            # If a index below the current one has already been removed we should decrease our current index by 1
            adjusted_ind = atom_ind
            for ri in removed_atom_inds:
                if atom_ind > ri:
                    adjusted_ind -= 1

            # Add element to poscar lines
            if j == 0:
                poscar_comment += defect_atom_names[i] + '_' + str(len(a_group)) + ' '
                poscar_elements += str(defect_structure[adjusted_ind].specie.symbol) + ' '
                number_atoms_line += str(int(len(a_group))) + ' '

            # Update poscar
            site = defect_structure[adjusted_ind]     # save site
            defect_structure.pop(adjusted_ind)        # remove atom
            defect_structure.append(site.species, site.coords, coords_are_cartesian=True) # add atom at the back of the poscar
            removed_atom_inds.append(atom_ind)

    # Make new poscar
    new_poscar = Poscar(defect_structure, comment=poscar_comment)
    new_poscar.write_file(defect_poscar_loc)

    # Change species list in poscar
    with open(defect_poscar_loc, 'r') as f:
        lines = f.readlines()
    lines[5] = poscar_elements + '\n'
    lines[6] = number_atoms_line + '\n'
    with open(defect_poscar_loc, 'w') as f:
        f.writelines(lines)

    # Check if sum of all atoms in number_atoms_line is equal to the number of atoms in the structure object
    if len(defect_structure.species) != sum(map(int,number_atoms_line.split())): # split number_atoms_line and convert to int
        raise Exception('The sum of all atoms in number_atoms_line is not equal to the number of atoms in the structure object! \n'
                        +'You must set the potcar manually!')

    print('WARNING: Poscar was automatically generated by make_defect_poscar and should be checked!' +
          'Check the header and make sure the defect atoms are at the bottom of the poscar and all atoms are accounted for!')


#################################
# HELPER FUNCTIONS
#################################


def calc_electron_fraction_basic(Achar, mlt=None):
    """
    This function calculates the fraction of an electron (Xi and Zeta) that needs to be
    substracted of each orbital of each atom for a defect dft-1/2 calculation

    Achar:    A list of list with each list containing the character of each
              atom ie [[Cs,Cp,Cd],[Ns,Np,Nd]]
    mlt:      A list containing the multiplicity of each atom
    """

    if isinstance(mlt,type(None)):
        mlt = np.ones(len(Achar))

    # Calculate norm such that Sum char = 1/2
    norm = 0
    for c in range(0, len(Achar)):
        norm += mlt[c] * np.sum(Achar[c])
    norm = 2 * norm  # extra factor 2 such that the sums is equal to 1/2

    # Calculate electron fraction
    Efrac = []  # new list with normalised characters
    for ch in Achar:
        Efrac.append(ch / norm)

    return Efrac

def calc_electron_fraction_fullinput(Peign, iocc,iunocc,atominds, mlt=None, kp=0):
    """
    Allows to calculate the electron fraction a more lazy manner using more inputs.
    :param Peign: projected eigenvalue object from pymatgen
    :param iocc: index occupied orbital
    :param iunocc: index unoccupied orbital
    :param atominds: indeces of all atoms involved
    :param mlt: list with multiplicity of each atoms. If none is give the multiplicty of each atom is assumed to be 1
    :param kp: kpoint index
    :return:
    """

    if not(isinstance(iocc,list)):
        iocc = [iocc]
    if not(isinstance(iunocc,list)):
        iunocc = [iunocc]

    char_occ   = []
    char_unocc = []
    for i, ai in enumerate(atominds):
        co = np.average(Peign[kp, iocc  , ai], 0)
        char_occ.append(co)
        cu = np.average(Peign[kp, iunocc, ai],0)
        char_unocc.append(cu)

    Xi = calc_electron_fraction_basic(char_occ  , mlt=mlt)
    Ze = calc_electron_fraction_basic(char_unocc, mlt=mlt)
    Efrac = [Xi, Ze]

    return  Efrac

def _check_index_in_sublist(index, indexlist):
    # Checks if index is in any sublist of list
    for sublist in indexlist:
        for i in sublist:
            if i == index:
                return True
    return False