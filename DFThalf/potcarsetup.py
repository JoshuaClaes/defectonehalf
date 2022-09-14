import numpy as np
import os
import json
import AtomWrapper
import orbital
from PotcarWrapper import FindKmax

class potcarsetup:

    def __init__(self,workdir,ExCorrAE = 'pb' , isfullpath=False):
        # read config file
        with open('potcarsetupconfig.json') as json_file:
            config = json.load(json_file)
            self.potdir = config['potdir']

        # Set work directory
        if isfullpath:
            self.workdir = workdir
        else:
            self.workdir = self.potdir + '/' + workdir
        # Create working directory if it doeorbitalss not exist
        if not (os.path.isdir(self.workdir)):
            os.makedirs(self.workdir)
        # Set Exchange correlation for all electron code
        self.ExCorrAE = ExCorrAE
        # initiate atom wrapper
        self.AW = AtomWrapper.AtomWrapper() # object to run atom calculations

        # constants
        self.alpha = 13.605803  # conversion factor Ryd to eV
        self.beta  = 0.52917706 # conversion factor bohr to angstrom
        # self energy
        self.Vs = np.array([])


    def MakeSweepPotcarfiles(self):
        # Creates a set of potcarfiles going from Cstart with Cend with steps of Cstep
        return 0

    def CalcSelfEnPot(self,atomname,atom,orb_structure,GSorbs,xi,zeta,nrowspot=None):
        # 1) Creates files structure for running atom.
        # 2) Runs ATOM to create in Xi and Zeta folder
        # 3) Calculate self energy potential (without cutoff function)

        # CREATE DIRECTORY FOR CALCULATION
        atomdir = self.workdir + '/' + atomname
        if not (os.path.isdir(atomdir)):
            os.makedirs(atomdir)
        else:
            print('Self energy potentials already calculated!\nTry another atomname or delete this folder!')

        # GET DFT-1/2 OCCUPATION
        DFT12_occupied_orbs = [] # occupied orbitals (valence or defect band) affected by the xi electron fraction
        DFT12_empty_orbs    = [] # empty orbitals (conduction or defect band) affected by zeta electron fraction
        for i, orb in enumerate(GSorbs):
            # the use of this orbitals class might be a bit overkill
            # Xi
            neworb = orb - orbital.orbital(orb.n,orb.l,xi[i])
            DFT12_occupied_orbs.append(neworb.as_dict())
            # Zeta
            neworb = orb - orbital.orbital(orb.n, orb.l, zeta[i])
            DFT12_empty_orbs.append(neworb.as_dict())

        # RUN ATOM FOR OCCUPIED BANDS (Xi)
        os.makedirs(atomdir + '/Xi')
        self.AW.MakeInputFile(atomdir + '/Xi', atom, orb_structure, DFT12_occupied_orbs, EXtype=self.ExCorrAE)
        self.AW.RunATOM(atomdir + '/Xi')

        # RUN ATOM FOR UNOCCUPIED BANDS (Xi)
        os.makedirs(atomdir + '/Zeta')
        self.AW.MakeInputFile(atomdir + '/Zeta', atom, orb_structure, DFT12_empty_orbs, EXtype=self.ExCorrAE)
        self.AW.RunATOM(atomdir + '/Zeta')

        # SUBSTRACT
        if nrowspot==None:
            nrowspot = self.AW.Calcnrows(atomdir + '/Xi/Vtotal1')
        pot_xi   = self.AW.ReadPotfile(atomdir + '/Xi/Vtotal1',nrows=nrowspot)
        pot_zeta = self.AW.ReadPotfile(atomdir + '/Xi/Vtotal1',nrows=nrowspot)
        # We already multiply with these constants to convert from
        # the atom format(Ryd and bohr) to the vasp format (eV and A)
        self.Vs = 4.0*np.pi*self.alpha*(self.beta**3)*(pot_xi - pot_zeta)

        return self.Vs

    def MakePotcar(self,potcarfile,newpotcarfile,CutFuncPar,Cutfunc='DFT-1/2',kmax=None):
        # ADD SELF ENERGY POTENTIAL TO POTCAR FILE
        if isinstance(kmax,None):
            kmax, potcarjump = FindKmax(potcarfile)
        potcar = self.ReadPotcarfile(potcarfile, potcarjump, nk)  # read local part op potcar





    def RunAEcode(self, dir):
        # run ATOM
        self.AW.RunATOM(dir)
        return None

    def CalcAEpot(self, dir, atom, orb_structure, DFT12_occupied_orbs):
        # create input fi
        self.AW.MakeInputFile(dir, atom, orb_structure, DFT12_occupied_orbs, EXtype=self.ExCorrAE)
        self.AW.RunATOM(dir)


    def CalcElectronFraction(Achar, mlt=None):
        # This might be better in another function for defect dft-1/2
        """
        This function calculates the fraction of an electron that needs to be
        substracted of each orbital of each atom for a defect dft-1/2 calculation

        Achar:    A list of list with each list containing the character of each
                  atom ie [[Cs,Cp,Cd],[Ns,Np,Nd]]
        mlt:      A list containing the multiplicity of each atom
        """
        if mlt == None:
            mlt = np.ones(1, len(Achar))

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