import numpy as np
import os
import json
import fortranformat as ff
import AtomWrapper
import orbital
from PotcarWrapper import FindKmax, ReadPotcarfile

class potcarsetup:

    def __init__(self,workdir,atomname,atom,orb_structure,GSorbs,ExCorrAE = 'pb' , isfullpath=False, typeCutfunc='DFT-1/2'):
        # read config file
        with open('potcarsetupconfig.json') as json_file:
            config = json.load(json_file)
            self.potdir = config['potdir']
            self.ldadir = config['ldadir']
            self.pbedir = config['pbedir']

        # Set work directory
        if isfullpath:
            self.workdir = workdir
        else:
            self.workdir = self.potdir + '/' + workdir

        # Create working directory if it does not exist
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
        self.Radii = np.array([]) # radii of potvalues
        self.typeCutfunc = typeCutfunc
        # Atom properties
        self.atomname       = atomname
        self.atom           = atom
        self.orb_structure  = orb_structure
        self.GSorbs         = GSorbs

    def CalcSelfEnPot(self,xi,zeta,nrowspot=None):
        """
        # 1) Creates files structure for running atom.
        # 2) Runs ATOM to create in Xi and Zeta folder
        # 3) Calculate self energy potential (without cutoff function)
        :string atomname: Name/label of the atom. This does not have to be a chemical element. This is mostly used for file names
        :string atom: chemical element example 'C','N' and 'Cd'
        :param orb_structure: a list containing [int #core orbitals, #valence orbitals] Example [1,2] first orbitals a core orbitals the last two are valence orbitals
        :param GSorbs: Orbital object containing the valence orbitals and their ground state occupation.
        :param xi: List with electron fraction removed from each orbital given in GSorbs for the occupied(valence or defect) band.
        :param zeta: List with electron fraction removed from each orbital given in GSorbs for the unoccupied(conduction or defect) band.
        :param nrowspot: Debugging parameter in case we need to manually set nrows
        :return:
        """


        # CREATE DIRECTORY FOR CALCULATION
        atomdir = self.workdir + '/' + self.atomname
        if not (os.path.isdir(atomdir)):
            os.makedirs(atomdir)
            # GET DFT-1/2 OCCUPATION
            DFT12_occupied_orbs = []  # occupied orbitals (valence or defect band) affected by the xi electron fraction
            DFT12_empty_orbs = []  # empty orbitals (conduction or defect band) affected by zeta electron fraction
            for i, orb in enumerate(self.GSorbs):
                # the use of this orbitals class might be a bit overkill
                # Xi
                neworb = orb - orbital.orbital(orb.n, orb.l, xi[i])
                DFT12_occupied_orbs.append(neworb.as_dict())
                # Zeta
                neworb = orb - orbital.orbital(orb.n, orb.l, zeta[i])
                DFT12_empty_orbs.append(neworb.as_dict())

            # RUN ATOM FOR OCCUPIED BANDS (Xi)
            os.makedirs(atomdir + '/Xi')
            self.AW.MakeInputFile(atomdir + '/Xi', self.atom, self.orb_structure, DFT12_occupied_orbs, EXtype=self.ExCorrAE)
            self.AW.RunATOM(atomdir + '/Xi')

            # RUN ATOM FOR UNOCCUPIED BANDS (Xi)
            os.makedirs(atomdir + '/Zeta')
            self.AW.MakeInputFile(atomdir + '/Zeta', self.atom, self.orb_structure, DFT12_empty_orbs, EXtype=self.ExCorrAE)
            self.AW.RunATOM(atomdir + '/Zeta')
        else:
            print('Self energy potentials already calculated!\nTry another atomname or delete this folder!')
            print('Calculation of the self energy potential proceeds with the excisting files!')

        # SUBSTRACT
        if nrowspot==None:
            nrowspot = self.AW.Calcnrows(atomdir + '/Xi/VTOTAL1')
        self.Radii = self.AW.ReadPotfile(atomdir + '/Xi/VTOTAL1'  , nrows=nrowspot,skiprows=1) # We need to save this for later
        print(self.Radii)
        pot_xi     = self.AW.ReadPotfile(atomdir + '/Xi/VTOTAL1'  , nrows=nrowspot,skiprows=nrowspot+3)
        pot_zeta   = self.AW.ReadPotfile(atomdir + '/Zeta/VTOTAL1', nrows=nrowspot,skiprows=nrowspot+3)
        # We already multiply with these constants to convert from
        # the atom format(Ryd and bohr) to the vasp format (eV and A)
        self.Vs = 4.0*np.pi*self.alpha*(self.beta**3)*(pot_xi - pot_zeta)

        return self.Vs

    def MakePotcar(self, potcarfile, CutFuncPar,numdecCut=3):
        """
        Makes DFT-1/2 potcar with the speciefied parameters
        :param potcarfile: file location of potcar file
        :param CutFuncPar: dict with parameter cutoff function
        :param Cutfunc: tag to tell potcar setup which trimming function is used. (currently this does nothing)
        :param numdecCut: number of decimals for the cutoff. Usually you should not need to touch this.
        :return:
        """
        # MAKE POTCAR DIRECTORY
        potcarfolder = self.workdir + '/' + self.atomname + '/POTCAR_DFThalf'
        if not (os.path.isdir(potcarfolder)):
            os.makedirs(potcarfolder)

        Cutoff = CutFuncPar['Cutoff']

        if isinstance(Cutoff,list) or isinstance(Cutoff,type(np.array([]))):
            if potcarfile == 'LDA' or potcarfile == 'lda':
                potcarfile = self.ldadir + '/' + self.atom + '/POTCAR'
            elif potcarfile == 'PBE' or potcarfile == 'pbe':
                potcarfile = self.pbedir + '/' + self.atom + '/POTCAR'
            # Read potcar such that it only needs to be read once
            kmax, potcarjump = FindKmax(potcarfile)
            potcar, nrows, _, _ = ReadPotcarfile(potcarfile)  # read local part op potcar

            # LOOP OVER ALL GIVEN CUTS
            for i,Cut in enumerate(Cutoff):
                # setup parameters
                newCutFuncPar = {
                    'Cutoff'    : np.round(Cut,numdecCut),
                    'n'         : CutFuncPar['n']
                }
                newpotcarfile = potcarfolder + '/POTCAR_rc_' + str(np.round(Cut,numdecCut)) + '_n_' +  str(CutFuncPar['n'])# newpotcarfile will now be used a the first part of the name
                Vs = self.DefCalcTrimmedVs(newCutFuncPar)
                newpotcar = self.AddVs2Potcar(Vs, potcarfile, newpotcarfile, newCutFuncPar['Cutoff'],
                                              kmax=kmax, potcarjump=potcarjump, potcar=potcar, nrows=nrows)
        else:
            # CONSTRUCT SELF ENERGY POTENTIAL X TRIMMING FUNCTION
            Vs = self.DefCalcTrimmedVs(CutFuncPar)

            # ADD TRIMMED SELF ENERGY POTENTIAL TO POTCAR
            newpotcarfile = potcarfolder + '/POTCAR_' + str(np.round(Cutoff,numdecCut))
            newpotcar = self.AddVs2Potcar(Vs, potcarfile, newpotcarfile, CutFuncPar['Cutoff'])


    def DefCalcTrimmedVs(self,CutFuncPar):
        # CONSTRUCT SELF ENERGY POTENTIAL X TRIMMING FUNCTION
        Cutoff = CutFuncPar['Cutoff']
        n = CutFuncPar['n']
        if self.typeCutfunc == 'DFT-1/2':
            if Cutoff != 0:
                Vs = self.Vs * (1.0 - (self.Radii / Cutoff) ** n) ** 3 * (self.Radii < Cutoff)  # Apply trimming function
            else:
                Vs = np.zeros(self.Vs.shape)
        else:
            raise Exception('Unknown Cutoff function type was used.')
        return Vs

    def AddVs2Potcar(self,Vs,potcarfile,newpotcarfile,Cutoff,kmax=None,potcarjump=None,potcar=None,nrows=None,useFortanmethod=False):
        # Check if potcar file is already made
        if os.path.isfile(newpotcarfile):
            return 0
        print(newpotcarfile)
        # READ POTCAR FILE
        if kmax == None:
            kmax, potcarjump = FindKmax(potcarfile)
        if isinstance(potcar, type(None)):
            potcar, nrows, _, _ = ReadPotcarfile(potcarfile)  # read local part op potcar

        nk = nrows*5 # number of kvalues in potcar file
        newpotcar = None
        if useFortanmethod:
            # older method which mimics fortran code (very slow)
            Nrad = np.shape(self.Radii)[0]
            ca = 0.0
            newpotcar = potcar.copy()
            for i in range(nk):
                ca = ca + kmax / nk
                newpotcar[i] = newpotcar[i] + self.AW.Add2PotcarFourier(self.beta * ca, Nrad, self.Radii, Vs, Cutoff, 0.0, 0.0, 0.0) / (
                            self.beta * ca)
        else:
            # new method that  should produce the same result but faster
            newpotcar = self.Add2PotcarFourier(Vs,potcar,Cutoff,nk,kmax)

        # WRITE NEW POTCAR
        lineformat = ff.FortranRecordWriter('(5e16.8)')
        with open(potcarfile,'r') as pfile:
            npfile = open(newpotcarfile,'x') # new potcar file
            # copy potcar until local part
            for i in range(potcarjump):
                npfile.write(pfile.readline())
            # place new local part
            for i in range(nrows):
                for j in range(5):
                    line = lineformat.write([newpotcar[5*i + j]])
                    npfile.write( line )
                npfile.write('\n')

        with open(potcarfile,'r') as pfile:
            # copy the rest
            skiprows = nrows + potcarjump -1
            for i,line in enumerate(pfile):
                if i > skiprows:
                    npfile.write(line)
            npfile.close()

        return newpotcar

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

    def Add2PotcarFourier(self,Vs,potcar,Cut,nk,kmax):
        kp = self.beta * np.linspace(kmax / nk, kmax, nk)
        newpotcar = potcar.copy()
        # first element
        I = np.arange(1, len(self.Radii))
        for i, k in enumerate(kp):
            fourier = 0
            fourier += (Vs[0] * np.sin(k * self.Radii[0])) * (self.Radii[0]) / 2.0
            # elements with R<cut
            fourier += np.sum(((Vs[I] * np.sin(k * self.Radii[I]) + Vs[I - 1] * np.sin(k * self.Radii[I - 1])) *
                               (self.Radii[I] - self.Radii[I - 1])) * (self.Radii[I] < Cut)) / 2.0
            # elements with R>cut
            indRcut = np.argmax(self.Radii >= Cut)
            fourier += (Vs[indRcut] * np.sin(k * self.Radii[indRcut]) * (
                        Cut - self.Radii[indRcut])) / 2.0  # This is always 0 it seems
            # update potcar with value or fourier transform
            newpotcar[i] = newpotcar[i] + fourier / (k)
        return newpotcar