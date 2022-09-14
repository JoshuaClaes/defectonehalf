import os
import json
import linecache
import fortranformat as ff
import pandas as pd
import numpy as np


class AtomWrapper:
    def __init__(self):
        # read config file
        with open('AtomWrapperConfig.json') as json_file:
            self.config = json.load(json_file)
        #print(self.config)

    def RunATOM(self,dir):
        cwd = os.getcwd()
        os.chdir(dir)
        os.system(self.config['AtomEx'])
        os.chdir(cwd)
        return 0

    def MakeInputFile(self,dir,atom,orbitals,occupation,EXtype='ca'):
        """
        Function makes an input file for ATOM
        :string dir: directory for input file
        :string atom: name/symbol of atom
        :param EXtype: Exchange correlation type used in ATOM. ca Ceperley-Alder (best LDA), pb PBE
        :return:
        """
        with open(dir + '/INP','w') as f:
            f.write('#\n# ' + atom +'\n')
            f.write('   ae      '+ atom + '\n') # ae -> tell atom to perform calculation, there rest is a title
            f.write(' n=' + atom + '  c=' +EXtype + '\n')
            f.write('       0.0       0.0       0.0       0.0       0.0       0.0\n')
            f.write('    ' + str(orbitals[0])+ '    ' + str(orbitals[1])+'\n')
            for occ in occupation:
                f.write('    ' + str(occ['n']) + '    ' + str(occ['l']) + '      ' + str(occ['occupation']) + '      0.00' +'\n')
            f.write('100 maxit')
        return 0

    def CalcSelfEnergy(self,radfile, potfile_xi, potfile_zeta,Nrad=None):
        # READ RADII
        radii = self.ReadRadii(radfile, Nrad)
        # READ RADII
        radii = self.ReadRadii(radfile, Nrad)

        # READ POTENTIAL OCCUPIED BAND XI
        skiprows = int(np.ceil(Nrad / 4) + 3)  # skip all rows of radii + head of radii (size = 1)
        # + head of potential (size = 2)
        pot_xi = self.ReadPotfile(potfile_xi, Nrad, skiprows=skiprows)

        # READ POTENTIAL UNOCCUPIED BANDS ZETA
        pot_zeta = self.ReadPotfile(potfile_zeta, Nrad, skiprows=skiprows)



    def Add2Potcar(self,Nrad,radfile,potfile_zeta,potfile_xi,CutFuncPar,potcarfile, nk, potcarjump,newpotcarfile):
        
        #Function is based on the add2POTCAR-eng.f90 fortran script which add the self energy potential to a potcar
        #rfile = radius file in fortran
        #kmax number under local part in POTCAR FILE
        #:param Nrad: Number of radii (nuk in fortran)
        #:param potfile_zeta: location atom potential file for unoccupied bands
        #:param potfile_xi: location atom potential file for occupied bands
        #:param CutFuncPar: parameters of the cutoff function n CUT amplitude
        #:param potcarfile: location potcar file
        #:return: None
        

        nrows = self.Calcnrows(radfile)
        # READ RADII
        radii = self.ReadPotfile(radfile,nrows=nrows, skiprows=1)

        # READ POTENTIAL OCCUPIED BAND XI
        skiprows = int( np.ceil(Nrad/4) + 3)    # skip all rows of radii + head of radii (size = 1)
                                                # + head of potential (size = 2)
        pot_xi = self.ReadPotfile(potfile_xi, nrows=nrows, skiprows=skiprows)

        # READ POTENTIAL UNOCCUPIED BANDS ZETA
        pot_zeta = self.ReadPotfile(potfile_zeta, nrows=nrows, skiprows=skiprows)

        # CALCULATE SELF ENERGY POTENTIAL
        # We use V_zeta - V_xi following the paper of Lucatto et. al. about DFT-1/2 and defects
        # Because Vs(occupied) = Vx(f0) - Vx(f0-xi) and Vs(unoccupied) = -[ Vx(f0) - Vx(f0-zeta)]
        # resulting in Vs = Vs(occupied) - Vs(unoccupied) = Vs(f0-zeta) - Vs(f0-xi) = V_zeta - V_xi
        # This means we can do both the valence band correction where zeta gets the role of the fully occupied atom
        # and xi is the removed fraction of bulk atoms. And ofcourse the defect correction.
        alpha   = 13.605803
        beta    = 0.52917706
        n       = CutFuncPar[0]
        CUT     = CutFuncPar[1]
        Vs      = 4.0*np.pi*alpha*(beta**3) * ( 1.0 - (radii/CUT)**n )**3 *  (pot_xi - pot_zeta)
        Vs      = Vs *  (radii < CUT)

        # ADD SELF ENERGY POTENTIAL TO POTCAR FILE
        kmax = float( linecache.getline(potcarfile,potcarjump) ) # read kmax
        potcar = self.ReadPotcarfile(potcarfile,potcarjump, nk)  # read local part op potcar

        # add self energy
        ca = 0.0
        newpotcar = potcar.copy()
        for i in range(nk):
            ca = ca + kmax/nk
            newpotcar[i] = newpotcar[i] + self.Add2PotcarFourier(beta*ca,Nrad,radii,Vs,CUT,0.0,0.0,0.0)/(beta*ca)


        # MAKE NEW POTCAR
        nrows = int(np.ceil(nk / 5))
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

        return radii, pot_xi, pot_zeta, Vs, newpotcar
    
   
    def ReadRadii(self,file,nval):
        # special case of ReadPotfile maybe we don't need a seperate function
        nrows = np.ceil(nval/4)
        radii = pd.read_csv(file, nrows=nrows, delim_whitespace=True, skiprows=1, header=None)
        radii = np.concatenate(radii.to_numpy())    # convert to numpy and concatenate to single array
        radii = radii[~np.isnan(radii)]             # Remove Nan element from array. These elements appear when the last
                                                    # line in the potential file does not consist of 4 numbers
        if radii.shape[0] != nval:
            raise Exception('An unexpected amount of NaN were removed while reading the radii file')

        return radii

    def ReadPotfile(self,file,nrows=None,nval=None,skiprows=0):
        """
        Reads and atom potential file
        :param file: filename/ location
        :param nval: number of values which will be read
        :param skiprows: number of rows which should be skipped in the beginning of the file
        :return: numpy array of potential
        """
        # read potential file generated by atom and return potentials as numpy array
        if nrows==None:
            nrows = np.ceil(nval/4)
        pot = pd.read_csv(file, nrows=nrows, delim_whitespace=True, skiprows=skiprows, header=None)
        pot = np.concatenate(pot.to_numpy())    # convert to numpy and concatenate to single array
        pot = pot[~np.isnan(pot)]               # Remove Nan element from array. These elements appear when the last
                                                # line in the potential file does not consist of 4 numbers
        return pot

    def Add2PotcarFourier(self,ca,nrad,radii,Vs,Cut,inicio,inicial,final):
        # based on the add2POTCAR-eng.f90 fortran script
        # this is quite slow 
        fourier = 0
        iniciovez = 0
        for i,r in enumerate(radii):
            if r <= 0:
                continue
                # nothting
            elif iniciovez == 0:
                fourier = fourier + (Vs[i]*np.sin(ca*r) + inicial*np.sin(ca* np.sin(ca*inicio)))*(r-inicio)/2.0
                iniciovez = 1
                continue
            elif r > Cut:
                fourier = fourier + (Vs[i]*np.sin(ca*r) + final*np.sin(ca*np.sin(ca*Cut)))*(Cut-r)/2.0
                break
            else:
                fourier = fourier + (Vs[i]*np.sin(ca*r) + Vs[i-1]*np.sin(ca*radii[i-1]))*(r-radii[i-1])/2.0
        return fourier

    def  Calcnrows(self,potfile):
        with open(potfile) as pfile:
            for i,line in enumerate(pfile):
                if i ==0:
                    continue
                elif line[0] != ' ':
                    return i-1
