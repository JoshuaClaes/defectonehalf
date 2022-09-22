import AtomWrapper
import potcarsetup
import orbital


AW = AtomWrapper.AtomWrapper()

# CARBON test
Occupation = [{'n' : 2 ,'l' : 0, 'occupation' : 1.75},{'n' : 2, 'l' : 1, 'occupation' : 1.75}]
AW.MakeInputFile('/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/carbon','C',[1,2],Occupation,EXtype='pb')
AW.RunATOM('/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/carbon')

# NITROGEN TEST
Occupation = [{'n' : 2 ,'l' : 0, 'occupation' : 2.00},{'n' : 2, 'l' : 1, 'occupation' : 3.00}]
AW.MakeInputFile('/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/nitrogen','N',[1,2],Occupation,EXtype='pb')
AW.RunATOM('/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/nitrogen')

# both calculation produce the same result as was previously obtained
ps = potcarsetup.potcarsetup('test_potcarsetup')
atomname= 'Ctestatom'
atom= 'C'
orbitals= [1, 2]
GSorb = [orbital.orbital(n=2,l=0,occ=2.00), orbital.orbital(n=2,l=1,occ=2.00)]
xi    = [0.25,0.25]
zeta  = [0.00,0.00]
#ps.CalcSelfEnPot(atomname,atom,orbitals,GSorb,xi,zeta)

# TEST ADD2POTCAR
Nrad= 1006
radfile = '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/test_AW/VTOTAL1_normal'
potfile_zeta =  '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/test_AW/VTOTAL1_normal'
potfile_xi =  '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/test_AW/VTOTAL1_sp_quarter'
CutFuncPar= [8, 2.5]
potcarfile= '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/test_AW/POTCAR'
nk = 1000
potcarjump = 62
newpotcarfile = '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/test_AW/newPOTCAR2'
radii, pot_xi, pot_zeta, Vs, potcar = AW.Add2Potcar(Nrad, radfile, potfile_zeta, potfile_xi, CutFuncPar, potcarfile, nk, potcarjump,newpotcarfile)
#print(radii)
#print(pot_xi)
#print(radii.shape)
#print(pot_xi.shape)
print(Vs)
#plt.figure(figsize=(14,10))
#plt.plot(radii,Vs)
#plt.vlines(2.5,np.min(Vs),np.max(Vs))
#plt.xlim([0,5])
#plt.show()

#print(potcar)