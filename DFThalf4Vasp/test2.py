
import PotcarWrapper

# TEST POTCAR WRAPPER
potcarfile = '/mnt/extradata/DFThalf4Vasp/SelfEnergyPot_Auto/test/test_AW/POTCAR'
kmax, linekmax = PotcarWrapper.FindKmax(potcarfile)
print('kmax=',kmax)
print('linemax',linekmax)

nrows = PotcarWrapper.FindNrows(potcarfile)
print('Nrows=',nrows)

potcar = PotcarWrapper.ReadPotcarfile(file = potcarfile)
print('Potcar local part\n', potcar)