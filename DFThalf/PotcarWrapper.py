def FindKmax(potcarfile):
    with open(potcarfile) as pfile:
        for i,line in enumerate(pfile):
            if line == ' local part':
                kmax = float(pfile[i+1])
                linekmax = i+1;
                return kmax, linekmax

    raise Exception('kmax was not found in the given potcar file!')
