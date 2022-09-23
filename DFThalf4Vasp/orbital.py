class orbital:
    def __init__(self,n,l,occ):
        self.n = n
        self.l = l
        self.occ = occ

    def as_dict(self):
        orbitaldict = {'n':self.n,'l':self.l,'occupation':self.occ}
        return orbitaldict

    def __add__(self, other):
        # return new orbital object with sum of both occupations
        if self.n == other.n and self.l == other.l :
            newocc = self.occ + other.occ
            return orbital(self.n,self.l,newocc)
        else:
            raise ValueError('Orbitals characters n and l do not match')

    def __sub__(self,other):
        # return new orbital object with sum of both occupations
        if self.n == other.n and self.l == other.l :
            newocc = self.occ - other.occ
            if newocc >= 0:
                return orbital(self.n,self.l,newocc)
            else:
                raise ValueError('Occupation is negative!')
        else:
            raise ValueError('Orbitals characters n and l do not match!')