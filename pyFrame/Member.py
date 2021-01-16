import numpy as np
from math import isclose

class Member(object):
    def __init__(self, Name, nNode, pNode, E, G, J, Iy,Iz, A):
        """
            :param Name: unique member name given by user 
            :type Name: String
            :param nNode: refernece to negative Node attached to member
            :type nNode: Node 
            :param pNode: reference to positve Node attached to member
            :type pNode: Node 
            :param E: Elastic Moduls of Member
            :type E: float
            :param G: Shear Moduls of Member
            :type G: float
            :param J: torsional constant
            :type J: float
            :param Iy:
            :type Iy: float
            :param Iz:
            :type Iz: float
            :param A:
            :type A: float

            
            local reference frame 
              z
              |
              | 
              |
              0[========]--x
             /     
            y

        """
        self.ID = None #Internal ID 
        self.Name = Name
        self.nNode = nNode
        self.pNode = pNode
        self.E = E
        self.G = G
        self.J = J 
        self.A = A
        self.Iy = Iy  
        self.Iz = Iz  

        #Orientation
        self._L = None #length of Member 
        self._R = None

        #releases of degree of freedom

        #releases on the negative end 
        self.nReleases = [False, False, False, False, False, False]

        #releases on the positive end 

        self.pReleases = [False, False, False, False, False, False]

        #stiffness matrix 
        self._Kl_unc = None
        self._Kl = None
        
    @property
    def L(self):
        if self._L is None:
            self._L = self._compute_L()

        return self._L


    @property
    def R(self):
        """
            Returns the Rotation Matrix to transform from local to Global Coordinates and vica versa
        """

        if self._R is None:
            self._R = self._compute_R()
   
        return self._R  
    
    @property
    def Kl_unc(self):
        """
            Local uncondensed Stiffness Matrix 
        """
        if self._Kl_unc is None:
            self._Kl_unc = self._compute_Kl_unc()

        return self._Kl_unc

    @property
    def Kl(self):
        """
            Local condensed Stiffness Matrix 
        """
        if self._Kl is None:
            self._Kl = self._compute_Kl()

        return self._Kl
    
    
    @property
    def Kg(self):
        """
            paritioned Global Stiffness Matrix
        """
        #not storing because we only need to get the global stiffness matrix for the prime stiffness matrix of the frame
        
        K = self.R.T@self.Kl@self.R

        return self._partition(K)
    
    def _partition(self, m):
        """
            Partition matrixes according to releases 
        """
        released_idx = []
        unreleased_idx = []

        for idx, release in list(enumerate(self.nReleases)) + list(enumerate(self.pReleases, start=6)):
            if release:
                released_idx.append(idx)
            else:
                unreleased_idx.append(idx)
       
        #partition matrix 
        m11 = m[unreleased_idx, :][:, unreleased_idx]
        m12 = m[unreleased_idx, :][:, released_idx]
        m21 = m[released_idx, :][:, unreleased_idx]
        m22 = m[released_idx, :][:, released_idx]
        return m11, m12, m21, m22 


    def _compute_L(self):
        return np.sum(np.square(self.pNode.pos()-self.nNode.pos()))**0.5

    def _compute_R(self):
        nX,nY,nZ = self.nNode.pos()
        pX,pY,pZ = self.pNode.pos()

        #get cosine directions
        x = [(pX-nX)/self.L, (pY-nY)/self.L, (pZ-nZ)/self.L]

        if isclose(nX, pX) and isclose(nZ, pZ):
            #vertical member
            if pY > nY:
                y = [-1, 0, 0]
                z = [0, 0, 1]
            else:
                y = [1, 0, 0]
                z = [0, 0, 1]

        elif isclose(nY, pY):
            # Horizontal members
            y = [0, 1, 0]
            z = np.cross(x, y)

            # make z unit vector
            z = z / np.linalg.norm(z)

        else:
            # Members neither vertical or horizontal

            # Find the projection of x on the global XZ plane
            proj = [pX-nX, 0, pZ-nZ]

            if pY > nY:
                z = np.cross(proj, x)
            else:
                z = np.cross(x, proj)

            # make z unit vector
            z = z / np.linalg.norm(z)

            # Find the direction cosines for the local y-axis
            y = np.cross(z, x)
            y = y / np.linalg.norm(y)

        
        # the direction cosine matrix
        dirCos = np.array([x, y, z])
        transMatrix = np.zeros((12, 12))
        transMatrix[0:3, 0:3] = dirCos
        transMatrix[3:6, 3:6] = dirCos
        transMatrix[6:9, 6:9] = dirCos
        transMatrix[9:12, 9:12] = dirCos
        
        return transMatrix
        #tmp = np.concatenate((dirCos,dirCos),axis=1)
        #return np.concatenate((tmp,tmp), axis=0)
 
    
    def _compute_Kl_unc(self):
        E = self.E
        G = self.G
        J = self.J

        A = self.A
        Iy = self.Iy
        Iz = self.Iz
        L = self.L
        
        KlAA = np.array([
                        [A*E/L,             0,             0,    0,             0,            0],
                        [    0,  12*E*Iz/L**3,             0,    0,             0,  6*E*Iz/L**2],
                        [    0,             0,  12*E*Iy/L**3,    0,  -6*E*Iy/L**2,            0],
                        [    0,             0,             0, G*J/L,            0,            0],
                        [    0,             0,  -6*E*Iy/L**2,    0,      4*E*Iy/L,            0],
                        [    0,   6*E*Iz/L**2,             0,    0,             0,     4*E*Iz/L]])
            
        KlAB = np.array([
                        [-A*E/L,             0,              0,      0,            0,              0],
                        [     0, -12*E*Iz/L**3,              0,      0,            0,    6*E*Iz/L**2],
                        [     0,             0,  -12*E*Iy/L**3,      0, -6*E*Iy/L**2,              0], 
                        [     0,             0,              0, -G*J/L,            0,              0],
                        [     0,             0,    6*E*Iy/L**2,      0,     2*E*Iy/L,              0],
                        [     0,  -6*E*Iz/L**2,              0,      0,            0,       2*E*Iz/L]])

        KlBA = np.array([
                        [-A*E/L,              0,              0,      0,            0,              0],
                        [     0,  -12*E*Iz/L**3,              0,      0,            0,   -6*E*Iz/L**2],
                        [     0,              0,  -12*E*Iy/L**3,      0,  6*E*Iy/L**2,              0],
                        [     0,              0,              0, -G*J/L,            0,              0],
                        [     0,              0,   -6*E*Iy/L**2,      0,     2*E*Iy/L,              0],
                        [     0,    6*E*Iz/L**2,              0,      0,            0,       2*E*Iz/L]])

        KlBB = np.array([
                        [A*E/L,            0,              0,       0,             0,              0],
                        [    0, 12*E*Iz/L**3,              0,       0,             0,   -6*E*Iz/L**2],
                        [    0,            0,   12*E*Iy/L**3,       0,   6*E*Iy/L**2,              0],
                        [    0,            0,              0,   G*J/L,             0,              0],
                        [    0,            0,    6*E*Iy/L**2,       0,      4*E*Iy/L,              0],
                        [    0, -6*E*Iz/L**2,              0,       0,             0,       4*E*Iz/L]])
   
        #concatenate matrices to full local stiffness matrix
        top = np.concatenate((KlAA,KlAB),axis=1)
        btm = np.concatenate((KlBA,KlBB),axis=1)

        return np.concatenate((top,btm), axis=0)

  

    def _compute_Kl(self):
        """
            compute condensed stiffness matrix
        """
        KlAA_unc,KlAB_unc,KlBA_unc,KlBB_unc = self._partition(self.Kl_unc)

        #condensing the matrix
        Kl = KlAA_unc - (KlAB_unc @ np.linalg.inv(KlBB_unc)) @ KlBA_unc

        #For each released DoF we add zero row and a zero column at the appropiate index
        for idx, release in list(enumerate(self.nReleases)) + list(enumerate(self.pReleases, start=6)):
            if release:
                Kl = np.insert(Kl, idx, 0, axis = 0)
                Kl = np.insert(Kl, idx, 0, axis = 1)

        return Kl
