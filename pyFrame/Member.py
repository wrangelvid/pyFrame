import numpy as np
from math import isclose
from pyFrame.Loads import MemberPtForce, MemberPtMoment
from matplotlib import pyplot as plt

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

        #initilize member specific loads
        self.ptLoads = []

        #store Member Forces
        self._Fl = None
        self._Fg = None
        
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
            Global Stiffness Matrix
        """
        #not storing because we only need to get the global stiffness matrix for the prime stiffness matrix of the frame
        
        return self.R.T@self.Kl@self.R

    @property
    def Ug(self):
        """
            global displacement vector
        """
        Ug = np.zeros((12,1))
        #retrieve displacements of negative node
        Ug[0,0] = self.nNode.Ux 
        Ug[1,0] = self.nNode.Uy 
        Ug[2,0] = self.nNode.Uz 
        Ug[3,0] = self.nNode.Rx 
        Ug[4,0] = self.nNode.Ry 
        Ug[5,0] = self.nNode.Rz 

        #retrieve displacements of positve node
        Ug[6,0] = self.pNode.Ux 
        Ug[7,0] = self.pNode.Uy 
        Ug[8,0] = self.pNode.Uz 
        Ug[9,0] = self.pNode.Rx 
        Ug[10,0] = self.pNode.Ry 
        Ug[11,0] = self.pNode.Rz 
        return Ug

    @property
    def Ul(self):
        """
            local displacement vector
        """
        return self.R@self.Ug

    @property
    def PIl_unc(self):
        """
            uncondensed local force vector due to meber fixed end action
        """
        return self._compute_PIl_unc()
    
    
    @property
    def PIl(self):
        """
            condensed local force vector due to meber fixed end action
        """
        
        return self._compute_PIl()

    @property
    def Fl(self):
        """
           local force vector due to meber fixed end action
        """           
        if self._Fl is None:
            self._Fl = self._compute_Fl()

        return self._Fl
    
    
    @property
    def Fg(self):
        """
            condensed local force vector due to meber fixed end action
        """
        if self._Fg is None:
            self._Fg = self._compute_Fg()

        return self._Fg
    
    
    @property
    def PIg(self):
        """
            global force vector due to meber fixed end action
        """
        return np.linalg.inv(self.R)@self.PIl

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

        if  m.shape[1] ==1:
            #matrix is a vector
            v1 = m[unreleased_idx,:]
            v2 = m[released_idx,:]
            return v1, v2
        else:
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
            # Find the projection of x on the global Z axis
            proj = [0, 0, pZ-nZ]

            if pZ > nZ:
                y = np.cross(proj, x)
            else:
                y = np.cross(x, proj)

            # make y unit vector
            y = y / np.linalg.norm(y)

            # Find the direction cosines for the local z-axis
            z = np.cross(x,y)

            z = z / np.linalg.norm(z)
       
        # the direction cosine matrix
        dirCos = np.array([x, y, z])
        return np.kron(np.eye(4),dirCos)
   
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

    def _compute_PIl_unc(self):
        '''
            uncondensed local nodal force vector due to member fixed end actions 
        '''

        PIl = np.zeros((12,1))
        L = self.L 

        for mPtLoad in self.ptLoads:
            nX = mPtLoad.x
            pX = L - nX
            if type(mPtLoad) == MemberPtForce:
                
                #negative end
                PIl[0,0]  += -mPtLoad.Fx*pX/L #axial load
                PIl[1,0]  += -mPtLoad.Fy*pX**2*(L+2*nX)/L**3
                PIl[2,0]  += -mPtLoad.Fz*pX**2*(L+2*nX)/L**3
                PIl[4,0]  +=  mPtLoad.Fz*pX**2*nX/L**2
                PIl[5,0]  += -mPtLoad.Fy*pX**2*nX/L**2

                #positve end
                PIl[6,0]  += -mPtLoad.Fx*nX/L #axial load
                PIl[7,0]  += -mPtLoad.Fy*nX**2*(L+2*pX)/L**3
                PIl[8,0]  += -mPtLoad.Fz*nX**2*(L+2*pX)/L**3
                PIl[10,0] += -mPtLoad.Fz*nX**2*pX/L**2
                PIl[11,0] +=  mPtLoad.Fy*nX**2*pX/L**2

            if type(mPtLoad) == MemberPtMoment:
                
                #negative end
                PIl[1,0]  +=  mPtLoad.Mz*6*nX*pX/L**3 
                PIl[2,0]  += -mPtLoad.My*6*nX*pX/L**3
                PIl[3,0]  += -mPtLoad.Mx*pX/L # Torque
                PIl[4,0]  +=  mPtLoad.My*pX*(2*nX-pX)/L**2 
                PIl[5,0]  +=  mPtLoad.Mz*pX*(2*nX-pX)/L**2 

                #positve end
                PIl[7,0]  += -mPtLoad.Mz*6*pX*nX/L**3
                PIl[8,0]  +=  mPtLoad.My*6*pX*nX/L**3
                PIl[9,0]  += -mPtLoad.Mx*nX/L # Torque
                PIl[10,0] +=  mPtLoad.My*nX*(2*pX-nX)/L**2
                PIl[11,0] +=  mPtLoad.Mz*nX*(2*pX-nX)/L**2
            
        return PIl


    def _compute_PIl(self):
        '''
            condensed local nodal force vector due to member fixed end actions 
        '''
        
        # Partition the local stiffness matrix
        KlAA_unc,KlAB_unc,KlBA_unc,KlBB_unc = self._partition(self.Kl_unc)
        #partition internal nodal forces (fixed end reactions)
        PIl1_unc, PIl2_unc = self._partition(self.PIl_unc)
        
        #condense vector
        PIl = PIl1_unc - (KlAB_unc@np.linalg.inv(KlBB_unc))@PIl2_unc
        
        #For each released DoF we add zero row at the appropiate index
        for idx, release in list(enumerate(self.nReleases)) + list(enumerate(self.pReleases, start=6)):
            if release:
                PIl = np.insert(PIl, idx, 0, axis = 0)

        return PIl
        
    def _compute_Fl(self):
        """
            compute local end force vector
        """
        return self.Kl@self.Ul + self.PIl
    
    def _compute_Fg(self):
        """
            compute global end force vector
        """
        return np.linalg.inv(self.R)@self.Fl

    def plot(self, label_offset=0.01, xMargin=0.25, yMargin=0.25, zMargin=0.5, elevation=20, rotation=35, deformed = True, xFac = 1.0): 
    
        fig = plt.figure() 
        axes = fig.add_axes([0.1,0.1,3,3],projection='3d') #Indicate a 3D plot 
        axes.view_init(elevation, rotation) #Set the viewing angle of the 3D plot

        #Set offset distance for node label
        dx = label_offset #x offset for node label
        dy = label_offset #y offset for node label
        dz = label_offset #z offset for node label

        #Provide space/margin around structure
        x_margin = xMargin #x-axis margin
        y_margin = yMargin #y-axis margin
        z_margin = zMargin #z-axis margin

        #everything is plotted in local coordinates 

        #plot nodes in local coordinates
        axes.plot3D([0],[0],[0],'bo',ms=6) #negative node
        axes.text(-dx, dy, dz, self.nNode.Name, fontsize=16)

        axes.plot3D([self.L],[0],[0],'bo',ms=6) #positve node
        axes.text(self.L + dx, dy, dz, self.pNode.Name, fontsize=16)

        #plot member
        axes.plot3D(*zip([0,0,0], [self.L,0,0]),'b')
        axes.text(*[self.L/2,dy,dz], self.Name, fontsize=16)

        #plot member point forces
        for mLoad in filter(lambda load: type(load) == MemberPtForce ,self.ptLoads):
            mag = np.sum(np.square(mLoad.vector().T))**0.5
            #axes.quiver(*[mLoad.x,0,0],*mLoad.vector().tolist(), length=0.5, normalize=True, pivot='tip')

        #plot member local end forces
        #TODO show moments
        nFl = self.Fl[:3,0].T*np.eye(3)
        pFl = self.Fl[9:,0].T*np.eye(3)

        axes.quiver(*[0,0,0],*nFl[:,0], length=0.1, normalize=True, pivot='tip', colors='r')
        axes.quiver(*[0,0,0],*nFl[:,1], length=0.1, normalize=True, pivot='tip', colors='r')
        axes.quiver(*[0,0,0],*nFl[:,2], length=0.1, normalize=True, pivot='tip', colors='r')

        axes.quiver(*[self.L,0,0],*pFl[:,0], length=0.1, normalize=True, pivot='tip', colors='r')
        axes.quiver(*[self.L,0,0],*pFl[:,1], length=0.1, normalize=True, pivot='tip', colors='r')
        axes.quiver(*[self.L,0,0],*pFl[:,2], length=0.1, normalize=True, pivot='tip', colors='r')


        if deformed:
            ndeformation = xFac*self.Ul[:3,0]
            pdeformation = xFac*self.Ul[6:9 ,0]
            nNode_pos = (np.array([0,0,0]) + ndeformation).tolist()
            pNode_pos = (np.array([self.L,0,0]) + pdeformation).tolist()
            axes.plot3D(*zip(nNode_pos,pNode_pos),'r')
        

        axes.set_xlim([-x_margin,self.L+x_margin])
        axes.set_ylim([-y_margin,y_margin])
        axes.set_zlim([0,z_margin])

        axes.set_xlabel('X-coordinate (m)')
        axes.set_ylabel('Y-coordinate (m)')
        axes.set_zlabel('Z-coordinate (m)')
        axes.set_title(f'Beam {self.L}')
 
        axes.grid()
        plt.show()