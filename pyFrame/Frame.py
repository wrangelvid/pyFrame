import numpy as np
from pyFrame.Node import Node
from pyFrame.Member import Member 
from pyFrame.Loads import NodalForce, NodalMoment
from matplotlib import pyplot as plt

class Frame(object):
    def __init__(self):
        self.Nodes = {}
        self.Members = {}
        self.NodalLoads = []

        self._K = None
        
    def addNode(self, Name, x, y, z):
        node = Node(Name, x, y, z )
        if Name in self.Nodes:
            print(f"warning! {Name} is already defined. this is updating the node.")
        self.Nodes.update({Name: node})
        
    def delNode(self, Name):
        #TODO need to delete all assigned members
        del self.Nodes[Name]

    def addMember(self, Name, nNode_name, pNode_name, E, G, J, Iy,Iz, A):
        """
            Adding member to frame.
            :param Name: unique member name given by user 
            :type Name: String
            :param nNode_name: name idenfitifer to attech to the negative end of the member
            :type nNode_name: string
            :param pNode_name: name idenfitifer to attech to the positive end of the member
            :type pNode_name: string
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

        """
        if Name in self.Members:
            print(f"warning! {Name} is already defined. this is updating the Member.")

        nNode = self.Nodes[nNode_name]
        pNode = self.Nodes[pNode_name]
        member = Member(Name, nNode, pNode, E, G, J, Iy,Iz, A)

        self.Members.update({Name: member})
    
    def makeSupport(self, Node_name, Ux = None, Uy = None, Uz = None, Rx = None, Ry = None, Rz = None):
        """
            Constraints a Node in its DoF
            Default is None, which means no support in translation/rotation for the given direction.

            :param Ux: any number for a fixed displacement, eg. 0 for standard support. None if not restrained in x 
            :type Ux: float  
            :param Uy: any number for a fixed displacement, eg. 0 for standard support. None if not restrained in y 
            :type Uy: float 
            :param Uz: any number for a fixed displacement, eg. 0 for standard support. None if not restrained in z 
            :type Uz: float 
            :param Rx: any number for a fixed rotation, eg. 0 for standard support. None if not restrained around x 
            :type Rx: float
            :param Ry: any number for a fixed rotation, eg. 0 for standard support. None if not restrained around y 
            :type Ry: float
            :param Rz: any number for a fixed rotation, eg. 0 for standard support. None if not restrained around z 
            :type Rz: float
        """
         
        node = self.Nodes[Node_name]

        #global displacement relation
        node.Ux = Ux
        node.Uy = Uy
        node.Uz = Uz

        node.Rx = Rx 
        node.Ry = Ry 
        node.Rz = Rz 

    def makeRelease(self, Node_name, Ux = None, Uy = None, Uz = None, Rx = None, Ry = None, Rz = None):

        node = self.Nodes[Node_name]
        #TODO
        pass
    
    def addNodeLoad(self, Node_name, Force_name = None, Fx = 0, Fy = 0, Fz = 0):
        #get node
        node = self.Nodes[Node_name] 
        #initilze Nodal Force
        nForce = NodalForce(node, Force_name, Fx, Fy, Fz)
        self.NodalLoads.append(nForce)


    def addNodeMoment(self, Node_name, Force_name = None, Mx = 0, My = 0, Mz = 0):
        #get node
        node = self.Nodes[Node_name] 
        #initilze Nodal Moment
        nMoment = NodalMoment(node, Force_name, Mx, My, Mz)
        self.NodalLoads.append(nMoment)


    
    def _reassing_ids(self):
 
        # Number each node in the model
        i = 0
        for _,node in self.Nodes.items():
            node.ID = i
            i += 1
        
        # Number each member in the model
        i = 0
        for _,member in self.Members.items():
            member.ID = i
            i += 1
    
    def _compute_K(self):
        #compute K
        max_n = len(self.Nodes)*6
        Kp = np.zeros((max_n,max_n))
        for _,mbr in self.Members.items():
            #get global partitioned member stiffness matrix
            KAA, KAB, KBA, KBB = mbr.Kg

            n = mbr.nNode.ID*6
            p = mbr.pNode.ID*6

            Kp[n:n+KAA.shape[0], n:n+KAA.shape[1]] += KAA
            Kp[n:n+KAB.shape[0], p:p+KAB.shape[1]] += KAB
            Kp[p:p+KBA.shape[0], n:n+KBA.shape[1]] += KBA
            Kp[p:p+KBB.shape[0], p:p+KBB.shape[1]] += KBB
        
        return Kp 

    def __partition_Kp(self, Kp, U1_DoF_idx, U2_DoF_idx):
        '''
        partition primary stiffness matrix into submatrices b/c of DoF indices 
        '''
        #partition matrix
        K11 = Kp[U1_DoF_idx, :][:, U1_DoF_idx]
        K12 = Kp[U1_DoF_idx, :][:, U2_DoF_idx]
        K21 = Kp[U2_DoF_idx, :][:, U1_DoF_idx]
        K22 = Kp[U2_DoF_idx, :][:, U2_DoF_idx]
        return K11, K12, K21, K22

    def __partition_vec(self, vec, U1_DoF_idx, U2_DoF_idx):
        if vec.shape[1] == 1:
            #partition vector
            V1 = vec[U1_DoF_idx, :]
            V2 = vec[U2_DoF_idx, :]
            return V1, V2
        else:
            raise Exception('Input vector has to be column vector')

    def _partition_idx(self):
        U1_DoF_idx = [] #indecies of all unkown DoF
        U2_DoF_idx = [] #indecies of all kown DoF
        U2 = [] #known displacements
        for _, node in self.Nodes.items():
            
            if not node.Ux is None:
                #support displacement is prescripbed
                U2_DoF_idx.append(node.ID*6 + 0)
                U2.append(node.Ux)
            else:
                #unkown Displacement
                U1_DoF_idx.append(node.ID*6 + 0)

            if not node.Uy is None:
                #support displacement is prescripbed
                U2_DoF_idx.append(node.ID*6 + 1)
                U2.append(node.Uy)
            else:
                #unkown Displacement
                U1_DoF_idx.append(node.ID*6 + 1)

            if not node.Uz is None:
                #support displacement is prescripbed
                U2_DoF_idx.append(node.ID*6 + 2)
                U2.append(node.Uz)
            else:
                #unkown Displacement
                U1_DoF_idx.append(node.ID*6 + 2)

            if not node.Rx is None:
                #support rotation is prescripbed
                U2_DoF_idx.append(node.ID*6 + 3)
                U2.append(node.Rx)
            else:
                #unkown rotation 
                U1_DoF_idx.append(node.ID*6 + 3)

            if not node.Ry is None:
                #support rotation is prescripbed
                U2_DoF_idx.append(node.ID*6 + 4)
                U2.append(node.Ry)
            else:
                #unkown rotation 
                U1_DoF_idx.append(node.ID*6 + 4)

            if not node.Rz is None:
                #support rotation is prescripbed
                U2_DoF_idx.append(node.ID*6 + 5)
                U2.append(node.Rz)
            else:
                #unkown rotation 
                U1_DoF_idx.append(node.ID*6 + 5)
 
        U2 = np.array(U2).T
            
        return U1_DoF_idx, U2_DoF_idx, U2
    
    def _compute_PE(self):
        #nodal force vector
        PE = np.zeros((len(self.Nodes)*6,1))

        for nLoad in self.NodalLoads:
            base_idx = nLoad.Node.ID*6
            if type(nLoad) == NodalMoment:
                base_idx +=3

            PE[base_idx: base_idx+3] += nLoad.vector() 
        
        return PE
        
    def analyze(self):
        """

            PE1    K11 | K12   U1   PI1
            ---  = ----|---- @ -- + ---
            PE2    K21 | K22   U2   PI2

            U1 contains the unknown displacements 
            U2 contains the prescribed support movements/rotations

            PE1 Nodal force Vector 
            PE2 Nodal force Vector

            PI1 contains prescribed joint loads
            PI2 contains unkown reactants

            K is the global stiffness Matrix, which is partioned according to known and unkown elements
            
            Equations to solve:
            U1 = inv(K11)@(F1 - K12@U2 - P1)
        """
        #assign internatl ids 
        self._reassing_ids()

        #compute primary K matrix 
        Kp = self._compute_K()
        
        #determine how to partition the stiffness matrix and vectors given the restrained Degrees of Freedom (DoF)
        U1_DoF_idx, U2_DoF_idx, U2 = self._partition_idx()

        #get global nodal force vector
        PE = self._compute_PE()
        #partition PE
        PE1, PE2 = self.__partition_vec(PE, U1_DoF_idx, U2_DoF_idx)


        
        K11, K12, K21, K22 = self.__partition_Kp(Kp, U1_DoF_idx, U2_DoF_idx)

        #Calculate the unknown displacements U1
        U1 = np.linalg.inv(K11)@(PE1 - K12@U2)# - PI1) TODO implement internal point loads later

        #build node_id to node_name map to save computation
        nID_to_nName = dict(map(lambda item: (item[1].ID, item[0]), self.Nodes.items()))
        for idx, DoF_idx in enumerate(U1_DoF_idx):
            remainder = DoF_idx % 6
            id = (DoF_idx - remainder)/6
            #access node 
            node = self.Nodes[nID_to_nName[id]]
            
            if remainder == 0:
                node.Ux = U1[idx,0]

            if remainder == 1:
                node.Uy = U1[idx,0]

            if remainder == 2:
                node.Uz = U1[idx,0]
            
            if remainder == 3:
                node.Rx = U1[idx,0]

            if remainder == 4:
                node.Ry = U1[idx,0]

            if remainder == 5:
                node.Rz = U1[idx,0]


    def plot(self, label_offset=0.01, xMargin=0.25, yMargin=0.25, zMargin=0.5, elevation=20, rotation=35, deformed = True): 
    
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

        #Plot members
        for _,mbr in self.Members.items():  
            nNode_pos = mbr.nNode.pos().tolist()
            pNode_pos = mbr.pNode.pos().tolist()
            axes.plot3D(*zip(nNode_pos, pNode_pos),'b') #Plot 3D member

            #plot deformed members
            #TODO somehow implement rotation
            if deformed:
                ndeformation = np.array([mbr.nNode.Ux, mbr.nNode.Uy, mbr.nNode.Uz])
                pdeformation = np.array([mbr.pNode.Ux, mbr.pNode.Uy, mbr.pNode.Uz])
                nNode_pos = (mbr.nNode.pos() + ndeformation).tolist()
                pNode_pos = (mbr.pNode.pos() + pdeformation).tolist()
                axes.plot3D(*zip(nNode_pos, pNode_pos),'r') #Plot 3D member



        #Plot nodes
        maxX =float("-inf") 
        maxY =float("-inf") 
        maxZ =float("-inf") 
        minX =float("inf") 
        minY =float("inf") 
        minZ =float("inf") 
 
        for name, node in self.Nodes.items():
            if node.x > maxX:
                maxX = node.x
            if node.x < minX:
                minX = node.x

            if node.y > maxY:
                maxY = node.y
            if node.y < minY:
                minY = node.y

            if node.z > maxZ:
                maxZ = node.z
            if node.z < minZ:
                minZ = node.z

            axes.plot3D([node.x],[node.y],[node.z],'bo',ms=6) #Plot 3D node
            axes.text(node.x+dx, node.y+dy, node.z+dz, name, fontsize=16) #Add node label

            if deformed:
                axes.plot3D([node.x + node.Ux],[node.y + node.Uy],[node.z + node.Uz],'ro',ms=6) #Plot 3D node

        #Set axis limits to provide margin around structure
        axes.set_xlim([minX-x_margin,maxX+x_margin])
        axes.set_ylim([minY-y_margin,maxY+y_margin])
        axes.set_zlim([minZ,maxZ+z_margin])

        axes.set_xlabel('X-coordinate (m)')
        axes.set_ylabel('Y-coordinate (m)')
        axes.set_zlabel('Z-coordinate (m)')
        axes.set_title('Structure to analyse')
        axes.grid()
        plt.show()






        
