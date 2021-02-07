import numpy as np
from matplotlib import pyplot as plt

class BeamSeg(object):
    def __init__(self, segType,  x1, x2, EA = None, EI = None):
        """
            :param segType: tells us the local directino of the analysis. Could be ['X', 'Y', 'Z']
            :type segType: string
            :param x1: realtive start location of the beam 
            :type x1: float
            :param x2: realtive end location of the beam
            :type x2: float
            :param EI: Flexuarl stiffness of the beam 
            :type EI: float
            :param EA: Axial stiffness of the Beam
            :type EA: float 
        """
        self.segType = segType
        self.x1 = x1   
        self.x2 = x2 
        self.EI = EI 
        self.EA = EA

        self.theta1 = None # Slope at start of beam segment
        self.delta1 = None #displacement at start of segment
        self.delta_x1 = None #axial displacement at start of segment
        self.S1 = None # Internal shear force at start of segment
        self.M1 = None # Internal moment at start of segment
        self.P1 = None # Internal axial force at start of segment
        self.T1 = None # Torsional moment at start of segment

        self._L = None #length of Segment 

    @property
    def L(self):
        """
            returns segments length
        """
        if self._L is None:
            self._L = self._compute_L()

        return self._L


    def _compute_L(self):
        return self.x2 - self.x1

    def Slope(self, x = None):
        """
            Returns the slope at a point on the segment at x

            :param x: location where slope should be calculated (relative to start of segment) 
            :type x: float

            :return type: float
        """  
        if x is None:
            x = self.L
        
        if self.segType == 'Z':
            return self.theta1 - (self.M1*x - self.S1*x**2/2)/(self.EI)
        else:
            return self.theta1 - (self.M1*x + self.S1*x**2/2)/(self.EI)
    
    def Deflection(self, x = None):
        """
            Returns the deflection at a location on the segment
        """

        if x is None:
            x = self.L

        if self.segType == 'Z': 
            return self.delta1 + self.theta1*x - self.M1*x**2/(2*self.EI) + self.S1*x**3/(6*self.EI) 
        else:
            return self.delta1 - self.theta1*x + self.M1*x**2/(2*self.EI) + self.S1*x**3/(6*self.EI) 




    def AxialDeflection(self, x = None):
        """
            Returns the axial deflection at a location on the segment
            The x is on the local segments coordinates and does not correspond to the members x
        """
        if x is None:
            x = self.L

        return self.delta_x1 - 1/self.EA*(self.P1*x)

    def Shear(self, x):
        """
            Returns the shear force at a location on the segment
        """
        #currently shear is constant since we have no distributed loads
        return self.S1

    def Moment(self, x):
        """
            Returns the moment at a location on the segment
        """
        return self.M1 - self.S1*x

    def Axial(self, x):
        """
            Returns the axial force at a location on the segment
        """
        #currently axial load is constant sine we have no distriubed loads
        return self.P1

    def Torsion(self):
        """
            Returns the torsional moment in the segment.
        """
        # Here torsinal moment is constant across the segment
        return self.T1

class Material(object):
    def __init__(self, E, G):
        """
            :param E: Elastic Moduls of Member
            :type E: float
            :param G: Shear Moduls of Member
            :type G: float
 
        """
        self.E = E
        self.G = G



class Crosssection(object):
    def __init__(self, points):
        """
            :param points: list of (y,z) coordinate points describinh a polygon boundary
            :type points: list of typles
        local reference frame 

              z
              |
              | 
              |
              0-------y
        """
        self.pts = np.asarray(points)
        if (self.pts[0] != self.pts[-1]).any():
            #close polygon
            self.pts = np.append(self.pts, [self.pts[0]], axis=0)

        self._A = None
        self._Iy = None
        self._Iz = None
        self._J = None

    
    @property
    def A(self):
        """
            Area of crosssection
        """
        if self._A is None:
            self._A = self._compute_area()

        return self._A
    
    @property
    def centroid(self):
        """
            Centroid coordinates
        """
        return self._compute_centroid()
    
    @property
    def Iy(self):
        """
            Moment of Inertia about the Y axis 
        """
        if self._Iy is None:
            self._Iy, self._Iz, self._J = self._compute_inertia()

        return self._Iy

    @property
    def Iz(self):
        """
            Moment of Inertia about the Z axis
        """
        if self._Iz is None:
            self._Iy, self._Iz, self._J = self._compute_inertia()

        return self._Iz
    
    @property
    def J(self):
        """
            Polar Moment of Inertia or Torsinal constant 
        """
        if self._J is None:
            self._Iy, self._Iz, self._J = self._compute_inertia()

        return self._J
    
    def _compute_area(self):
        """
            determines the area enclosed by the points 
        """
        y, z = self.pts[:,0], self.pts[:,1]
        a = 0.5 * np.sum(y[:-1]*z[1:] - y[1:]*z[:-1])

        return abs(a)
    
    def _compute_centroid(self):
        """
            returns the centroid coordinates of the given crosssection 
        """
        y, z = self.pts[:,0], self.pts[:,1]
        a = 3 * np.sum(y[:-1]*z[1:] - y[1:]*z[:-1])

        c = y[:-1] * z[1:] - y[1:] * z[:-1]
        cy = (y[:-1] + y[1:]) * c
        cy = np.sum(cy) / a

        cz = (z[:-1] + z[1:]) * c
        cz = np.sum(cz) / a

        return np.array([cy, cz])
    
    def _compute_inertia(self):
        pts_shifted = self.pts - self.centroid 
        y, z = pts_shifted[:,0], pts_shifted[:,1]

        c = y[:-1] * z[1:] - y[1:] * z[:-1]
        Iy = c * (z[:-1]**2 + z[:-1]*z[1:] + z[1:]**2)
        Iy = np.sum(Iy) / 12

        Iz = c * (y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)
        Iz = np.sum(Iz) / 12

        return abs(Iy), abs(Iz), abs(Iy) + abs(Iz)
    
    
    def plot(self):
        """
        Plots the cross section defined by the input boundary points.
        """
        fig = plt.figure()
        axes = fig.add_subplot(111, xlabel='Y', ylabel='Z', aspect='equal')

        # Plot boundary
        axes.plot(self.pts[:,0], self.pts[:,1], 'b-')

        # Plot centroid
        origin = self.centroid
        axes.plot(origin[0], origin[1], 'go')

        plt.show()