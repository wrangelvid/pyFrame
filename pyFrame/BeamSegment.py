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
        
        return self.theta1 - (self.M1*x - self.S1*x**2/2)/(self.EI)
    
    def Deflection(self, x = None):
        """
            Returns the deflection at a location on the segment
        """

        if x is None:
            x = self.L
       
        return self.delta1 + self.theta1*x - self.M1*x**2/(2*self.EI) + self.S1*x**3/(6*self.EI) 




    def AxialDeflection(self, x = None):
        """
            Returns the axial deflection at a location on the segment
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
