from numpy import array

class NodalForce(object):
    def __init__(self, Node, Name = None, Fx = 0.0, Fy = 0.0, Fz = 0.0):
        self.Node = Node
        self.Name = Name

        #global force at moment
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz 
    
    def vector(self):
        return array([[self.Fx,self.Fy,self.Fz]]).T

class NodalMoment(object):
    def __init__(self, Node, Name = None, Mx = 0.0, My = 0.0, Mz = 0.0):
        self.Node = Node
        self.Name = Name

        #global nodal moment
        self.Mx = Mx
        self.My = My
        self.Mz = Mz

    def vector(self):
        return array([[self.Mx,self.My,self.Mz]]).T



