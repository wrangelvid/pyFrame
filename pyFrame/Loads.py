from numpy import array

class NodalForce(object):
    def __init__(self, Node, Name = None, Fx = 0.0, Fy = 0.0, Fz = 0.0):
        self.Node = Node
        self.Name = Name

        #global force at node
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

class MemberPtForce(object):
    def __init__(self, Member, x, Name = None, Fx = 0.0, Fy = 0.0, Fz = 0.0):
        """
            Point Force applied to member
            Forces are in members local coordinates 
            x must be between 0 and maximum member length
        """
        self.Member = Member
        self.Name = Name
        self.x = x 

        #local force at members position x  
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz 

    def vector(self):
        return array([[self.Fx,self.Fy,self.Fz]]).T

        
class MemberPtMoment(object):
    def __init__(self, Member, x, Name = None, Mx = 0.0, My = 0.0, Mz = 0.0):
        """
            Point Moment applied to member
            Moments are in members local coordinates 
            x must be between 0 and maximum member length
        """
        self.Member = Member
        self.Name = Name
        self.x = x 

        #local moment at members position x  
        self.Mx = Mx
        self.My = My
        self.Mz = Mz 
 


