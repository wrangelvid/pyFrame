from numpy import array
class Node(object):
    def __init__(self,Name,x,y,z):
        self.Name = Name
        self.ID = None
        self.x = x 
        self.y = y
        self.z = z
        
        #global displacement relation
        self.Ux = None
        self.Uy = None
        self.Uz = None

        #global rotations relations
        self.Rx = None
        self.Ry = None
        self.Rz = None

        #define which displacement/rotation relations are immutable due to support condition
        self.support = [False for i in range(6)] 
        
    def pos(self):
        return array([self.x, self.y, self.z])
    
    
