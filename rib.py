import numpy as np
from hjelpefunksjonar import norm, sortClockwise

class Rib:
    def __init__(self, coords, mu=0.5):
        self.vertices = sortClockwise(np.array(coords))
        
        self.normals = [norm(np.cross(self.vertices[1]-self.vertices[0],np.array([0,0,-1]))[:2]),
                        norm(np.cross(self.vertices[2]-self.vertices[1],np.array([0,0,-1]))[:2]), 
                        norm(np.cross(self.vertices[3]-self.vertices[2], np.array([0,0,-1]))[:2]),
                        norm(np.cross(self.vertices[0]-self.vertices[3],np.array([0,0,-1]))[:2]) ]
        
                        # Må sjekka om punkta skal gå mot eller med klokka. 
                        # Nett no går dei MED klokka. Normals skal peika UT.
        
        self.mu = mu
    def get_rib_middle(self):
        return np.sum(self.vertices, axis=0)/len(self.vertices)

    # def __init__(self, origin, width, height):
    #     # Bør kanskje ha informasjon om elastisiteten ved kollisjonar òg?
    #     self.origin = np.array(origin)
    #     self.width = width
    #     self.height = height
