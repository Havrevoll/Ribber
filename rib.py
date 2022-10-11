import numpy as np
from hjelpefunksjonar import norm, sortClockwise
from constants import µ

class Rib:
    def __init__(self, coords, µ=µ):
        self.vertices = sortClockwise(np.asarray(coords))
        
        self.normals = [norm(np.cross(self.vertices[1]-self.vertices[0],np.array([0,0,-1]))[:2,None]),
                        norm(np.cross(self.vertices[2]-self.vertices[1],np.array([0,0,-1]))[:2,None]), 
                        norm(np.cross(self.vertices[3]-self.vertices[2], np.array([0,0,-1]))[:2,None]),
                        norm(np.cross(self.vertices[0]-self.vertices[3],np.array([0,0,-1]))[:2,None]) ]
        
                        # Må sjekka om punkta skal gå mot eller med klokka. 
                        # Nett no går dei MED klokka. Normals skal peika UT.
        
        self.µ = µ
    def get_rib_middle(self):
        return np.sum(self.vertices, axis=0)/len(self.vertices)

    def get_rib_dimensions(self):
        lengder = np.sqrt(np.square(np.roll(self.vertices,shift=-1,axis=0) - self.vertices).sum( axis=1 ))
        return np.min(lengder), np.max(lengder)

        shortest = np.inf
        longest = 0
        for i,v  in enumerate(self.vertices):
            lengd = ((self.vertices[i]-self.vertices[i-1])**2).sum()**0.5
            shortest = min(lengd, shortest)
            longest = max(lengd, longest)

        return shortest,longest
    # def __init__(self, origin, width, height):
    #     # Bør kanskje ha informasjon om elastisiteten ved kollisjonar òg?
    #     self.origin = np.array(origin)
    #     self.width = width
    #     self.height = height
