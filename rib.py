import numpy as np
from hjelpefunksjonar import norm, sortClockwise
from constants import µ
from scipy.spatial import KDTree

class Rib:
    def __init__(self, coords, µ=µ):
        self.tre = KDTree(coords)
        self.array = coords


        # self.vertices = sortClockwise(np.array(coords))
        
        # self.normals = [norm(np.cross(self.vertices[1]-self.vertices[0],np.array([0,0,-1]))[:2,None]),
        #                 norm(np.cross(self.vertices[2]-self.vertices[1],np.array([0,0,-1]))[:2,None]), 
        #                 norm(np.cross(self.vertices[3]-self.vertices[2], np.array([0,0,-1]))[:2,None]),
        #                 norm(np.cross(self.vertices[0]-self.vertices[3],np.array([0,0,-1]))[:2,None]) ]
        
                        # Må sjekka om punkta skal gå mot eller med klokka. 
                        # Nett no går dei MED klokka. Normals skal peika UT.
        
        self.µ = µ

    # def dist(self, data):
    #     distances, numbers = self.tre(data,2)


    def normal(self, data):
        distances, numbers = self.tre.query(data[:2],2)
        if abs(numbers[1] - numbers[0]) > 5:
            numbers[1] = numbers[0]+1
        # numbers = np.sort(numbers)
        # comp_normal = norm(np.cross(self.array[sorted(numbers)[1]]-self.array[sorted(numbers)[0]],np.array([0,0,1]))[:2,None])
        comp_normal = norm(np.cross(self.array[sorted(numbers)[1]]-self.array[sorted(numbers)[0]],np.array([0,0,1]))[:2,None])
        return comp_normal,numbers[0]
    # # def get_rib_middle(self):
    # #     return np.sum(self.vertices, axis=0)/len(self.vertices)

    # # def get_rib_dimensions(self):
    # #     lengder = np.sqrt(np.square(np.roll(self.vertices,shift=-1,axis=0) - self.vertices).sum( axis=1 ))
    # #     return np.min(lengder), np.max(lengder)

    #     shortest = np.inf
    #     longest = 0
    #     for i,v  in enumerate(self.vertices):
    #         lengd = ((self.vertices[i]-self.vertices[i-1])**2).sum()**0.5
    #         shortest = min(lengd, shortest)
    #         longest = max(lengd, longest)

    #     return shortest,longest
    # def __init__(self, origin, width, height):
    #     # Bør kanskje ha informasjon om elastisiteten ved kollisjonar òg?
    #     self.origin = np.array(origin)
    #     self.width = width
    #     self.height = height
