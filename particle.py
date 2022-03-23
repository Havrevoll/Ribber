from math import pi
from constants import ρ_p

class Particle:
    #Lag ein tabell med tidspunkt og posisjon for kvar einskild partikkel.
    def __init__(self, diameter, init_position, init_time=0, density=ρ_p ):
        self.diameter= diameter # mm
        self.init_position = init_position # mm
        self.init_time = init_time # s
        self.density = density # kg/mm³
        self.volume = self.diameter**3 * pi * 1/6 #  mm³
        self.mass = self.volume * self.density #  kg
        self.radius = self.diameter/2 # mm
        self.index = 0
        self.atol = 1e-6
        self.rtol = 1e-3
        self.method = 'RK45'
        self.linear = True
        self.lift = True
        self.addedmass = True
        self.resting = False
        self.still = False
        self.wrap_counter = 0
        self.wrap_max = 50
        self.resting_tolerance = 0.01


def particle_copy(pa):
    return Particle(pa.diameter, pa.init_position, pa.density)

