from math import pi

class Particle:
    #Lag ein tabell med tidspunkt og posisjon for kvar einskild partikkel.
    def __init__(self, diameter, init_position, init_time=0, density=2.65e-6 ):
        self.diameter= diameter
        self.init_position = init_position
        self.init_time = init_time
        self.density = density
        self.volume = self.diameter**3 * pi * 1/6
        self.mass = self.volume * self.density
        self.radius = self.diameter/2
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

