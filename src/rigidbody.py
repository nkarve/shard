import numpy as np

class Rigidbody():
    def __init__(self, mass, inertia, position, angle=0):
        self.inv_m = 0 if mass == 0 else 1. / mass
        self.inv_i = 0 if inertia == 0 else 1. / inertia

        self.p = position
        self.v = np.zeros(2)
        self.f = np.zeros(2)
        
        self.a = angle
        self.w = 0.
        self.t = 0.

        self.R = np.zeros((2, 2))
        self.updateR()

        self.awake = True
        self.count = 0
    
    def updateR(self):
        c, s = np.cos(self.a), np.sin(self.a)
        self.R[0, 0] = self.R[1, 1] = c
        self.R[1, 0] = s
        self.R[0, 1] = -s

    def add_force(self, force, at):
        self.f += force
        self.t += np.cross(at - self.p, force) 

    def add_impulse(self, impulse, at):
        self.v += self.inv_m * impulse
        self.w += self.inv_i * np.cross(at - self.p, impulse)

    def update(self, dt):       
        self.v += self.f * self.inv_m * dt
        self.p += self.v * dt
        self.f = np.zeros(2)
 
        self.w += self.t * self.inv_i * dt
        self.a += self.w * dt
        self.t = 0.

        self.updateR()
    
    def l2g(self, v, vec=False):
        if vec:
            return v @ self.R
        else:
            return v @ self.R + self.p
    
    def g2l(self, v, vec=False):
        return v @ self.R.T if vec else (v - self.p) @ self.R.T

    
