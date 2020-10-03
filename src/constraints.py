import numpy as np
from gameobject import GameObject
from collider import orth

class PersistentContactConstraint():
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.M = np.diag([a.rb.inv_m, a.rb.inv_m, a.rb.inv_i, b.rb.inv_m, b.rb.inv_m, b.rb.inv_i])
        self.manifolds = None
        self.J, self.A, self.q = None, None, None
        self.l_acc = 0.
    
    def _jacobian(self, manifold, dt):
        mpv, ca, cb = manifold.mpv, manifold.ca, manifold.cb
        dist = np.linalg.norm(mpv)
        n = mpv / dist
        ra, rb = ca - self.a.rb.p, cb - self.b.rb.p

        J = np.hstack((-n, np.cross(ra, n), n, -np.cross(rb, n)))
        s, sl = 0.5, 0.01
        q = s / dt * max(dist - sl, 0.)

        return J, q

    def set_manifolds(self, manifolds):
        # Separate method in case I have to do any manifold preprocessing, e.g. for warm starting
        self.manifolds = manifolds

    def cache_jacobian(self, scene):
        self.J = []
        for manifold in self.manifolds:
            self.J.append(self._jacobian(manifold, scene.dt))

    def reset(self):
        self.manifolds = None
        self.J, self.A, self.q = None, None, None

class NewDistanceConstraint():
    def __init__(self, a, b, max_dist, compressible=False):
        self.a, self.b = a, b
        self.M = np.diag([a.rb.inv_m, a.rb.inv_m, a.rb.inv_i, b.rb.inv_m, b.rb.inv_m, b.rb.inv_i])
        self.max_dist = max_dist
        self.compressible = compressible
    
    def cache_jacobian(self, scene):
        ap, bp = self.a.rb.p, self.b.rb.p # edit to generalise
        n = bp - ap
        d = np.linalg.norm(n)
        if self.compressible and d < self.max_dist:
            return
        
        n /= d
        rsa, rsb = orth(self.a.rb.p - ap), orth(self.b.rb.p - bp)
        
        J = np.hstack((-n, -rsa.dot(n), n, rsb.dot(n)))
        s = 0.2
        q = s * (np.fabs(d) - self.max_dist) / scene.dt

        self.J = J, q
    
    def reset(self):
        self.J = None

class MultiConstraint():
    def __init__(self, a, b, *constraints):
        self.a, self.b = a, b
        self.M = np.diag([a.rb.inv_m, a.rb.inv_m, a.rb.inv_i, b.rb.inv_m, b.rb.inv_m, b.rb.inv_i])
        self.constraints = constraints
    
    def jacobian(self, dt=1/60):
        J = np.zeros((len(self.constraints), 6))
        q = np.zeros((len(self.constraints), 2))
        
        for i, constraint in enumerate(self.constraints):
            J[i], q[i] = constraint.jacobian() 
        return J, q
