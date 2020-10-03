import numpy as np

def orth(v):
    if v.ndim == 1:
        return np.array([-v[1], v[0]])
    
    o = np.zeros_like(v)
    o[:, 0], o[:, 1] = -v[:, 1], v[:, 0]
    return o

class AABB():
    def __init__(self, min_, max_):
        self.min, self.max = min_, max_
    
    def intersects(self, other):
        return ((self.min <= other.max) & (self.max >= other.min)).all()
    
    def glVerts(self):
        return (self.min[0], self.min[1], self.max[0], self.min[1],
             self.max[0], self.min[1], self.max[0], self.max[1],
             self.max[0], self.max[1], self.min[0], self.max[1],
             self.min[0], self.max[1], self.min[0], self.min[1])

class Collider():
    def __init__(self, verts, restitution=1.0, friction=.8, offset=np.zeros(2)):
        vverts = np.zeros((verts.shape[0] + 1, verts.shape[1]))
        vverts[:-1], vverts[-1] = verts, verts[0]
        
        self.area = 0
        self.centroid = np.zeros(2)

        for i in range(vverts.shape[0] - 1):
            a = vverts[i, 0] * vverts[i + 1, 1] - vverts[i + 1, 0] * vverts[i, 1]
            self.area += a
            self.centroid += a * (vverts[i] + vverts[i + 1])
        
        self.area *= 0.5
        self.centroid /= 6 * self.area
        self.verts = verts - self.centroid + offset
        self.res = restitution  
        self.mu = friction
    
    def edges(self):
        edges = np.zeros_like(self.verts)
        edges[:-1] = self.verts[1:] - self.verts[:-1]
        edges[-1] = self.verts[0] - self.verts[-1]
        return edges

    def aabb(self):
        return AABB(np.min(self.verts, axis=0), np.max(self.verts, axis=0))

    @staticmethod
    def regular_polygon(n, size):
        verts = np.exp(np.arange(n) * np.pi * 2j / n) * size
        return Collider(verts.view('(2,)float'))
    
    @staticmethod
    def random_polygon(n, size):
        angles = np.cumsum(np.random.rand(n))
        angles = angles / np.max(angles)
        verts = np.exp(angles * np.pi * 2j) * size
        return Collider(verts.view('(2,)float'))
    


