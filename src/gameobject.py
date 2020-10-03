from collider import *
from rigidbody import *
import itertools
import numpy as np

class ContactManifold():
    def __init__(self, mpv, ca, cb):
        self.mpv, self.ca, self.cb = mpv, ca, cb


class GameObject():
    def __init__(self, collider, rigidbody):
        self.col = collider
        self.rb = rigidbody
        self.cache_verts()
        self.edges = np.zeros_like(self.verts)
        self.active = True

    def cache_verts(self):
        self.verts = self.rb.l2g(self.col.verts)
    
    def get_edges(self):
        self.edges[:-1] = self.verts[1:] - self.verts[:-1]
        self.edges[-1] = self.verts[0] - self.verts[-1]
        return self.edges

    def glVerts(self, off=np.array([0, 0.])):
        self.cache_verts()
        v = self.verts - off
        v = tuple(np.repeat(v, 2, axis=0).flatten())
        return v[2:] + v[0:2]

    def aabb(self):
        v = self.verts
        return AABB(v.min(axis=0), v.max(axis=0))
    
    def support(self, dir_):
        v = self.verts
        i = np.argmax(v.dot(dir_))
        return v[i], i
    
    def vsupport(self, dir_, rev=False):
        v = self.verts
        if rev == True: v = v[::-1]
        return v[np.argmax(v.dot(dir_))]

    def sat(self, other):
        eps = 1e-4
        v1, v2, e1, e2 = self.verts, other.verts, self.get_edges(), other.get_edges()

        orths = np.vstack((orth(e1), orth(e2)))

        p1 = v1 @ orths.T
        p2 = v2 @ orths.T
        min1, max1 = p1.min(axis=0), p1.max(axis=0)
        min2, max2 = p2.min(axis=0), p2.max(axis=0)

        d = np.array([max1 - min2, max2 - min1]).min(axis=0)
        if (d >= -eps).all():
            p = orths * (d / (orths * orths).sum(axis=1))[:, None]
        else:
            return []

        mpv = min(p, key=lambda v: np.inner(v, v))
        d = self.rb.p - other.rb.p
        if np.inner(mpv, d) > -eps:
            mpv = -mpv
        
        
        class EdgeData():
            def __init__(self, furthest, from_, to_):
                self.furthest = furthest
                self.from_, self.to_ = from_, to_
                self.edge = self.to_ - self.from_

        def best_edge(g, v, e, dir_):
            furthest, index = g.support(dir_)
            cand1, cand2 = -e[index], e[(index - 1) % len(e)]
            cand1, cand2 = cand1 / np.linalg.norm(cand1), cand2 / np.linalg.norm(cand2)

            if cand1.dot(dir_) <= cand2.dot(dir_):
                return EdgeData(furthest, v[index], v[(index + 1) % len(v)])
            else:
                return EdgeData(furthest, v[(index - 1) % len(v)], v[index])
        
        
        def clip(v1, v2, n, target):
            contacts = []
            d1, d2 = v1.dot(n) - target, v2.dot(n) - target
            
            if d1 >= -eps: contacts.append(v1)
            if d2 >= -eps: contacts.append(v2)
            if d1 * d2 <= eps: contacts.append(v1 + d1 / (d1 - d2) * (v2 - v1))

            return contacts

        edge_data1 = best_edge(self, v1, e1, mpv)
        edge_data2 = best_edge(other, v2, e2, -mpv)

        if np.fabs(edge_data1.edge.dot(mpv)) <= np.fabs(edge_data2.edge.dot(mpv)):
            ref, inc, flipped = edge_data1, edge_data2, False
        else:
            ref, inc, flipped = edge_data2, edge_data1, True

        refv = ref.edge
        refv = refv / np.linalg.norm(refv)
        
        o1 = refv.dot(ref.from_)
        cp = clip(inc.from_, inc.to_, refv, o1)

        if len(cp) < 2: return []

        o2 = refv.dot(ref.to_)
        cp = clip(cp[0], cp[1], -refv, -o2)
        
        if len(cp) < 2: return []
        
        refnorm = orth(refv)
        if flipped: 
            refnorm = -refnorm
            mpv = -mpv
        max_ = refnorm.dot(ref.furthest)

        d0 = cp[0].dot(-mpv)
        d1 = cp[1].dot(-mpv)
        diff = d1 - d0

        eps = 10e-2
        if diff > eps:
            del cp[0]
        elif diff < -eps:
            del cp[1]
        
        if flipped:
            if len(cp) > 1:
                return [ContactManifold(mpv, cp[0], cp[0] + mpv), ContactManifold(mpv, cp[1], cp[1] + mpv)]
            else:
                return [ContactManifold(mpv, cp[0], cp[0] + mpv)]
        else:
            if len(cp) > 1:
                return [ContactManifold(-mpv, cp[0] + mpv, cp[0]), ContactManifold(-mpv, cp[1] + mpv, cp[1])]
            else:
                return [ContactManifold(-mpv, cp[0] + mpv, cp[0])]

