import numpy as np
import itertools
import pyglet
from pyglet.gl import *
from copy import deepcopy
from constraints import *

from collider import Collider, orth
from rigidbody import Rigidbody
from gameobject import GameObject

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

class ConstraintScene():
    def __init__(self, gravity=-981):
        self.objects = []
        self.pts = []
        self.vecs = []
        self.constraints = []
        self.paused = False
        self.gravity = np.array([0., gravity])

    def add_game_object(self, obj):
        self.objects.append(obj)

        self.constraints = [PersistentContactConstraint(a, b) for a, b in itertools.combinations(self.objects, 2)]
    
    def add_game_objects(self, objs):
        self.objects.extend(objs)
        self.constraints = [PersistentContactConstraint(a, b) for a, b in itertools.combinations(self.objects, 2)]

    def run(self, frame_rate):
        snapshot = deepcopy(self.objects)

        config = Config(double_buffer=True, samples=4)
        window = pyglet.window.Window(config=config, resizable=True)
        fps_display = pyglet.window.FPSDisplay(window=window)

        self.offset = np.array([0., 0])

        @window.event
        def on_key_press(key, modifiers):
            if key == pyglet.window.key.E:                
                self.add_game_objects((GameObject(Collider.random_polygon(6, np.random.randint(
                    50, 101)), Rigidbody(1., 2e4, np.array([650., 800 + i * 300]), np.pi / 3)) for i in range(8)))
                print(len(self.constraints))
            elif key == pyglet.window.key.R:
                self.offset = np.zeros(2)
                self.objects = deepcopy(snapshot)
            elif key == pyglet.window.key.P:
                self.paused = not self.paused

        @window.event
        def on_draw():
            window.clear()
            fps_display.draw()
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glEnable(GL_BLEND)
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
            glLineWidth(2)
            glPointSize(3)

            batch = pyglet.graphics.Batch()
            for obj in self.objects:
                glColor4f(1, 0, 0, 1)
                v = obj.glVerts(self.offset)
                batch.add(len(v) // 2, GL_LINES, None, ('v2f', v))
            batch.draw()

            glColor4f(1, 1, 1, 1)
            batch = pyglet.graphics.Batch()
            for p in self.pts:
                batch.add(1, GL_POINTS, None, ('v2f', (p[0], p[1])))
            batch.draw()
            
            glColor4f(0, 1, 0, 1)
            for v in self.vecs:
                pyglet.graphics.draw(2, GL_LINES, ('v2f', (v[0][0], v[0][1], v[1][0], v[1][1])))

        self.dt = 1. / frame_rate
        pyglet.clock.schedule_interval(self.update, self.dt)
        pyglet.app.run()
    
    def solve_constraint(self, constraint, dt):
        a, b, M, Js, qs = constraint.a, constraint.b, constraint.M, constraint.J, constraint.q
        V = np.hstack([a.rb.v, a.rb.w, b.rb.v, b.rb.w])
        dV = np.zeros(6)

        for J, q in Js:            
            A = J @ M @ J.T 
            B = -q - J @ V

            l = B / A
            
            # This is for future updates, i.e. for clamping
            l_acc_old = constraint.l_acc
            constraint.l_acc = constraint.l_acc + l
            if constraint.l_acc > 0: constraint.l_acc = 0.
            l = constraint.l_acc - l_acc_old
            dV += M @ J.T * l

        dV[np.fabs(dV) < 1e-6] = 0

        a.rb.v += dV[0:2]
        a.rb.w += dV[2]
        b.rb.v += dV[3:5]
        b.rb.w += dV[5] 


    def update(self, dt):
        # Delete objects below the ground
        self.objects[1:] = [x for x in self.objects[1:] if x.rb.p[1] > 0]
        if self.paused: return;

        for obj in self.objects:
            obj.rb.f += np.array([0, -981 / obj.rb.inv_m if obj.rb.inv_m != 0.0 else 0.0])     

        for constraint in self.constraints:
            if constraint.a in self.objects and constraint.b in self.objects:
                if constraint.a.aabb().intersects(constraint.b.aabb()) and constraint.a.rb.inv_m + constraint.b.rb.inv_m > 1e-6:
                    constraint.set_manifolds(constraint.a.sat(constraint.b))
                    constraint.cache_jacobian(self)
        
        for i in range(8):
            for constraint in self.constraints:
                if constraint.manifolds:
                    self.solve_constraint(constraint, dt)
        
        for constraint in self.constraints: 
            constraint.reset()
        for obj in self.objects: obj.rb.update(dt)
