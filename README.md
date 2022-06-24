# shard
## A 2D sequential impulse physics engine
### Features:
- SAT collision detection with contact manifolds for generic convex polygons
- Jacobian-based sequential impulse for collisions
- Symplectic Euler integration update
- Motors and Distance constraints for pendulums
### Getting Started:

```python
from shard.engine import ConstraintScene
from shard.gameobject import GameObject
from shard.rigidbody import Rigidbody
from shard.collider import Collider

scene = ConstraintScene()

c1 = Collider.regular_polygon(6, 150)
c2 = Collider.regular_polygon(5, 80)

ground_collider = Collider(np.array([[0, 0], [1200, 0], [1200, 100], [0, 100]]))
wall_collider = Collider(np.array([[0, 0], [50, 0], [50, 700], [0, 700]]))

# A paddle motor
r1 = Rigidbody(0., 0, np.array([640., 340.]))
g1 = GameObject(c1, r1)
paddle1 = GameObject(Collider(np.array([[0, 0], [20, 0], [20, 100], [0, 100]]), offset=np.array((0, 150))), r1)
r1.w = 1

ground = GameObject(ground_collider, Rigidbody(0, 0, np.array([700., 40.])))
left_wall = GameObject(wall_collider, Rigidbody(0, 0, np.array([50., 340.])))
right_wall = GameObject(wall_collider, Rigidbody(0, 0, np.array([1350., 340.])))

for i in range(5):
    scene.add_game_object(GameObject(c2, Rigidbody(1, 2e4, np.array([440., 180. + i * 150.]))))
    scene.add_game_object(GameObject(c2, Rigidbody(1, 2e4, np.array([640., 180. + i * 150.]))))
    scene.add_game_object(GameObject(c2, Rigidbody(1, 2e4, np.array([840., 180. + i * 150.]))))

scene.add_game_objects((ground, left_wall, right_wall))

scene.run(60)
```
### Demos

![](https://github.com/nkarve/shard/blob/main/demos/gif1.gif)

![](https://github.com/nkarve/shard/blob/main/demos/gif2.gif)

![](https://github.com/nkarve/shard/blob/main/demos/gif3.gif)

![](https://github.com/nkarve/shard/blob/main/demos/gif4.gif)

### Guide

See the [*How to Make a Physics Engine*](https://nkarve.github.io/programming/2021/06/29/physeng1.html) series on my website for an in-depth explanation of the physics, mathematics and programming behind this library.
