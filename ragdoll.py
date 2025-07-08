import pymunk

def create_full_human_realistic(space, x, y):
    parts = {}

    def add_box(name, pos, size, mass=1):
        b = pymunk.Body(mass, pymunk.moment_for_box(mass, size))
        b.position = pos
        s = pymunk.Poly.create_box(b, size)
        s.friction = 1.0
        space.add(b, s)
        parts[name] = (b, s)
        return b

    def add_circle(name, pos, rad, mass=0.5):
        b = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, rad))
        b.position = pos
        s = pymunk.Circle(b, rad)
        s.friction = 1.0
        space.add(b, s)
        parts[name] = (b, s)
        return b

    torso = add_box("torso", (x, y), (20, 60))
    head = add_circle("head", (x, y - 45), 15)
    left_leg = add_box("left_leg", (x - 7, y + 45), (10, 45))
    right_leg = add_box("right_leg", (x + 7, y + 45), (10, 45))
    left_arm = add_box("left_arm", (x - 20, y - 10), (10, 35))
    right_arm = add_box("right_arm", (x + 20, y - 10), (10, 35))

    def connect(b1, b2, a1, a2, limits=(-0.5, 0.5)):
        space.add(
            pymunk.PinJoint(b1, b2, a1, a2),
            pymunk.PivotJoint(b1, b2, b1.position + a1),
            pymunk.RotaryLimitJoint(b1, b2, *limits)
        )

    connect(torso, head, (0, -30), (0, 15), (-0.5, 0.5))
    connect(torso, left_leg, (-5, 30), (0, -20), (-0.5, 1.0))
    connect(torso, right_leg, (5, 30), (0, -20), (-0.5, 1.0))
    connect(torso, left_arm, (-10, -20), (0, -20), (-1.0, 1.0))
    connect(torso, right_arm, (10, -20), (0, -20), (-1.0, 1.0))

    return parts
