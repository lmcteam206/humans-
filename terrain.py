import pymunk

def create_parkour(space):
    segs = []

    def add(p1, p2):
        s = pymunk.Segment(space.static_body, p1, p2, 5)
        s.friction = 1.0
        space.add(s); segs.append(s)

    add((0, 500), (300, 500))
    add((400, 450), (650, 450))
    add((700, 400), (1000, 400))
    add((1100, 500), (1300, 500))
    return segs
