import pygame, pymunk, pickle, neat
from ragdoll import create_full_human_realistic
from terrain import create_parkour

FPS = 60
GEN_TIME = 10 * FPS
END_X = 1250

def draw(screen, parts, terrain, font, fitness, timeleft):
    screen.fill((240, 240, 255))
    for seg in terrain:
        pygame.draw.line(screen, (50,50,50), seg.a, seg.b, 5)
    for name, (b, s) in parts.items():
        pos = int(b.position.x), int(b.position.y)
        if isinstance(s, pymunk.Circle):
            pygame.draw.circle(screen, (180,0,0), pos, int(s.radius))
        else:
            verts = [v.rotated(b.angle) + b.position for v in s.get_vertices()]
            pygame.draw.polygon(screen, (0,0,180), verts)
    info = font.render(f"Fit: {fitness:.1f}  Time: {timeleft/FPS:.1f}s", True, (0,0,0))
    screen.blit(info, (20,20))

def main():
    with open("winner.pkl", "rb") as f:
        genome = pickle.load(f)

    config_path = "config-feedforward.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    pygame.init()
    screen = pygame.display.set_mode((1300, 700))
    font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    space = pymunk.Space(); space.gravity = (0,900)
    terrain = create_parkour(space)
    parts = create_full_human_realistic(space, 50, 440)

    torso = parts["torso"][0]
    left_leg = parts["left_leg"][0]
    right_leg = parts["right_leg"][0]
    start_x = torso.position.x
    fitness = 0

    for t in range(GEN_TIME):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); quit()

        vx, vy = torso.velocity
        lx, ly = left_leg.velocity
        rx, ry = right_leg.velocity
        inputs = [
            vx/100, vy/100,
            lx/100, ly/100,
            rx/100, ry/100,
            torso.position.x/800,
            torso.position.y/600,
        ]
        out = net.activate(inputs)

        left_leg.apply_force_at_local_point((out[0]*2000, 0), (0,0))
        right_leg.apply_force_at_local_point((out[1]*2000, 0), (0,0))

        space.step(1 / FPS)

        draw(screen, parts, terrain, font, fitness, GEN_TIME - t)
        pygame.display.flip()
        clock.tick(FPS)

        fitness = torso.position.x - start_x
        if torso.position.x > END_X:
            break

    print("Final fitness (replay):", round(fitness, 2))
    pygame.time.wait(2000)

if __name__ == "__main__":
    main()
