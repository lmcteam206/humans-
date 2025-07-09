import pygame, pymunk, neat, os, pickle
from ragdoll import create_full_human_realistic
from terrain import create_parkour

FPS = 60
GEN_TIME = 15 * FPS
END_X = 1250
VISUALIZE = False

def draw_all(screen, parts, terrain, font, generation, fitness, timeleft):
    screen.fill((240,240,255))
    for seg in terrain:
        pygame.draw.line(screen, (50,50,50), seg.a, seg.b, 5)
    for name,(b,s) in parts.items():
        pos = int(b.position.x), int(b.position.y)
        color = (0,0,180) if "arm" in name or "leg" in name else (180,0,0)
        if isinstance(s, pymunk.Circle):
            pygame.draw.circle(screen, color, pos, int(s.radius))
        else:
            verts = [v.rotated(b.angle)+b.position for v in s.get_vertices()]
            pygame.draw.polygon(screen, color, verts)
    info = font.render(f"Gen:{generation} Fit:{fitness:.1f} T:{timeleft/FPS:.1f}s", True, (0,0,0))
    screen.blit(info, (20,20))

def eval_genome(genome, config, generation):
    space = pymunk.Space(); space.gravity=(0,900)
    terrain = create_parkour(space)
    parts = create_full_human_realistic(space, 50, 440)
    torso = parts["torso"][0]
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    start_x = torso.position.x; fitness = 0

    if VISUALIZE:
        pygame.init(); screen=pygame.display.set_mode((1300,700))
        font=pygame.font.SysFont("Arial",24); clock=pygame.time.Clock()

    for t in range(GEN_TIME):
        if VISUALIZE:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit(); quit()

        vx, vy = torso.velocity
        ll = parts["left_leg"][0]; rl = parts["right_leg"][0]
        inputs = [
            vx/100, vy/100,
            (torso.position.x-start_x)/800,
            torso.position.y/600,
            (ll.position.x-torso.position.x)/100,
            (ll.position.y-torso.position.y)/100,
            (rl.position.x-torso.position.x)/100,
            (rl.position.y-torso.position.y)/100
        ]
        out = net.activate(inputs)
        force_leg = 4000; force_arm = 2000
        parts["left_leg"][0].apply_force_at_local_point((out[0]*force_leg,0),(0,0))
        parts["right_leg"][0].apply_force_at_local_point((out[1]*force_leg,0),(0,0))
        parts["left_arm"][0].apply_force_at_local_point((out[2]*force_arm,0),(0,0))
        parts["right_arm"][0].apply_force_at_local_point((out[3]*force_arm,0),(0,0))

        space.step(1/FPS)
        fitness = max(fitness, torso.position.x - start_x)

        if VISUALIZE:
            draw_all(screen, parts, terrain, font, generation, fitness, GEN_TIME-t)
            pygame.display.flip(); clock.tick(FPS)

        if torso.position.x >= END_X:
            fitness = torso.position.x - start_x
            break

    genome.fitness = max(0.0, fitness)

def run():
    local = os.path.dirname(__file__)
    cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      os.path.join(local,"config-feedforward.txt"))
    pop = neat.Population(cfg)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    generation = 0

    def eval_all(genomes, config):
        nonlocal generation
        for gid, genome in genomes:
            eval_genome(genome, config, generation)
        print(f"--- Completed generation {generation}")
        generation += 1

    winner = pop.run(eval_all, 10000)

    with open("winner.pkl","wb") as f:
        pickle.dump(winner,f)
    print("Training complete. Winner saved as 'winner.pkl'")

if __name__=="__main__":
    run()
