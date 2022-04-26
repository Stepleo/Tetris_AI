
from __future__ import print_function
import os
import neat
import visualize

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    chemin_config = os.path.join(local_dir, 'Configuration_AI.txt')
    run(chemin_config)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 100 generations.
    winner = p.run(evaluation_genome, 10)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-2:'Gauche',
                  -1: 'Droite',
                  0:'Rotation',
                  1 :'Position_x',
                  2:'Postion_y',
                  3:'Position_alpha'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(evaluation_genome, 10)



def evaluation_genome(genomes, config):
    positions_statiques = {}
    grille = creation_grille(positions_statiques)

    changement_tetrimino = False
    run = True
    tetrimino_actuel = obtenir_forme()
    prochain_tetrimino = obtenir_forme()
    temps = pygame.time.Clock()
    temps_de_chute = 0
    temps_de_jeu = 0
    vitesse_chute = 0.33
    score = 0


    nets = []
    tetrimino_training = []
    g = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        tetrimino_training.append(obtenir_forme())
        g.append(genome)



    for i,t in enumerate(tetrimino_training):
        output = nets[i].activate(t.x, t.y, t.rotation )
        while run:
            temps_de_chute += temps.get_rawtime()
            temps_de_jeu += temps.get_rawtime()
            temps.tick()

            if temps_de_chute / 1000 > vitesse_chute:
                temps_de_chute = 0
                t.y += 1
                if not (espace_disponible(t, grille)) and t.y > 0:
                    t.y -= 1
                    changement_tetrimino = True


            if temps_de_jeu / 1000 > 5:
                temps_de_jeu = 0
                if vitesse_chute > 0.12:
                    vitesse_chute -= 0.005

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

                if output[i][0] > 0.5:
                    t.x -= 1
                    if not (espace_disponible(t, grille)):
                        t.x += 1
                if output[i][1] > 0.5:
                    t.x += 1
                    if not (espace_disponible(t, grille)):
                        t.x -= 1
                if output[i][2] > 0.5:
                    t.rotation += 1
                    if not (espace_disponible(t, grille)):
                        t.rotation -= 1



            tetrimino_pos = conversion_format(t)

            for j in range(len(tetrimino_pos)):
                x, y = tetrimino_pos[j]
                if y > -1:
                    grille[y][x] = t.couleur

            if changement_tetrimino:
                for pos in tetrimino_pos:
                    p = (pos[0], pos[1])
                    positions_statiques[p] = t.couleur
                changement_tetrimino = False
                g[i] += clear_rows(grille, positions_statiques) * 5
                run = False

        dessine_fenetre(win, grille, score)
        dessine_prochaine_forme(prochain_tetrimino, win)
        pygame.display.update()

        #if check_lost(positions_statiques):
        #   dessine_texte_milieu("PERDU", 80, (255, 255, 255), win)
        #    pygame.display.update()
        #    pygame.time.delay(1500)
        #    run = False