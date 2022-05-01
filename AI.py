
from __future__ import print_function
import os
import neat
import visualize
import Tetris as T
import numpy as np
import pygame

def Run(config_file):
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
    winner = p.run(evaluation_genome, 1000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-2:'Gauche',
                  -1: 'Droite',
                  0:'Rotation',
                  1 :'Position_x',
                  2:'Postion_y',
                  3:'Position_alpha'}
    #visualize.draw_net(config, winner, True, node_names=node_names)
    #visualize.plot_stats(stats, ylog=False, view=True)
    #visualize.plot_species(stats, view=True)
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(evaluation_genome, 1000)


def plus_haut_bloc(grille):
    ind = len(grille)
    for j in range(len(grille)):
        row = grille[j]
        for espace in row:
            if espace != (0,0,0):
                return ind
        ind -= 1
    return ind

def espace_perdu(grille):
    count = 0
    for j in range (plus_haut_bloc(grille)-1, -1, -1):
        row_inf = grille[j]
        row_sup = grille[j-1]
        for espace in row_inf:
            if espace == (0,0,0) and row_sup[row_inf.index(espace)] != (0,0,0):
                count += 1
    return count




def evaluation_genome(genomes, config):
    pygame.init()
    surface = pygame.display.set_mode((T.Largeur_fenetre, T.Hauteur_fenetre))
    positions_statiques = {}
    for pos in [[np.random.randint(0,10), np.random.randint(15,20)] for _ in range(40)]:
        p = (pos[0], pos[1])
        positions_statiques[p] = T.Tetriminos_couleur[0]

    grille = T.creation_grille(positions_statiques)
    T.dessine_fenetre(surface, grille)
    pygame.display.update()
    nets = []
    tetrimino_training = []
    g = []

    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        tetrimino_training.append(T.obtenir_forme())
        g.append(genome)

    Fin = False
    run = True
    while run and len(tetrimino_training) > 0 :

        for t in tetrimino_training:
            temps = pygame.time.Clock()
            temps_de_chute = 0
            temps_de_jeu = 0
            vitesse_chute = 0.22
            phb2 = 20
            ep2 = 0
            output = nets[tetrimino_training.index(t)].activate((t.x, t.y, t.rotation))
            temps_de_chute += temps.get_rawtime()
            temps_de_jeu += temps.get_rawtime()
            temps.tick()

            #if temps_de_jeu / 1000 > 5:
            #    temps_de_jeu = 0
            #    if vitesse_chute > 0.12:
            #        vitesse_chute -= 0.005

            #for event in pygame.event.get():
            #    if event.type == pygame.QUIT:
            #        T.run = False


            if temps_de_chute > 0:
                temps_de_chute = 0
                t.y += 1
                g[tetrimino_training.index(t)].fitness += 0.1
                if not (T.espace_disponible(t, grille)) and t.y > 0:
                    t.y -= 1
                    Fin = True

            if output[0] > 0.5:
                t.x -= 1
                g[tetrimino_training.index(t)].fitness += 0.1
                if not (T.espace_disponible(t, grille)):
                    t.x += 1
                    g[tetrimino_training.index(t)].fitness -= 0.1
            if output[1] > -1:
                t.x += 1
                g[tetrimino_training.index(t)].fitness += 0.1
                if not (T.espace_disponible(t, grille)):
                    t.x -= 1
                    g[tetrimino_training.index(t)].fitness -= 0.1
            if output[2] > 0.5:
                t.rotation += 1
                g[tetrimino_training.index(t)].fitness += 0.1
                if not (T.espace_disponible(t, grille)):
                    t.rotation -= 1
                    g[tetrimino_training.index(t)].fitness -= 0.1

            if output[3] > -1:
                t.y += 1
                g[tetrimino_training.index(t)].fitness += 10
                if not (T.espace_disponible(t,grille)) and t.y > 0:
                    t.y -= 1
                    Fin = True


            tetrimino_pos = T.conversion_format(t)

            for j in range(len(tetrimino_pos)):
                x, y = tetrimino_pos[j]
                if y > -1:
                    grille[y][x] = t.couleur

            if Fin == True:
                if t.y >= 10:
                    g[tetrimino_training.index(t)].fitness += 50
                phb,phb2 = phb2,plus_haut_bloc(grille)
                ep,ep2 = ep2,espace_perdu(grille)
                for pos in tetrimino_pos:
                    p = (pos[0], pos[1])
                    positions_statiques[p] = t.couleur
                Fin = False
                g[tetrimino_training.index(t)].fitness += T.clear_rows(grille, positions_statiques) * 20
                if T.clear_rows(grille, positions_statiques) > 0:
                    print("Reussi")
                grille = T.creation_grille(positions_statiques)
                if phb2 > phb:
                    #g[tetrimino_training.index(t)].fitness -= 10
                    print("dmg")
                #if ep2 > ep:
                    #g[tetrimino_training.index(t)].fitness -= 20
                #    print("ducon")
                run = False
            T.dessine_fenetre(surface, grille)
            pygame.display.update()
        nets.pop(tetrimino_training.index(t))
        g.pop(tetrimino_training.index(t))
        tetrimino_training.pop(tetrimino_training.index(t))

        #if check_lost(positions_statiques):
        #   dessine_texte_milieu("PERDU", 80, (255, 255, 255), win)
        #    pygame.display.update()
        #    pygame.time.delay(1500)
        #    run = False

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    chemin_config = os.path.join(local_dir, 'Configuration_AI')
    Run(chemin_config)
