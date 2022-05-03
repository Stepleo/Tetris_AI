from __future__ import print_function
import os
import neat
import visualize
import Tetris as T
import numpy as np
import pygame
import multiprocessing
import time

Core = 0




def Run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    #p = neat.checkpoint.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    corecount = 1
    if Core == 0:
        corecount = multiprocessing.cpu_count()
    else:
        corecount = Core
    pe = neat.ParallelEvaluator(corecount, evaluation_genome)
    # Run for up to 100 generations.
    winner = p.run(pe.evaluate, 1000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # node_names = {-2:'Gauche',
    #-1: 'Droite',
    #0: 'Rotation',
    #1: 'Position_x',
    #2: 'Postion_y',
    #3: 'Position_alpha'}
    # visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(evaluation_genome, 1000)


def plus_haut_bloc(grille):
    ind = len(grille)
    for j in range(len(grille)):
        row = grille[j]
        for espace in row:
            if espace != (0, 0, 0):
                return ind
        ind -= 1
    return ind


def trous_ajoutees(grille):
    count = 0
    for j in range(19,plus_haut_bloc(grille) - 1, -1):
        row = grille[j]
        for espace in row:
            if espace == (0, 0, 0) :
                count += 1
    return count

def verticalite(grille):
    c = 0
    for j in range (9):
        a,b = 0,0
        column1 = [grille[i][j] for i in range(20)]
        column2 = [grille[i][j+1] for i in range(20)]
        for espace in column1:
            if espace != (0,0,0):
                a += 1
        for espace in column2:
            if espace != (0,0,0):
                b += 1
        c += abs(b-a)
    return c

def colonnes_vide(grille):
    c = 0
    for j in range(10):
        a = 0
        for i in [grille[k][j] for k in range(20)]:
            if i != (0,0,0):
                a += 1
        if a != 0:
            c += 1
    return c

def voisin(tetrimino_pos,grille):
    c = 0
    position_valide = [[(j, i) for j in range(10) if grille[i][j] == (0, 0, 0)] for i in range(20)]
    position_valide = [j for sub in position_valide for j in sub]
    for j in range(len(tetrimino_pos)):
        x,y = tetrimino_pos[j]
        V = [(x-1,y-1),(x,y-1),(x+1,y-1),(x+1,y),(x+1,y+1),(x,y+1),(x-1,y+1),(x-1,y)]
        for i in V:
            if i in tetrimino_pos:
                continue
            if i not in position_valide:
                if i[1] > -1:
                    c+=1
        return c



def evaluation_genome(genome, config):
    surface = pygame.display.set_mode((T.Largeur_fenetre, T.Hauteur_fenetre))
    positions_statiques = {}
    grille = T.creation_grille(positions_statiques)
    T.dessine_fenetre(surface, grille)
    changement_tetrimino = False
    run = True
    tetrimino_actuel = T.obtenir_forme()
    prochain_tetrimino = T.obtenir_forme()
    temps = pygame.time.Clock()
    temps_de_chute = 0
    temps_de_jeu = 0
    temps_debut = time.time()
    vitesse_chute = 0.22
    score = 0
    pygame.init()


    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0

    phb2 = 20
    ta2 = 0
    nb_rotation = 0
    while run:
        grille = T.creation_grille(positions_statiques)
        temps_de_chute += temps.get_rawtime()
        temps_de_jeu += temps.get_rawtime()
        temps.tick(120)


        output = net.activate((tetrimino_actuel.x, tetrimino_actuel.y, tetrimino_actuel.rotation))




        # for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        T.run = False

        if temps_de_chute / 1000 > vitesse_chute:
            temps_de_chute = 0
            tetrimino_actuel.y += 1
            if not (T.espace_disponible(tetrimino_actuel, grille)) and tetrimino_actuel.y > 0:
                tetrimino_actuel.y -= 1
                changement_tetrimino = True

        if temps_de_jeu / 1000 > 5:
           temps_de_jeu = 0
           if vitesse_chute > 0.12:
               vitesse_chute -= 0.005

        if output[0] > 0.5:
            tetrimino_actuel.x -= 1
            if not (T.espace_disponible(tetrimino_actuel, grille)):
                tetrimino_actuel.x += 1
                fitness -= 1

        if output[1] > 0.5:
            tetrimino_actuel.x += 1
            if not (T.espace_disponible(tetrimino_actuel, grille)):
                tetrimino_actuel.x -= 1
                fitness -= 1

        if output[2] > 0.5:
            if nb_rotation >= 20:
                fitness -= 10
            else:
                tetrimino_actuel.rotation += 1
                nb_rotation += 1
                if not (T.espace_disponible(tetrimino_actuel, grille)):
                    tetrimino_actuel.rotation -= 1
                    fitness -= 1

        if output[3] > 0.1:
            tetrimino_actuel.y += 1
            fitness += 0.1
            if not (T.espace_disponible(tetrimino_actuel, grille)) and tetrimino_actuel.y > 0:
                tetrimino_actuel.y -= 1
                changement_tetrimino = True

        tetrimino_pos = T.conversion_format(tetrimino_actuel)

        if voisin(tetrimino_pos,grille) >= 5:
            fitness += 5
        else:
            fitness -= 1

        for j in range(len(tetrimino_pos)):
            x, y = tetrimino_pos[j]
            if y > -1:
                grille[y][x] = tetrimino_actuel.couleur

        if changement_tetrimino == True:
            phb, phb2 = phb2, plus_haut_bloc(grille)
            ta, ta2 = ta2, trous_ajoutees(grille)
            nb_rotation = 0
            for pos in tetrimino_pos:
                p = (pos[0], pos[1])
                positions_statiques[p] = tetrimino_actuel.couleur
            tetrimino_actuel = prochain_tetrimino
            prochain_tetrimino = T.obtenir_forme()
            changement_tetrimino = False

            fitness -= max(0,ta2 - ta)*5

            fitness -= max(0,phb2 - phb)*20

            fitness -= verticalite(grille)*10

            fitness -= colonnes_vide(grille)*10

            score += T.clear_rows(grille, positions_statiques)
            if T.clear_rows(grille, positions_statiques) > 0:
                print("Reussi")
            fitness += score * 1000

        if T.check_lost(positions_statiques):
            fitness += (time.time() - temps_debut)*100
            for j in grille:
                if j ==(0,0,0):
                    fitness -= 10
                else:
                    fitness += 20
            run = False

        T.dessine_fenetre(surface, grille)
        T.dessine_prochaine_forme(prochain_tetrimino,surface)
        pygame.display.update()
    return fitness


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    chemin_config = os.path.join(local_dir, 'Configuration_AI')
    Run(chemin_config)

