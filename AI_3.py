from __future__ import print_function
import os
import neat
import Tetris as T
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
    pe = neat.ParallelEvaluator(1, evaluation_genome)
    # Run for up to 100 generations.
    winner = p.run(pe.evaluate, 1000)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


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

def simulation(tetrimino,direction,orientation,grille):
    ta = trous_ajoutees(grille)
    phb = plus_haut_bloc(grille)
    cv = colonnes_vide(grille)
    v = verticalite(grille)
    tetrimino.rotation = orientation
    tetrimino.x = direction
    if not T.espace_disponible(tetrimino,grille):
        return False
    while T.espace_disponible(tetrimino,grille):
        tetrimino.y += 1
        if not T.espace_disponible(tetrimino,grille):
            tetrimino.y -= 1
    tetrimino_pos = T.conversion_format(tetrimino)
    nb_voisin = voisin(tetrimino_pos,grille)
    for j in range(len(tetrimino_pos)):
        x, y = tetrimino_pos[j]
        if y > -1:
            grille[y][x] = tetrimino_actuel.couleur
    nouveau_trous = trous_ajoutees(grille) - ta
    hauteur_ajoutee = plus_haut_bloc(grille) - phb
    remplissage_colonne = colonnes_vide(grille) - cv
    applanissement = verticalite(grille) - v
    return [nb_voisin,nouveau_trous,hauteur_ajoutee,remplissage_colonne,applanissement]



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
    vitesse_chute = 0.22
    pygame.init()


    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    c = 0
    while run:
        grille = T.creation_grille(positions_statiques)
        temps_de_chute += temps.get_rawtime()
        temps_de_jeu += temps.get_rawtime()
        temps.tick()

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


        if c == 0:
            c += 1
            direction = [i for i in range(10)]
            orientation = [k for k in range(4)]
            choix = {}
            for i in direction:
                for k in orientation:
                    simu = simulation(tetrimino_actuel,i,k,grille)
                    if simu != False:
                        choix[i,k] = simu

            for j in choix:
                etat = choix[j]
                score = net.activate(etat)
                choix[j] = score

            meilleur_placement = None
            meilleur_score = None
            for j in choix:
                if meilleur_score == None:
                    meilleur_placement = j
                    meilleur_score = choix[j]
                elif choix[j] > meilleur_score:
                    meilleur_placement = j
                    meilleur_score = choix[j]

            position_visee = meilleur_placement[0]
            rotation_visee = meilleur_placement[1]

        if tetrimino_actuel.rotation != rotation_visee:
            tetrimino_actuel.rotation += 1
        if tetrimino_actuel.x > position_visee:
            tetrimino_actuel.x -= 1
        if tetrimino_actuel.x < position_visee:
            tetrimino_actuel.x += 1

        tetrimino_pos = T.conversion_format(tetrimino_actuel)

        for j in range(len(tetrimino_pos)):
            x, y = tetrimino_pos[j]
            if y > -1:
                grille[y][x] = tetrimino_actuel.couleur

        if changement_tetrimino == True:
            for pos in tetrimino_pos:
                p = (pos[0], pos[1])
                positions_statiques[p] = tetrimino_actuel.couleur
            tetrimino_actuel = prochain_tetrimino
            prochain_tetrimino = T.obtenir_forme()
            changement_tetrimino = False

            points += T.clear_rows(grille, positions_statiques)
            fitness += points * 1000

        if T.check_lost(positions_statiques):
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
    chemin_config = os.path.join(local_dir, 'Configuration_AI_3')
    Run(chemin_config)

