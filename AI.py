
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
    winner = p.run(evaluation_genome, 100)

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
            ta2 = 0
            output = nets[tetrimino_training.index(t)].activate((t.x, t.y, t.rotation))
            temps_de_chute += temps.get_rawtime()
            temps_de_jeu += temps.get_rawtime()
            temps.tick()

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
            
            if voisin(tetrimino_pos,grille) >= 5: #on récompense l'IA si elle arrive à avoir beaucoup de voisins pour l'orienter vers la complétion de lignes.
                g[tetrimino_training.index(t)].fitness += 5
            else:
                g[tetrimino_training.index(t)].fitness -= 1

            for j in range(len(tetrimino_pos)):
                x, y = tetrimino_pos[j]
                if y > -1:
                    grille[y][x] = t.couleur

            if Fin:
                phb,phb2 = phb2,plus_haut_bloc(grille)
                ta,ta2 = ep2,trous_ajoutees(grille)
                for pos in tetrimino_pos:
                    p = (pos[0], pos[1])
                    positions_statiques[p] = t.couleur
                Fin = False
                g[tetrimino_training.index(t)].fitness += T.clear_rows(grille, positions_statiques) * 20
                grille = T.creation_grille(positions_statiques)
                g[tetrimino_training.index(t)].fitness -= max(0,ta2 - ta)*5 

                g[tetrimino_training.index(t)].fitness -= max(0,phb2 - phb)*20

                g[tetrimino_training.index(t)].fitness -= verticalite(grille)*10

                g[tetrimino_training.index(t)].fitness
                -= colonnes_vide(grille)*10
                
                run = False
                
                
            T.dessine_fenetre(surface, grille)
            pygame.display.update()
            
        nets.pop(tetrimino_training.index(t))
        g.pop(tetrimino_training.index(t))
        tetrimino_training.pop(tetrimino_training.index(t))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    chemin_config = os.path.join(local_dir, 'Configuration_AI')
    Run(chemin_config)
