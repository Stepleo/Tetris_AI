from __future__ import print_function
import os
import neat
import Tetris as T
import pygame
import multiprocessing
import time

Core = 0

def Run(config_file):
    # On charge la configuration qu'utilisera notre intelligence artificielle pour sa reproduction.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # On crée notre population, qui est l'objet le plus important lorsque l'on utilise NEAT.
    p = neat.Population(config)
    # On crée un rapporteur d'écart-type pour montrer la progression de nos générations dans le terminal.
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
    # On choisit de l'exécuter pour 1000 générations.
    winner = p.run(pe.evaluate, 1000)

    # On affiche le meilleur génome.
    print('\nBest genome:\n{!s}'.format(winner))


# Les 5 fonctions suivantes servent à déterminer l'évolution de la partie à chaque action de l'IA et donc à la noter plus tard.
def plus_haut_bloc(grille):
    ind = len(grille)
    for j in range(len(grille)):
        row = grille[j]
        for espace in row:
            if espace != (0, 0, 0):
                return ind
        ind -= 1
    return ind


def trous_ajoutees(grille): #cette fonction sert à déterminer le nombre de trous qui ont été ajoutés, c'est-à-dire le nombre d'espaces inaccessibles car recouverts
    count = 0
    for j in range(19,plus_haut_bloc(grille) - 1, -1):
        row = grille[j]
        for espace in row:
            if espace == (0, 0, 0) :
                count += 1
    return count

def verticalite(grille):#cette fonction sert à déterminer la différence de hauteur entre deux colonnes adjacentes.
    c = 0
    for j in range (9):
        a,b = 0,0
        column1 = [grille[i][j] for i in range(20)] # On prend nos deux colonnes adjacentes et on compte le nombre de blocs qui y sont présents.
        column2 = [grille[i][j+1] for i in range(20)]
        for espace in column1:
            if espace != (0,0,0):
                a += 1
        for espace in column2:
            if espace != (0,0,0):
                b += 1
        c += abs(b-a)
    return c #on renvoie la valeur absolue de la somme des différences.

def colonnes_vide(grille): # Cette fonction porte bien son nom.
    c = 0
    for j in range(10):
        a = 0
        for i in [grille[k][j] for k in range(20)]:
            if i != (0,0,0):
                a += 1
        if a != 0:
            c += 1
    return c

def voisin(tetrimino_pos,grille):# Cette fonction renvoie le nombre de blocs adjacents au tétrimino.
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
    # On initialise dans un premier temps le jeu Tetris, qui a été importé comme T.
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

    #création de notre neurone à partir du genome
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

        #On récupère la réponse du neurone aux informations qu'on lui donne dans une liste dont la taille est définit dans Configuration_AI
        output = net.activate((tetrimino_actuel.x, tetrimino_actuel.y, tetrimino_actuel.rotation))


        if temps_de_chute / 1000 > vitesse_chute:
            temps_de_chute = 0
            tetrimino_actuel.y += 1
            if not (T.espace_disponible(tetrimino_actuel, grille)) and tetrimino_actuel.y > 0:
                tetrimino_actuel.y -= 1
                changement_tetrimino = True


        if output[0] > 0.5: #ce coeffcient de output est celui qui correspond à l'action d'aller à gauche donc si cette condition est remplie, l'IA essaye d'aller à gauche
            tetrimino_actuel.x -= 1
            if not (T.espace_disponible(tetrimino_actuel, grille)): #si le tetrimino ne peut pas aller à gauche, alors on annule le mouvement et pénalise l'IA
                tetrimino_actuel.x += 1
                fitness -= 1

        if output[1] > 0.5:#ce coefficient de output est celui qui correspond à l'action d'aller à droite
            tetrimino_actuel.x += 1
            if not (T.espace_disponible(tetrimino_actuel, grille)):
                tetrimino_actuel.x -= 1
                fitness -= 1

        if output[2] > 0.5: #ce coefficient de output est celui qui correspond à l'action de faire une rotation
            if nb_rotation >= 20: #on pénalise l'IA si elle esssaye de trop tourner pour éviter certains bugs
                fitness -= 10
            else:
                tetrimino_actuel.rotation += 1
                nb_rotation += 1
                if not (T.espace_disponible(tetrimino_actuel, grille)): #de la même façon que les deux boucles précédentes, l'IA est pénalisée en cas d'erreur
                    tetrimino_actuel.rotation -= 1
                    fitness -= 1

        if output[3] > 0.1: #ce coefficient de output est celui qui correspond à l'action de descendre
            tetrimino_actuel.y += 1
            fitness += 0.1 #on la récompense afin qu'elle aille au plus vite pour qu'elle n'essaye pas de gagner du temps..
            if not (T.espace_disponible(tetrimino_actuel, grille)) and tetrimino_actuel.y > 0:
                tetrimino_actuel.y -= 1
                changement_tetrimino = True

        tetrimino_pos = T.conversion_format(tetrimino_actuel)

        if voisin(tetrimino_pos,grille) >= 5: #on récompense l'IA si elle arrive à avoir beaucoup de voisins pour l'orienter vers la complétion de lignes.
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
            #on pénalise l'IA si elle a fait des trous supplémentaires pour qu'elle essaye de les combler et si elle a rajouté de la hauteur, agrandi la différence
            #de hauteur entre deux colonnes ou n'a pas rempli des colonnes vides afin de l'encourager à ne pas empiler les blocs.
            fitness -= max(0,ta2 - ta)*5 

            fitness -= max(0,phb2 - phb)*20

            fitness -= verticalite(grille)*10

            fitness -= colonnes_vide(grille)*10

            score += T.clear_rows(grille, positions_statiques)
            fitness += score * 1000 #on la récompense massivement si elle parvient à éliminer des lignes.

        if T.check_lost(positions_statiques):
            fitness += (time.time() - temps_debut)*100 #plus sa partie dure longtemps et plus elle est récompensée.
            for j in grille:
                if j ==(0,0,0):
                    fitness -= 10 #plus elle a laissé de blocs vides lorsqu'elle a perdu et plus elle est pénalisée et vice-versa.
                else:
                    fitness += 20
            run = False

        T.dessine_fenetre(surface, grille)
        T.dessine_prochaine_forme(prochain_tetrimino,surface)
        pygame.display.update()
        
    return fitness


if __name__ == '__main__':
    # Détermine l'emplacement du fichier de configuration.
    # On a eu des problèmes avec cette commande lorsqu'on est pas sous linux
    local_dir = os.path.dirname(__file__)
    chemin_config = os.path.join(local_dir, 'Configuration_AI')
    Run(chemin_config)

