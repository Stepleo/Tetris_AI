# essai modification git
import numpy as np
import pygame
import random

# creating the data structure for pieces
# setting up global vars
# functions
# - create_grid
# - draw_grid
# - draw_window
# - rotating shape in main
# - setting up the main

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.font.init()

# Variables Globales
Largeur_fenetre = 800
Hauteur_fenetre = 700
Largeur_jeu = 300  # 30 x 10 donne 300
Hauteur_jeu = 600  # 30 x 20 donne 600
Taille_bloc = 30

# On ne veut pas que le terrain de jeu recouvre la totalité de la fenêtre que l'on va créer donc on
# choisit des valeurs différentes pour ses dimensions. De plus, vu que le jeu se déroule dans un rectangle
# de 20 blocs par 10, il faut que l'on puisse placer 10 blocs en largeur et 20 en hauteur.

# Définition de l'origine de notre grille de jeu: en pygame le point (0,0) de notre fenêtre ne se trouve
# ni au centre ni au milieu de celle-ci mais en haut à gauche. On conserve cette notion lorsque l'on crée
# l'origine de notre grille de jeu.

x0 = (Largeur_fenetre - Largeur_jeu) // 2
y0 = Hauteur_fenetre - Hauteur_jeu

# Ces deux valeurs nous serviront à localiser chaque tétrimino qui sera en train de tomber.
# Définissons maintenant ces tétriminos.


# Formes et formes des rotations possibles des tétriminos

S = [['.....',
      '......',
      '..00..',
      '.00...',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

# On utilise des listes de listes pour définir nos tétriminos, les zéros représentent les endroits où il
# y a des blocs. Chaque tétrimino a plusieurs formes possibles et elles sont représentées par les
# différentes dispositions des zéros.

Tetriminos = [S, Z, I, O, J, L, T]
Tetriminos_couleur = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]


# On indexe nos tétriminos de 0 à 6 et on utilise un codage rvb pour déterminer les couleurs de ceux-ci


class Piece(object):
    def __init__(self, x, y, tetrimino):
        self.x = x
        self.y = y
        self.tetrimino = tetrimino
        self.couleur = Tetriminos_couleur[Tetriminos.index(tetrimino)]
        self.rotation = 0


def creation_grille(positions_statiques):
    grille = [[(0, 0, 0) for k in range(10)] for k in range(20)]

    for i in range(len(grille)):
        for j in range(len(grille[i])):
            if (j, i) in positions_statiques:
                c = positions_statiques[(j, i)]
                grille[i][j] = c
    return grille


def conversion_format(tetrimino):
    positions = []
    format = tetrimino.tetrimino[tetrimino.rotation % len(tetrimino.tetrimino)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((tetrimino.x + j, tetrimino.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def espace_disponible(tetrimino, grille):
    position_valide = [[(j, i) for j in range(10) if grille[i][j] == (0, 0, 0)] for i in range(20)]
    position_valide = [j for sub in position_valide for j in sub]

    formate = conversion_format(tetrimino)

    for pos in formate:
        if pos not in position_valide:
            if pos[1] > -1:
                return False
    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


def obtenir_forme():
    return Piece(5, 0, random.choice(Tetriminos))


def dessine_texte_milieu(text, size, color, surface):
    police = pygame.font.SysFont("comicsans", size, bold=True)
    label = police.render(text, True, color)
    surface.blit(label, (x0 + Largeur_jeu / 2 - (label.get_width() / 2), y0 + Hauteur_jeu / 2 - label.get_width() / 2))


def dessine_grille(surface, grille):
    for i in range(len(grille)):
        pygame.draw.line(surface, (128, 128, 128), (x0, y0 + i * Taille_bloc), (x0 + Largeur_jeu, y0 + i * Taille_bloc))
        for j in range(len(grille[i])):
            pygame.draw.line(surface, (128, 128, 128), (x0 + j * Taille_bloc, y0),
                             (x0 + j * Taille_bloc, y0 + Hauteur_jeu))


def clear_rows(grille, positions_statiques):
    inc = 0
    for i in range(len(grille) - 1, -1, -1):
        row = grille[i]
        if (0, 0, 0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del positions_statiques[(j, i)]
                except:
                    continue

    if inc > 0:
        for key in sorted(list(positions_statiques), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                positions_statiques[newKey] = positions_statiques.pop(key)
    return (inc)


def dessine_prochaine_forme(tetrimino, surface):
    police = pygame.font.SysFont('comicsans', 30)
    label = police.render('Prochaine Forme', True, (255, 255, 255))

    sx = x0 + Largeur_jeu + 50
    sy = y0 + Hauteur_jeu / 2 + 100  # -100
    format = tetrimino.tetrimino[tetrimino.rotation % len(tetrimino.tetrimino)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, tetrimino.couleur,
                                 (sx + j * Taille_bloc, sy + i * Taille_bloc, Taille_bloc, Taille_bloc), 0)

    surface.blit(label, (sx + 10, sy - 30))


def dessine_fenetre(surface, grille, score=0):
    surface.fill((0, 0, 0))

    pygame.font.init()
    police = pygame.font.SysFont('comicsans', 60)
    label = police.render('Tetrisae', True, (255, 255, 255))

    surface.blit(label, (x0 + Largeur_jeu / 2 - (label.get_width() / 2), 30))

    police = pygame.font.SysFont('comicsans', 30)
    label = police.render('Score:' + str(score), True, (255, 255, 255))

    sx = x0 + Largeur_jeu + 50
    sy = y0 + Hauteur_jeu / 2 + 100  # -100
    surface.blit(label, (sx + 20, sy + 160))

    for i in range(len(grille)):
        for j in range(len(grille[i])):
            pygame.draw.rect(surface, grille[i][j],
                             (x0 + j * Taille_bloc, y0 + i * Taille_bloc, Taille_bloc, Taille_bloc), 0)

    pygame.draw.rect(surface, (255, 0, 0), (x0, y0, Largeur_jeu, Hauteur_jeu), 4)

    dessine_grille(surface, grille)
    # pygame.display.update()


def main(win):
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

    while run:
        grille = creation_grille(positions_statiques)
        temps_de_chute += temps.get_rawtime()
        temps_de_jeu += temps.get_rawtime()
        temps.tick()

        if temps_de_chute / 1000 > vitesse_chute:
            temps_de_chute = 0
            tetrimino_actuel.y += 1
            if not (espace_disponible(tetrimino_actuel, grille)) and tetrimino_actuel.y > 0:
                tetrimino_actuel.y -= 1
                changement_tetrimino = True

        if temps_de_jeu / 1000 > 5:
            temps_de_jeu = 0
            if vitesse_chute > 0.12:
                vitesse_chute -= 0.005

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    tetrimino_actuel.x -= 1
                    if not (espace_disponible(tetrimino_actuel, grille)):
                        tetrimino_actuel.x += 1
                if event.key == pygame.K_RIGHT:
                    tetrimino_actuel.x += 1
                    if not (espace_disponible(tetrimino_actuel, grille)):
                        tetrimino_actuel.x -= 1
                if event.key == pygame.K_DOWN:
                    tetrimino_actuel.y += 1
                    if not (espace_disponible(tetrimino_actuel, grille)):
                        tetrimino_actuel.y -= 1
                if event.key == pygame.K_UP:
                    tetrimino_actuel.rotation += 1
                    if not (espace_disponible(tetrimino_actuel, grille)):
                        tetrimino_actuel.rotation -= 1

        tetrimino_pos = conversion_format(tetrimino_actuel)

        for i in range(len(tetrimino_pos)):
            x, y = tetrimino_pos[i]
            if y > -1:
                grille[y][x] = tetrimino_actuel.couleur

        if changement_tetrimino:
            for pos in tetrimino_pos:
                p = (pos[0], pos[1])
                positions_statiques[p] = tetrimino_actuel.couleur
            tetrimino_actuel = prochain_tetrimino
            prochain_tetrimino = obtenir_forme()
            changement_tetrimino = False
            score += clear_rows(grille, positions_statiques) * 10

        dessine_fenetre(win, grille, score)
        dessine_prochaine_forme(prochain_tetrimino, win)
        pygame.display.update()

        if check_lost(positions_statiques):
            dessine_texte_milieu("PERDU", 80, (255, 255, 255), win)
            pygame.display.update()
            pygame.time.delay(1500)
            run = False


def main_menu(win):
    run = True
    while run:
        win.fill((0, 0, 0))
        dessine_texte_milieu("Appuyez pour commencer", 60, (255, 255, 255), win)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main(win)
    pygame.display.quit()


#win = pygame.display.set_mode((Largeur_fenetre, Hauteur_fenetre))
#pygame.display.set_caption('Tetrisae')
#main_menu(win)
