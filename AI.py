
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
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-2:'Gauche',
                  -1: 'Droite',
                  0:'Rotation',
                  1 :'Position', #tetrimino.x, terrimino.y, tetrimino.rotation(=pos dans liste)
                  2:'Espace_libre'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(fitnss, 10)









def fitness(config, genomes):
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