**L’Intelligence Artificielle et Tetris**

Tetris a été l’un des premiers jeux vidéo à faire grand bruit car simple à prendre en main mais très difficile à jouer. 
Tetris est un jeu de puzzle où le joueur doit essayer d’empiler des formes, appelées tétriminos, qui tombent au fur et à mesure de la partie dans une grille de 10 cases par 20. Afin de faire durer la partie le plus longtemps, le joueur doit remplir complètement une ligne, qui disparaît alors. Il y a 7 tétriminos qui représentent les lettres I, O, Z, S, L, J et T. Les contrôles sont simples: mouvements vers la gauche, la droite ou vers le bas ainsi qu’une possibilité de faire tourner la pièce qui tombe. Il est en revanche très difficile de faire durer la partie très longtemps car plus les tétriminos s’enchaînent et moins il devient facile d’envisager des placements avantageux afin d’éliminer les lignes. Nous avons donc programmé sur Python le jeu Tetris puis codé des intelligences artificielles qui y jouent et nous nous sommes demandés comment faire pour qu’elles réussissent au mieux.


![image](https://user-images.githubusercontent.com/104350956/166825156-77478e43-f2eb-475a-8bc1-98fce4a98013.png)



![Tetris_acceuil](https://user-images.githubusercontent.com/104350956/166825272-e7067931-8e22-43b6-8ad9-d852310dcadd.png)





![Tetris_jeu](https://user-images.githubusercontent.com/104350956/166825337-24fcd8cf-d881-4290-ae92-871a19110019.png)








**I- Programmation du Jeu Tetris**

Afin de programmer Tetris, nous devons créer une interface graphique où l’on voit le déroulé de la partie. Pour ce faire, nous avons utilisé le package python Pygame qui permet de créer des fenêtres sur lesquelles nous pouvons afficher des images et des surfaces que l’on peut modifier comme une grille de jeu. Nous commençons par définir notre grille de jeu et les fonctions qui nous permettent de faire évoluer la partie en fonction des actions du joueur.
En particulier, la fonction clear_rows nous sert à éliminer une ligne lorsqu’elle est entièrement complétée par le joueur. Voici son test unitaire:



![test_unitaire](https://user-images.githubusercontent.com/104350956/166825497-37163fe9-c8db-4541-8d39-55541d5ff650.png)




![test_unitaire_grille](https://user-images.githubusercontent.com/104350956/166825523-7ea1e85b-07e6-42e8-9595-dd52bdd0a7c7.png)



Nous avons désormais un jeu de tetris fonctionnel qui fait tomber les tétriminos tout au long de la partie, les génère aléatoirement et qui sait nous dire lorsque la partie est perdue, grâce à la fonction check_lost.

**II- Programmation de l’Intelligence Artificielle**

Pour faire notre intelligence artificielle nous avons utilisé le package python NEAT qui permet de générer et de faire évoluer un réseau de neurones intelligent. Le comportement de chaque neurone est déterminé par son génome et il évolue au fil des générations en fonction de la note (fitness) qui lui est attribuée à la fin de chaque partie. 
Le fichier Configuration_AI permet de configurer la manière d’évoluer des génomes, on y retrouve entre autres des variables qui paramètrent la population de chaque génération, le nombre d’informations que reçoit le neurone, le nombre de réponses qu’il doit donner, la proportion de génomes gardés à la génération suivante ou la probabilité de rajouter des connexions au sein du réseau de neurones.
Nous avons choisi de l’évaluer de la façon suivante:
 - En déterminant si la pose de chaque tétrimino a engendré un gain de hauteur par rapport à l’état précédent, c’est-à-dire si le plus haut bloc de la nouvelle grille est plus haut que celui de la précédente pour éviter qu’elle empile les blocs.
 - Si la pose du dernier tétrimino a ajouté des trous dans la grille, c’est-à-dire des espaces coincés entre deux lignes qui ne peuvent plus être atteints afin qu’elle s’efforce à compléter des lignes, qui est, lorsqu’elle le fait, l’action la plus récompensée pour elle.
 - Si la pose du dernier tétrimino a agrandi l’écart de hauteur entre deux colonnes adjacentes afin qu’elle n’empile pas les blocs sur une même colonne. 
 - Si la pose du dernier tétrimino n’a pas rempli des colonnes vides auparavant pour la même raison que précédemment.
 - Elle reçoit également plus de points si elle essaye d’accoler le tétrimino qu’elle contrôle à d’autres blocs: plus elle se crée de voisins plus elle reçoit de points.

Elle reçoit également des points en fonction du temps au cours duquel elle a joué: plus elle a joué longtemps, plus elle reçoit de points. On l’encourage en la récompensant si elle essaye de faire descendre par elle-même les tétriminos afin qu’elle n’essaye pas de gagner de temps. Nous la pénalisons si elle fait trop de rotations sur le même tétrimino pour éviter certains bugs et si elle essaye de faire des actions qui sont impossibles: bouger à gauche si elle est déjà à la bordure. Finalement, elle est grandement récompensée lorsqu'elle arrive à compléter des lignes.
Le réseau de neurones mute alors pour remplacer les individus dont le génome a obtenu une note faible par des individus dont le génome a obtenu une bonne note.

Nous avons envisagé trois pistes de fonctionnement différentes pour notre intelligence artificielle. Suivent leurs fonctionnements.

 **A) Première Piste**
 
Nous avons tout d’abord essayé de faire correspondre un neurone à un tétrimino pour ensuite réutiliser le génome du neurone le plus performant sur chaque tétrimino au cours d’une partie.
Dans ce cas, on considère une population de neurones, on simule une grille de Tetris  avec des blocs déjà installés et chaque neurone contrôle un tétrimino qui évolue dans cette grille.
Néanmoins nous n’avons obtenu aucun résultat concluant sur une réussite de cette piste. En effet, les tétrimino ne descendaient pas jusqu’en bas de la grille.


![AI_1](https://user-images.githubusercontent.com/104350956/166825765-c2b9f24b-5ea8-4367-a849-f1da6ba4fc43.png)





 **B)Seconde Piste**
 
Suite à cet échec nous avons choisi de refondre notre intelligence artificielle de la façon suivante: une population de 10 neurones jouent au même moment au jeu tetris et obtiennent chacun des résultats à la fin de leurs parties respectives. 
Cette fois, pas de problème de tétrimino qui reste bloqué en haut de la grille mais le plan de jeu de l’IA n’évoluait pas avec les générations. On le voit à partir du score (fitness) moyen de chaques générations qui n’évolue pas même après 500 générations.


![AI_2_plot](https://user-images.githubusercontent.com/104350956/166825823-4c913b96-bd3b-42a0-a16a-975e073bace2.png)



La fitness est négative car nous avions instauré plus de pénalités que de récompenses.

On peut alors se demander pourquoi l’Intelligence Artificielle n’évoluait pas et continuait à empiler les tétriminos les uns sur les autres. 
Cela vient du fait que pour chaque tétrimino, chaque fois que le tétrimino est descendu d’une case, on utilise la réponse du neurone pour choisir le prochain déplacement. Or, la majorité des pénalités et récompenses entrent en jeu lorsque le tétrimino est en bas, donc tout au cours de la chute le neurone décide du déplacement du tétrimino sans aucun appui. De ce fait le mouvement des tétriminos est aléatoire. Néanmoins, le fait que chaque neurone décide d’empiler les tétriminos témoigne d’un effort de mettre en place une stratégie qui reste toutefois non fructueuse car l’Intelligence Artificielle est en fait mal renseignée.
Nous vous renvoyons à la vidéo en pièce jointe dans le mail pour visualiser une partie jouée par cette Intelligence Artificielle.



![AI_2](https://user-images.githubusercontent.com/104350956/166825956-21e9ada6-5bcb-40cc-a450-9b4a5365b1b4.png)



**C)Troisième Piste**

Maintenant que nous avons établi que l’Intelligence Artificielle échouait à cause d’un manque d’informations, nous avons dû revoir la façon dont on lui permettait de faire ses choix.
Pour ce faire, nous nous sommes basés sur la recherche arborescente de Monte-Carlo, en étudiant pour chaque nouveau tétrimino l’ensemble des positions possibles une fois posé. Pour chacune de ces positions on retourne une liste contenant les informations qui permettaient de pénaliser notre Intelligence Artificielle dans nos versions précédentes (nombres de trous ajoutés, hauteur ajoutée,...) et c’est à partir de ces informations que chaque neurone décide de la position finale du tétrimino. Etant donné qu’il y a toujours une récompense lorsqu’une ligne est complétée, l’Intelligence Artificielle apprend à interpréter les informations pour faire le meilleur choix. 
Malheureusement, le calcul des possibilités pour chaque tétrimino est trop lourd et nous n’avons pas réussi à faire tourner l'algorithme pour des raisons de puissance.

**Conclusion**

Au cours de ce projet d’informatique, nous avons programmé un jeu vidéo ainsi qu’une interface graphique mais aussi tenté de développer une intelligence artificielle capable d’y jouer en autonomie.
Nous avons en effet réussi à faire tourner cette intelligence artificielle pendant plusieurs heures sans problème. Néanmoins, nous ne sommes pas parvenus à programmer une intelligence artificielle capable de jouer de façon satisfaisante à Tetris, rares sont les génomes qui sont arrivés à éliminer des lignes. Si nous avions eu accès à du matériel plus puissant nous aurions pu poursuivre la troisième piste, ce ne fut pas le cas. 




