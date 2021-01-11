# perceptron-POS-TAL1
Projet final de TAL1 de LI M1S1, P7, 2020. Romane Gallienne et Cécile Guitel.

Ceci est un classifieur PoS multiclasse sous la forme d'un perceptron moyenné basé sur le corpus français GSD de Universal Dependencies.

Pour en calculer les poids, entrez la ligne de commande suivante : `python3 perceptron_train.py` 
Pour évaluation la précision, entrez la ligne de commande suivante : `python3 perceptron_evaluate_accuracy.py`
Pour créer une matrice de confusion et obtenir un heatmap, entrez la ligne de commande suivante : `python3 perceptron_evaluate_confusion.py`

Actuellement, le perceptron est entrainé sur le corpus train de GSD, il évalue la précision sur GSD, Spoken et Old French, et il construit la matrice de confusion sur les données test de Old French.
Si vous voulez changer de fichier, il faut éditer le code dans le fichier `perceptron_train.py`, `perceptron_evaluate_accuracy.py` et perceptron_evaluate_confusion.py


Tous les fichiers sont disponibles à cette adresse : https://github.com/CGuitel/perceptron-POS-TAL1
