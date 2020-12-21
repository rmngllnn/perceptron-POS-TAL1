### Travail à effectuer

Certains points seront précisés au fur et à mesure des dernières séances de TP. Les points pouvant déjà être traités sont précédés d'une astérisque

- [ ]  Développement d'un classifieur multiclasse (régression logistique, perceptron ou SVM). Votre classifieur devra (à minima) implémenter une méthode fit capable d'estimer les paramètres du modèle à partir d'un ensemble d'apprentissage et une méthode predict qui retournera l'étiquette d'une observation

- [ ] Extraction des caractéristiques pour le PoS tagging. Les caractéristiques généralement utilisées sont le mot, le mot précédent, le mot suivant, la présence de certains préfixes, la présence de lettre en majuscule ; vous êtes libres de définir toutes les caractéristiques vous semblant pertinentes.

- [ ] Évaluation des performances du classifieur et des erreurs fréquentes. On distinguera notamment les performances obtenus sur les mots vue en apprentissage et sur les mots n'apparaissant qu'en test (mot hors vocabulaire).

- [ ] Évaluation de votre PoS tagger sur des données hors-domaine

- [ ] Amélioration des performances sur les données hors-domaines. On pourra considérer les deux méthodes suivantes :

  ​	1. Sélection des exemples d'apprentissage en fonction du domaine cible. Pour cela : on apprends un modèle de langue sur le domaine cible (par exemple avec [kenLM](https://github.com/kpu/kenlm)) et on sélectionne les phrases du [corpus](https://moodle.u-paris.fr/mod/resource/view.php?id=138839) d'apprentissage ayant la plus forte probabilité d'avoir été générée par ce modèle de langue

  ​	2. Définition de nouvelles caractéristiques robuste au changement de domaine (p. ex. après avoir identifié les erreurs fréquentes)