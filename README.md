# Prédiction de la Malignité des Tumeurs du Sein

## Description du projet
Ce projet d'intelligence artificielle appliquée à la santé vise à développer un modèle de machine learning capable de classifier des tumeurs du sein (bénignes ou malignes) à partir de caractéristiques cellulaires. 


## Source des données
Le jeu de données utilisé est le **Breast Cancer Wisconsin (Diagnostic) Dataset**, initialement hébergé sur l'UCI Machine Learning Repository. 
Il contient des caractéristiques géométriques cellulaires calculées à partir d'images numérisées de biopsies mammaires.

- **Patientes :** 569 instances
- **Caractéristiques :** 30 variables numériques (rayon, texture, périmètre, surface, lissage, etc.)
- **Cible :** Maligne (0) ou Bénigne (1)


## Technologies et Outils 
- **Python** : Langage de programmation principal.
- **Pandas** : Pour la manipulation, l'exploration et la structuration des données sous forme de tableaux (DataFrames).
- **Scikit-Learn** : Bibliothèque de Machine Learning utilisée pour la séparation des données, l'entraînement de l'algorithme et le calcul du score.
- **Environnement de développement** : VS Code sous Linux (Zorin OS), avec Git/GitHub pour le versioning du code.

## 🧠 Modèle et Résultats
- **Algorithme choisi :** Arbre de Décision (`DecisionTreeClassifier`).
- **Pourquoi ce choix ?** C'est un modèle "boîte blanche" (interprétable). Il permet de visualiser exactement quelles questions mathématiques la machine se pose pour arriver à son diagnostic, fonctionnant comme un organigramme médical.
- **Résultat :** Le modèle a atteint une précision de **93,8 %** sur les données de test (patients jamais vus par l'IA lors de son entraînement). 
- **Découverte :** L'algorithme a déterminé de façon autonome que le critère géométrique le plus déterminant pour détecter une tumeur maligne dans ce jeu de données est la concavité moyenne des points du noyau cellulaire (`mean concave points`).