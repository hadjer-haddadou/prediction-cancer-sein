from sklearn.datasets import load_breast_cancer
import pandas as pd
#panda outil pour manipuler des tableaux

# Chargement des données
data = load_breast_cancer()
#chercher le dossier médical des patientes dans les archives du pc 


# Transformation en tableau pour plus de lisibilité
df = pd.DataFrame(data.data, columns=data.feature_names)
#rangement des données via l'outil panda 

df['cible'] = data.target
#création de nouvelle colonne cible = diagnostic final des médecins (0 pour malin, 1 pour bénin).


print("Taille du dataset :", df.shape)
#demande les dimensions du tableau
print("\nLes 5 premières lignes :")
print(df.head())

print("\nRépartition des diagnostics :")
print(df['cible'].value_counts())
#value_counts de 0 et de 1

print("\nDonnées manquantes par colonne :")
print(df.isnull().sum())