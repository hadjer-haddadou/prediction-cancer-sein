from sklearn.datasets import load_breast_cancer
import pandas as pd

# Chargement des données
data = load_breast_cancer()

# Transformation en tableau pour plus de lisibilité
df = pd.DataFrame(data.data, columns=data.feature_names)
df['cible'] = data.target

print("Taille du dataset :", df.shape)
print("\nLes 5 premières lignes :")
print(df.head())

print("\nRépartition des diagnostics :")
print(df['cible'].value_counts())

print("\nDonnées manquantes par colonne :")
print(df.isnull().sum())