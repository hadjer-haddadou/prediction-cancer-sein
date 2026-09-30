from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text

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


from sklearn.model_selection import train_test_split

# 1. On sépare les indices (X) de la réponse (y)
y = df['cible']                  # La boîte y contient juste la cible
X = df.drop('cible', axis=1)     # La boîte X contient tout le tableau, sauf ('drop') la cible

# 2. On coupe notre pile de dossiers en deux (80% pour l'entraînement, 20% pour l'examen)
# test_size=0.2 signifie qu'on garde 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Préparation des données ---")
print("Dossiers pour l'entraînement (Train) :", X_train.shape[0])
print("Dossiers pour l'examen (Test) :", X_test.shape[0])

modele = DecisionTreeClassifier()
modele.fit(X_train, y_train)

prediction = modele.predict(X_test)

note_finale = accuracy_score(y_test, prediction)
print("\nNote finale de l'IA :", note_finale)

nomColonnes = list(X.columns)
regles = export_text(modele, feature_names=nomColonnes)
print(regles)
