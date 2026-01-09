import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

# Charger le dataset Diabetes depuis Scikit-learn
diabetes = load_diabetes()

# Créer un DataFrame pour faciliter l'exploration
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

print("=" * 60)
print("EXERCICE 1 - Chargement et exploration du dataset Diabetes")
print("=" * 60)

# 1. Afficher le nombre d'observations et de variables explicatives
print("\n1. Nombre d'observations (patients):", diabetes.data.shape[0])
print("   Nombre de variables explicatives:", diabetes.data.shape[1])

# 2. Afficher les noms des variables explicatives et la variable cible
print("\n2. Noms des variables explicatives:")
for i, feature_name in enumerate(diabetes.feature_names, 1):
    print(f"   {i}. {feature_name}")

print("\n   Variable cible: 'target'")

# 3. Statistiques sur la variable cible
print("\n3. Statistiques de la variable cible:")
print(f"   Type: {type(diabetes.target[0])}")
print(f"   Valeurs uniques: {len(np.unique(diabetes.target))} valeurs différentes")
print(f"   Plage de valeurs: [{diabetes.target.min():.2f}, {diabetes.target.max():.2f}]")
print(f"   Moyenne: {diabetes.target.mean():.2f}")
print(f"   Écart-type: {diabetes.target.std():.2f}")

# Aperçu des données
print("\n" + "=" * 60)
print("Aperçu des données:")
print("=" * 60)
print(df.head())
print("\nInformations sur le dataset:")
print(df.info())
