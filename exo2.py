import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Charger le dataset Diabetes depuis Scikit-learn
diabetes = load_diabetes()

# Créer un DataFrame pour faciliter l'exploration
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

print("=" * 60)
print("EXERCICE 2 - Préparation des données")
print("=" * 60)

# 1. Séparer le dataset en X (variables explicatives) et y (variable cible)
X = diabetes.data
y = diabetes.target

print("\n1. Séparation du dataset:")
print(f"   X (variables explicatives): shape {X.shape}")
print(f"   y (variable cible): shape {y.shape}")

# 2. Diviser les données en ensemble d'entraînement et ensemble de test
# 3. Choisir un pourcentage pour l'ensemble de test (20%)
test_size = 0.2
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    random_state=random_state
)

print(f"\n2. Division des données (test_size={test_size*100}%):")
print(f"   Ensemble d'entraînement (X_train): {X_train.shape[0]} observations")
print(f"   Ensemble d'entraînement (y_train): {y_train.shape[0]} observations")
print(f"   Ensemble de test (X_test): {X_test.shape[0]} observations")
print(f"   Ensemble de test (y_test): {y_test.shape[0]} observations")

print(f"\n3. Répartition:")
print(f"   Entraînement: {X_train.shape[0]} observations ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Test: {X_test.shape[0]} observations ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"   Total: {len(X)} observations")

# Aperçu des données d'entraînement
print("\n" + "=" * 60)
print("Aperçu des données d'entraînement:")
print("=" * 60)
print(f"X_train (premières lignes):\n{X_train[:5]}")
print(f"\ny_train (premières valeurs):\n{y_train[:5]}")

print("\n" + "=" * 60)
print("Aperçu des données de test:")
print("=" * 60)
print(f"X_test (premières lignes):\n{X_test[:5]}")
print(f"\ny_test (premières valeurs):\n{y_test[:5]}")

