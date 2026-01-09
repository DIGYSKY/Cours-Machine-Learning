import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

print("=" * 60)
print("TP4 - EXERCICE 2 - Préparation des données")
print("=" * 60)

X = cancer.data
y = cancer.target

print("\n1. Séparation des données:")
print(f"   X (variables explicatives): shape {X.shape}")
print(f"   y (variable cible): shape {y.shape}")
print(f"   Nombre de variables explicatives: {X.shape[1]}")
print(f"   Nombre d'échantillons: {X.shape[0]}")

test_size = 0.3
random_state = 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    random_state=random_state,
    stratify=y
)

print(f"\n2. Division du dataset (test_size={test_size*100}%):")
print(f"   Ensemble d'entraînement (X_train): {X_train.shape[0]} observations ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Ensemble d'entraînement (y_train): {y_train.shape[0]} observations")
print(f"   Ensemble de test (X_test): {X_test.shape[0]} observations ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"   Ensemble de test (y_test): {y_test.shape[0]} observations")
print(f"   Total: {len(X)} observations")

print("\n   Distribution des classes dans l'ensemble d'entraînement:")
train_class_counts = pd.Series(y_train).value_counts().sort_index()
for idx, count in train_class_counts.items():
    class_name = cancer.target_names[idx]
    percentage = (count / len(y_train)) * 100
    print(f"      Classe {idx} ({class_name}): {count} échantillons ({percentage:.1f}%)")

print("\n   Distribution des classes dans l'ensemble de test:")
test_class_counts = pd.Series(y_test).value_counts().sort_index()
for idx, count in test_class_counts.items():
    class_name = cancer.target_names[idx]
    percentage = (count / len(y_test)) * 100
    print(f"      Classe {idx} ({class_name}): {count} échantillons ({percentage:.1f}%)")

print("\n" + "=" * 60)
print("Statistiques sur les variables explicatives (échantillon):")
print("=" * 60)
print(f"   Plage de valeurs - Min: {X.min():.2f}, Max: {X.max():.2f}")
print(f"   Moyenne: {X.mean():.2f}, Écart-type: {X.std():.2f}")
print(f"   → Les variables ont des échelles différentes, normalisation nécessaire")

