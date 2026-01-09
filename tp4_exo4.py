import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 60)
print("TP4 - EXERCICE 4 - Entraînement du modèle")
print("=" * 60)

model = LogisticRegression(max_iter=1000, random_state=42)

print("\n1. Modèle de régression logistique créé:")
print(f"   Type: LogisticRegression")
print(f"   Paramètres: max_iter=1000, random_state=42")

model.fit(X_train_scaled, y_train)

print("\n2. Modèle entraîné sur l'ensemble d'apprentissage")
print(f"   Nombre d'observations d'entraînement: {X_train_scaled.shape[0]}")
print(f"   Nombre de variables explicatives: {X_train_scaled.shape[1]}")
print(f"   Nombre d'itérations nécessaires: {model.n_iter_[0]}")

print("\n   Coefficients du modèle (premiers 5):")
for i in range(min(5, len(cancer.feature_names))):
    print(f"      {cancer.feature_names[i]}: {model.coef_[0][i]:.4f}")

print(f"\n   Intercept (biais): {model.intercept_[0]:.4f}")

print("\n" + "=" * 60)
print("3. Informations sur la fonction logistique (sigmoïde):")
print("=" * 60)
print("   Le modèle utilise la fonction sigmoïde pour transformer")
print("   les valeurs linéaires en probabilités entre 0 et 1.")
print(f"   Nombre de coefficients: {len(model.coef_[0])}")
print(f"   Forme de la sortie: probabilité de classe (0 à 1)")

