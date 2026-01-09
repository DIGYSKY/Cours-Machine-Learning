import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Charger le dataset Diabetes depuis Scikit-learn
diabetes = load_diabetes()

# Séparer le dataset en X (variables explicatives) et y (variable cible)
X = diabetes.data
y = diabetes.target

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print("=" * 60)
print("EXERCICE 3 - Modélisation par régression linéaire")
print("=" * 60)

# 1. Créer un modèle de régression linéaire
model = LinearRegression()

print("\n1. Modèle créé: LinearRegression")

# 2. Entraîner le modèle sur l'ensemble d'apprentissage
model.fit(X_train, y_train)

print("\n2. Modèle entraîné sur l'ensemble d'apprentissage")
print(f"   Nombre d'observations d'entraînement: {X_train.shape[0]}")

# 3. Afficher les coefficients du modèle et l'ordonnée à l'origine
print("\n3. Coefficients du modèle:")
print(f"   Nombre de variables explicatives: {len(diabetes.feature_names)}")
print("\n   Coefficients par variable:")
for i, (feature_name, coef) in enumerate(zip(diabetes.feature_names, model.coef_), 1):
    print(f"   {i:2d}. {feature_name:15s}: {coef:10.4f}")

print(f"\n   Ordonnée à l'origine (intercept): {model.intercept_:.4f}")

print(f"\n   Coefficient le plus élevé: {diabetes.feature_names[np.argmax(np.abs(model.coef_))]} ({model.coef_[np.argmax(np.abs(model.coef_))]:.4f})")
print(f"   Coefficient le plus faible: {diabetes.feature_names[np.argmin(np.abs(model.coef_))]} ({model.coef_[np.argmin(np.abs(model.coef_))]:.4f})")

