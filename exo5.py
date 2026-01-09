import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur les ensembles d'entraînement et de test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("=" * 60)
print("EXERCICE 5 - Évaluation des performances")
print("=" * 60)

# 1. Calculer les métriques de performance
print("\n1. Calcul des métriques de performance:")

# Sur l'ensemble d'entraînement
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

# Sur l'ensemble de test
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)

print("\n   Ensemble d'entraînement:")
print(f"   MSE (Mean Squared Error): {mse_train:.4f}")
print(f"   RMSE (Root Mean Squared Error): {rmse_train:.4f}")
print(f"   R² (Coefficient de détermination): {r2_train:.4f}")

print("\n   Ensemble de test:")
print(f"   MSE (Mean Squared Error): {mse_test:.4f}")
print(f"   RMSE (Root Mean Squared Error): {rmse_test:.4f}")
print(f"   R² (Coefficient de détermination): {r2_test:.4f}")

# 3. Comparer les performances
print("\n" + "=" * 60)
print("3. Comparaison des performances:")
print("=" * 60)

print(f"\n   Différence entre entraînement et test:")
print(f"   MSE:  {mse_train:.4f} (train) vs {mse_test:.4f} (test) - Différence: {abs(mse_train - mse_test):.4f}")
print(f"   RMSE: {rmse_train:.4f} (train) vs {rmse_test:.4f} (test) - Différence: {abs(rmse_train - rmse_test):.4f}")
print(f"   R²:   {r2_train:.4f} (train) vs {r2_test:.4f} (test) - Différence: {abs(r2_train - r2_test):.4f}")

# Analyse de la différence
if abs(r2_train - r2_test) < 0.05:
    print("\n   → Les performances sont similaires entre train et test (bonne généralisation)")
elif r2_train > r2_test + 0.05:
    print("\n   → Performance meilleure sur train que test (léger surapprentissage possible)")
else:
    print("\n   → Performance meilleure sur test que train (cas rare)")

# Informations supplémentaires
print("\n" + "=" * 60)
print("Informations supplémentaires:")
print("=" * 60)
print(f"\n   Erreur moyenne sur test: {np.mean(np.abs(y_test - y_test_pred)):.2f}")
print(f"   Écart-type des erreurs sur test: {np.std(y_test - y_test_pred):.2f}")
print(f"   Pourcentage d'erreur moyenne: {(np.mean(np.abs(y_test - y_test_pred)) / np.mean(y_test)) * 100:.2f}%")

