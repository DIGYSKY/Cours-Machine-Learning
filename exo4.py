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

# Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

print("=" * 60)
print("EXERCICE 4 - Prédiction")
print("=" * 60)

# 1. Utiliser le modèle entraîné pour prédire les valeurs sur l'ensemble de test
y_pred = model.predict(X_test)

print("\n1. Prédictions effectuées sur l'ensemble de test")
print(f"   Nombre de prédictions: {len(y_pred)}")

# 2. Comparer les valeurs réelles et les valeurs prédites
print("\n2. Comparaison valeurs réelles vs valeurs prédites:")
print("\n   Premières 10 prédictions:")
print("   " + "-" * 60)
print(f"   {'Index':<6} {'Valeur réelle':<15} {'Valeur prédite':<15} {'Erreur':<15}")
print("   " + "-" * 60)
for i in range(min(10, len(y_test))):
    erreur = abs(y_test[i] - y_pred[i])
    print(f"   {i:<6} {y_test[i]:<15.2f} {y_pred[i]:<15.2f} {erreur:<15.2f}")

# Statistiques de comparaison
print("\n   Statistiques globales:")
print(f"   Valeur réelle - Minimum: {y_test.min():.2f}, Maximum: {y_test.max():.2f}, Moyenne: {y_test.mean():.2f}")
print(f"   Valeur prédite - Minimum: {y_pred.min():.2f}, Maximum: {y_pred.max():.2f}, Moyenne: {y_pred.mean():.2f}")
print(f"   Erreur absolue moyenne: {np.mean(np.abs(y_test - y_pred)):.2f}")
print(f"   Erreur absolue maximale: {np.max(np.abs(y_test - y_pred)):.2f}")
print(f"   Erreur absolue minimale: {np.min(np.abs(y_test - y_pred)):.2f}")

# Exemples d'interprétation
print("\n" + "=" * 60)
print("3. Exemples d'interprétation:")
print("=" * 60)

# Trouver les indices avec les valeurs prédites les plus élevées et les plus faibles
idx_max = np.argmax(y_pred)
idx_min = np.argmin(y_pred)

print(f"\n   Exemple 1 - Valeur prédite ÉLEVÉE (index {idx_max}):")
print(f"   Valeur prédite: {y_pred[idx_max]:.2f}")
print(f"   Valeur réelle: {y_test[idx_max]:.2f}")
print(f"   → Cette prédiction indique une progression du diabète élevée")

print(f"\n   Exemple 2 - Valeur prédite FAIBLE (index {idx_min}):")
print(f"   Valeur prédite: {y_pred[idx_min]:.2f}")
print(f"   Valeur réelle: {y_test[idx_min]:.2f}")
print(f"   → Cette prédiction indique une progression du diabète faible")

# Informations sur la plage des prédictions
print("\n" + "=" * 60)
print("4. Plage des valeurs prédites:")
print("=" * 60)
print(f"   Plage observée dans les prédictions: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
print(f"   Plage observée dans les valeurs réelles: [{y_test.min():.2f}, {y_test.max():.2f}]")
print(f"   Les prédictions peuvent prendre n'importe quelle valeur continue dans cette plage")

