import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)

print("=" * 60)
print("TP4 - EXERCICE 3 - Normalisation")
print("=" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n1. Normalisation appliquée (StandardScaler):")
print(f"   Méthode: Standardisation (moyenne=0, écart-type=1)")
print(f"   Données d'entraînement normalisées: shape {X_train_scaled.shape}")
print(f"   Données de test normalisées: shape {X_test_scaled.shape}")

print("\n   Statistiques après normalisation (échantillon):")
print(f"   Moyenne: {X_train_scaled.mean():.4f} (≈ 0)")
print(f"   Écart-type: {X_train_scaled.std():.4f} (≈ 1)")

print("\n2. Justification du choix (StandardScaler):")
print("   • Standardisation: transforme les données pour avoir moyenne=0 et écart-type=1")
print("   • Préserve la forme de la distribution originale")
print("   • Adapté à la régression logistique qui utilise la descente de gradient")
print("   • Évite que certaines variables dominent à cause de leur échelle")

print("\n" + "=" * 60)
print("3. Comparaison des performances:")
print("=" * 60)

print("\n   Modèle SANS normalisation:")
model_no_scaling = LogisticRegression(max_iter=1000, random_state=42)
model_no_scaling.fit(X_train, y_train)
y_pred_no_scaling = model_no_scaling.predict(X_test)
accuracy_no_scaling = accuracy_score(y_test, y_pred_no_scaling)

print(f"   Accuracy: {accuracy_no_scaling:.4f} ({accuracy_no_scaling*100:.2f}%)")
print(f"   Nombre d'itérations: {model_no_scaling.n_iter_[0]}")

print("\n   Modèle AVEC normalisation:")
model_scaling = LogisticRegression(max_iter=1000, random_state=42)
model_scaling.fit(X_train_scaled, y_train)
y_pred_scaling = model_scaling.predict(X_test_scaled)
accuracy_scaling = accuracy_score(y_test, y_pred_scaling)

print(f"   Accuracy: {accuracy_scaling:.4f} ({accuracy_scaling*100:.2f}%)")
print(f"   Nombre d'itérations: {model_scaling.n_iter_[0]}")

print("\n   Comparaison:")
print(f"   Amélioration de l'accuracy: {(accuracy_scaling - accuracy_no_scaling)*100:.2f}%")
print(f"   Réduction du nombre d'itérations: {model_no_scaling.n_iter_[0] - model_scaling.n_iter_[0]} itérations")

if accuracy_scaling > accuracy_no_scaling:
    print("\n   → La normalisation améliore les performances du modèle")
else:
    print("\n   → Les performances sont similaires")

print("\n" + "=" * 60)
print("Rapport de classification (avec normalisation):")
print("=" * 60)
print(classification_report(y_test, y_pred_scaling, target_names=cancer.target_names))

print("\n" + "=" * 60)
print("Matrice de confusion (avec normalisation):")
print("=" * 60)
print(confusion_matrix(y_test, y_pred_scaling))

