import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Charger et préparer les données
df = pd.read_csv('train.csv')

# Préparation basique des données
df['Age'].fillna(df['Age'].median(), inplace=True)

df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'].fillna('S', inplace=True)
df['Embarked_encoded'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
X = df[features]
y = df['Survived']

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("TP5 - EXERCICE 5 - Normalisation des données")
print("=" * 60)

# 1. Appliquer une normalisation aux variables explicatives
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n1. Normalisation appliquée (StandardScaler):")
print(f"   Données d'entraînement normalisées: shape {X_train_scaled.shape}")
print(f"   Données de test normalisées: shape {X_test_scaled.shape}")

# Statistiques avant/après normalisation
print("\n   Statistiques avant normalisation (échantillon):")
print(f"   Moyenne: {X_train.mean().values[:3]}")
print(f"   Écart-type: {X_train.std().values[:3]}")

print("\n   Statistiques après normalisation (échantillon):")
print(f"   Moyenne: {X_train_scaled.mean(axis=0)[:3]} (≈ 0)")
print(f"   Écart-type: {X_train_scaled.std(axis=0)[:3]} (≈ 1)")

# 2. Expliquer pourquoi la normalisation est cruciale pour k-NN
print("\n" + "=" * 60)
print("2. Pourquoi la normalisation est cruciale pour k-NN:")
print("=" * 60)
print("\n   • k-NN utilise la DISTANCE entre points pour classer")
print("   • Sans normalisation, variables à grande échelle dominent le calcul de distance")
print("   • Exemple: Fare (0-500) vs Pclass (1-3) → Fare domine")
print("   • Normalisation garantit que toutes les variables contribuent équitablement")
print("   • Améliore significativement les performances du modèle")

# 3. Comparer les performances avec et sans normalisation
print("\n" + "=" * 60)
print("3. Comparaison des performances:")
print("=" * 60)

# Modèle SANS normalisation
print("\n   Modèle k-NN SANS normalisation (k=5):")
knn_no_scaling = KNeighborsClassifier(n_neighbors=5)
knn_no_scaling.fit(X_train, y_train)
y_pred_no_scaling = knn_no_scaling.predict(X_test)
accuracy_no_scaling = accuracy_score(y_test, y_pred_no_scaling)

print(f"   Accuracy: {accuracy_no_scaling:.4f} ({accuracy_no_scaling*100:.2f}%)")

# Modèle AVEC normalisation
print("\n   Modèle k-NN AVEC normalisation (k=5):")
knn_scaling = KNeighborsClassifier(n_neighbors=5)
knn_scaling.fit(X_train_scaled, y_train)
y_pred_scaling = knn_scaling.predict(X_test_scaled)
accuracy_scaling = accuracy_score(y_test, y_pred_scaling)

print(f"   Accuracy: {accuracy_scaling:.4f} ({accuracy_scaling*100:.2f}%)")

# Comparaison
print("\n   Comparaison:")
improvement = accuracy_scaling - accuracy_no_scaling
print(f"   Amélioration: {improvement:.4f} ({improvement*100:.2f}%)")
if accuracy_scaling > accuracy_no_scaling:
    print("   → La normalisation améliore les performances")
else:
    print("   → Les performances sont similaires")
