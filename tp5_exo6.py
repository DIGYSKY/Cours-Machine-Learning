import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Charger et préparer les données
df = pd.read_csv('train.csv')

# Préparation des données
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'].fillna('S', inplace=True)
df['Embarked_encoded'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Sélectionner les variables explicatives
features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_encoded']
X = df[features]
y = df['Survived']

# Diviser en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normaliser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 60)
print("TP5 - EXERCICE 6 - Mise en place du modèle k-NN")
print("=" * 60)

# 1. Créer un modèle k-NN avec une valeur initiale de k
k_initial = 5
knn = KNeighborsClassifier(n_neighbors=k_initial)

print(f"\n1. Modèle k-NN créé:")
print(f"   Type: KNeighborsClassifier")
print(f"   k (nombre de voisins): {k_initial}")
print(f"   Distance: euclidienne (par défaut)")

# 2. Entraîner le modèle sur l'ensemble d'apprentissage
knn.fit(X_train_scaled, y_train)

print(f"\n2. Modèle entraîné sur l'ensemble d'apprentissage")
print(f"   Nombre d'observations d'entraînement: {X_train_scaled.shape[0]}")
print(f"   Nombre de variables explicatives: {X_train_scaled.shape[1]}")

# Prédictions
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   Performance sur l'ensemble de test:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 3. Expliquer le principe de fonctionnement de l'algorithme
print("\n" + "=" * 60)
print("3. Principe de fonctionnement de l'algorithme k-NN:")
print("=" * 60)

print("\n   Étape 1 - Calcul des distances:")
print("   • Pour chaque point de test, calculer la distance euclidienne")
print("   • avec tous les points d'entraînement")
print("   • Formule: d = √[(x₁-x₂)² + (y₁-y₂)² + ...]")

print("\n   Étape 2 - Sélection des k plus proches voisins:")
print(f"   • Identifier les {k_initial} points d'entraînement les plus proches")
print("   • Ces voisins sont ceux avec les distances les plus petites")

print("\n   Étape 3 - Vote majoritaire:")
print("   • Regarder les classes des k voisins")
print("   • Attribuer au point de test la classe la plus fréquente")
print("   • En cas d'égalité, choix aléatoire ou distance pondérée")

print("\n   Exemple concret:")
print("   • Point de test avec k=5")
print("   • 5 voisins les plus proches: [0, 1, 1, 1, 0]")
print("   • Vote: 3 votes pour classe 1, 2 votes pour classe 0")
print("   • Prédiction: classe 1 (survivant)")

print("\n   Caractéristiques importantes:")
print("   • Algorithme NON paramétrique: ne fait pas d'hypothèses sur les données")
print("   • Apprentissage paresseux (lazy): pas de phase d'entraînement complexe")
print("   • Stocke tous les points d'entraînement en mémoire")
print("   • Sensible aux valeurs aberrantes et au bruit")

print("\n   Avantages:")
print("   • Simple à comprendre et implémenter")
print("   • Efficace pour petits datasets")
print("   • Pas d'hypothèse sur la distribution des données")

print("\n   Inconvénients:")
print("   • Coûteux en calcul pour grands datasets")
print("   • Sensible aux variables non pertinentes")
print("   • Nécessite normalisation des données")

# Rapport de classification
print("\n" + "=" * 60)
print("Rapport de classification:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=['Non-survivant', 'Survivant']))

print("\n" + "=" * 60)
print("Matrice de confusion:")
print("=" * 60)
cm = confusion_matrix(y_test, y_pred)
print(f"\n   {'Réel/Pridit':<15} {'Non-survivant':<15} {'Survivant':<15}")
print("   " + "-" * 45)
print(f"   {'Non-survivant':<15} {cm[0][0]:<15} {cm[0][1]:<15}")
print(f"   {'Survivant':<15} {cm[1][0]:<15} {cm[1][1]:<15}")

