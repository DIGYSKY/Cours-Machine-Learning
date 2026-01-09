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

# Créer et entraîner le modèle k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

print("=" * 60)
print("TP5 - EXERCICE 7 - Prédiction et évaluation")
print("=" * 60)

# 1. Prédire la survie des passagers de l'ensemble de test
y_pred = knn.predict(X_test_scaled)

print("\n1. Prédictions effectuées sur l'ensemble de test")
print(f"   Nombre de prédictions: {len(y_pred)}")
print(f"   Nombre de passagers dans l'ensemble de test: {len(y_test)}")

# Afficher quelques exemples de prédictions
print("\n   Exemples de prédictions (10 premières):")
print("   " + "-" * 60)
print(f"   {'Index':<6} {'Réel':<10} {'Prédit':<10} {'Correct':<10}")
print("   " + "-" * 60)
for i in range(min(10, len(y_test))):
    real = "Survivant" if y_test.iloc[i] == 1 else "Non-survivant"
    pred = "Survivant" if y_pred[i] == 1 else "Non-survivant"
    correct = "✓" if y_test.iloc[i] == y_pred[i] else "✗"
    print(f"   {i:<6} {real:<10} {pred:<10} {correct:<10}")

# 2. Calculer l'accuracy et la matrice de confusion
print("\n" + "=" * 60)
print("2. Métriques de performance:")
print("=" * 60)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   → Le modèle prédit correctement {accuracy*100:.2f}% des passagers")

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("\n   Matrice de confusion:")
print("\n   " + " " * 20 + "Prédit")
print("   " + " " * 15 + f"{'Non-survivant':<15} {'Survivant':<15}")
print("   " + "-" * 45)
print(f"   Réel {'Non-survivant':<10} {cm[0][0]:<15} {cm[0][1]:<15}")
print(f"        {'Survivant':<10} {cm[1][0]:<15} {cm[1][1]:<15}")

# Détails de la matrice de confusion
print("\n   Détails:")
print(f"   - Vrais négatifs (TN): {cm[0][0]} - Non-survivants correctement prédits")
print(f"   - Faux positifs (FP): {cm[0][1]} - Non-survivants prédits comme survivants")
print(f"   - Faux négatifs (FN): {cm[1][0]} - Survivants prédits comme non-survivants")
print(f"   - Vrais positifs (TP): {cm[1][1]} - Survivants correctement prédits")

# Calculer précision, rappel, F1-score
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n   Métriques supplémentaires:")
print(f"   - Précision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   - Rappel: {recall:.4f} ({recall*100:.2f}%)")
print(f"   - F1-score: {f1:.4f} ({f1*100:.2f}%)")

# 3. Interpréter les résultats
print("\n" + "=" * 60)
print("3. Interprétation des résultats:")
print("=" * 60)

print("\n   Performance globale:")
print(f"   → Accuracy de {accuracy*100:.2f}% : le modèle prédit correctement")
print(f"     la survie pour {int(accuracy * len(y_test))} passagers sur {len(y_test)}")

print("\n   Analyse de la matrice de confusion:")
print(f"   → {cm[0][0]} non-survivants correctement identifiés (sur {cm[0][0] + cm[0][1]})")
print(f"   → {cm[1][1]} survivants correctement identifiés (sur {cm[1][0] + cm[1][1]})")
print(f"   → {cm[0][1] + cm[1][0]} erreurs au total ({cm[0][1]} faux positifs, {cm[1][0]} faux négatifs)")

print("\n   Interprétation des erreurs:")
if cm[0][1] > 0:
    print(f"   → {cm[0][1]} passagers non-survivants prédits comme survivants (faux positifs)")
    print(f"     Impact: surestimation de la survie")
if cm[1][0] > 0:
    print(f"   → {cm[1][0]} passagers survivants prédits comme non-survivants (faux négatifs)")
    print(f"     Impact: sous-estimation de la survie")

print("\n   Qualité du modèle:")
if accuracy >= 0.8:
    print(f"   → Performance EXCELLENTE (≥80%)")
elif accuracy >= 0.7:
    print(f"   → Performance BONNE (≥70%)")
else:
    print(f"   → Performance MODÉRÉE (<70%)")

print(f"\n   → Le modèle k-NN avec k=5 et normalisation donne de bons résultats")
print(f"   → L'accuracy de {accuracy*100:.2f}% est satisfaisante pour ce problème")
print(f"   → Le modèle est équilibré entre précision ({precision*100:.2f}%) et rappel ({recall*100:.2f}%)")

# Rapport de classification complet
print("\n" + "=" * 60)
print("Rapport de classification complet:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=['Non-survivant', 'Survivant']))

