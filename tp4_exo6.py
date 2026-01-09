import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("=" * 60)
print("TP4 - EXERCICE 6 - Évaluation des performances")
print("=" * 60)

accuracy = accuracy_score(y_test, y_pred)

print("\n1. Accuracy du modèle:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   → Le modèle prédit correctement {accuracy*100:.2f}% des cas")

cm = confusion_matrix(y_test, y_pred)

print("\n2. Matrice de confusion:")
print("\n   " + " " * 20 + "Prédit")
print("   " + " " * 15 + f"{cancer.target_names[0]:<15} {cancer.target_names[1]:<15}")
print("   " + "-" * 45)
print(f"   Réel {cancer.target_names[0]:<10} {cm[0][0]:<15} {cm[0][1]:<15}")
print(f"        {cancer.target_names[1]:<10} {cm[1][0]:<15} {cm[1][1]:<15}")

print("\n   Légende:")
print(f"   - Vrais positifs (TP): {cm[0][0]} - malignant correctement prédit")
print(f"   - Faux négatifs (FN): {cm[0][1]} - malignant prédit comme benign")
print(f"   - Faux positifs (FP): {cm[1][0]} - benign prédit comme malignant")
print(f"   - Vrais négatifs (TN): {cm[1][1]} - benign correctement prédit")

precision = precision_score(y_test, y_pred, pos_label=0)  # Pour la classe malignant
recall = recall_score(y_test, y_pred, pos_label=0)  # Pour la classe malignant
f1 = f1_score(y_test, y_pred, pos_label=0)  # Pour la classe malignant

precision_benign = precision_score(y_test, y_pred, pos_label=1)
recall_benign = recall_score(y_test, y_pred, pos_label=1)
f1_benign = f1_score(y_test, y_pred, pos_label=1)

print("\n3. Métriques détaillées:")
print("\n   Classe 'malignant' (classe 0):")
print(f"   Précision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Rappel: {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-score: {f1:.4f} ({f1*100:.2f}%)")

print("\n   Classe 'benign' (classe 1):")
print(f"   Précision: {precision_benign:.4f} ({precision_benign*100:.2f}%)")
print(f"   Rappel: {recall_benign:.4f} ({recall_benign*100:.2f}%)")
print(f"   F1-score: {f1_benign:.4f} ({f1_benign*100:.2f}%)")

precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print("\n   Moyennes (macro):")
print(f"   Précision moyenne: {precision_macro:.4f}")
print(f"   Rappel moyen: {recall_macro:.4f}")
print(f"   F1-score moyen: {f1_macro:.4f}")

print("\n" + "=" * 60)
print("Rapport de classification complet:")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

