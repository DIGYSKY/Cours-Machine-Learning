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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("=" * 60)
print("TP4 - EXERCICE 5 - Prédiction")
print("=" * 60)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

print("\n1. Prédictions effectuées sur l'ensemble de test")
print(f"   Nombre de prédictions: {len(y_pred)}")

print("\n   Exemples de prédictions (10 premières):")
print("   " + "-" * 80)
print(f"   {'Index':<6} {'Classe réelle':<15} {'Classe prédite':<15} {'Probabilité classe 0':<20} {'Probabilité classe 1':<20}")
print("   " + "-" * 80)
for i in range(min(10, len(y_test))):
    real_class = cancer.target_names[y_test[i]]
    pred_class = cancer.target_names[y_pred[i]]
    proba_0 = y_pred_proba[i][0]
    proba_1 = y_pred_proba[i][1]
    print(f"   {i:<6} {real_class:<15} {pred_class:<15} {proba_0:<20.4f} {proba_1:<20.4f}")

print("\n   Statistiques globales:")
print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred)*100:.2f}%)")

print("\n   Distribution des prédictions:")
pred_counts = pd.Series(y_pred).value_counts().sort_index()
for idx, count in pred_counts.items():
    class_name = cancer.target_names[idx]
    percentage = (count / len(y_pred)) * 100
    print(f"      Classe {idx} ({class_name}): {count} prédictions ({percentage:.1f}%)")

print("\n" + "=" * 60)
print("2. Signification des prédictions:")
print("=" * 60)
print(f"\n   Prédiction = 0 ({cancer.target_names[0]}):")
print(f"   → La tumeur est classée comme MALIGNE (cancéreuse)")
print(f"   → Action recommandée: traitement médical urgent nécessaire")

print(f"\n   Prédiction = 1 ({cancer.target_names[1]}):")
print(f"   → La tumeur est classée comme BÉNIGNE (non cancéreuse)")
print(f"   → Action recommandée: surveillance médicale régulière")

print("\n" + "=" * 60)
print("Matrice de confusion:")
print("=" * 60)
cm = confusion_matrix(y_test, y_pred)
print(f"\n   {'Réel/Pridit':<15} {cancer.target_names[0]:<15} {cancer.target_names[1]:<15}")
print("   " + "-" * 45)
print(f"   {cancer.target_names[0]:<15} {cm[0][0]:<15} {cm[0][1]:<15}")
print(f"   {cancer.target_names[1]:<15} {cm[1][0]:<15} {cm[1][1]:<15}")

print("\n   Interprétation:")
print(f"   - Vrais positifs (malignant prédit correctement): {cm[0][0]}")
print(f"   - Faux négatifs (malignant prédit comme benign): {cm[0][1]} ⚠️")
print(f"   - Faux positifs (benign prédit comme malignant): {cm[1][0]} ⚠️")
print(f"   - Vrais négatifs (benign prédit correctement): {cm[1][1]}")

