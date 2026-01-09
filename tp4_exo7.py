import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
y_pred_proba = model.predict_proba(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)

print("=" * 60)
print("TP4 - EXERCICE 7 - Analyse des erreurs")
print("=" * 60)

print("\n1. Identification des erreurs:")

false_negatives = np.where((y_test == 0) & (y_pred == 1))[0]
print(f"\n   Faux négatifs (malignant prédit comme benign): {len(false_negatives)}")
print(f"   → Ces erreurs sont TRÈS GRAVES : retardent le diagnostic du cancer")
if len(false_negatives) > 0:
    print(f"\n   Exemples de faux négatifs:")
    for i, idx in enumerate(false_negatives[:3]):
        proba_malignant = y_pred_proba[idx][0]
        proba_benign = y_pred_proba[idx][1]
        print(f"      Index {idx}: probabilité malignant={proba_malignant:.4f}, benign={proba_benign:.4f}")

false_positives = np.where((y_test == 1) & (y_pred == 0))[0]
print(f"\n   Faux positifs (benign prédit comme malignant): {len(false_positives)}")
print(f"   → Ces erreurs causent de l'anxiété mais permettent des examens supplémentaires")
if len(false_positives) > 0:
    print(f"\n   Exemples de faux positifs:")
    for i, idx in enumerate(false_positives[:3]):
        proba_malignant = y_pred_proba[idx][0]
        proba_benign = y_pred_proba[idx][1]
        print(f"      Index {idx}: probabilité malignant={proba_malignant:.4f}, benign={proba_benign:.4f}")

print("\n" + "=" * 60)
print("Statistiques des erreurs:")
print("=" * 60)
print(f"   Total d'erreurs: {len(false_negatives) + len(false_positives)}")
print(f"   Faux négatifs: {len(false_negatives)} ({len(false_negatives)/(len(false_negatives)+len(false_positives))*100:.1f}% des erreurs)")
print(f"   Faux positifs: {len(false_positives)} ({len(false_positives)/(len(false_negatives)+len(false_positives))*100:.1f}% des erreurs)")

print("\n" + "=" * 60)
print("2. Gravité des erreurs dans le contexte médical:")
print("=" * 60)
print("\n   Faux négatifs (malignant → benign) sont PLUS GRAVES car:")
print("   • Retardent le diagnostic du cancer")
print("   • Empêchent un traitement précoce")
print("   • Peuvent mettre la vie du patient en danger")
print("   • Coût humain très élevé")

print("\n   Faux positifs (benign → malignant) sont MOINS GRAVES car:")
print("   • Causent de l'anxiété mais permettent des examens supplémentaires")
print("   • Mieux vaut 'trop de précaution' que manquer un cancer")
print("   • Les examens complémentaires confirmeront le diagnostic")
print("   • Coût principalement émotionnel et financier")

print("\n" + "=" * 60)
print("3. Métrique plus pertinente que l'accuracy:")
print("=" * 60)

recall_malignant = recall_score(y_test, y_pred, pos_label=0)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   Accuracy actuelle: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Rappel (Recall) pour 'malignant': {recall_malignant:.4f} ({recall_malignant*100:.2f}%)")

print("\n   → Le RAPPEL (Recall/Sensitivity) est plus pertinent car:")
print("   • Mesure la capacité à détecter TOUS les cas de cancer")
print("   • Formule: TP / (TP + FN)")
print("   • Priorise la détection des vrais cas de cancer")
print("   • Dans le contexte médical, mieux vaut détecter tous les cancers")
print("   • Même si cela génère quelques faux positifs")

print("\n   Autres métriques pertinentes:")
print("   • F1-score: équilibre entre précision et rappel")
print("   • Spécificité: capacité à identifier correctement les cas bénins")
print("   • AUC-ROC: performance globale du modèle à différents seuils")

specificity = cm[1][1] / (cm[1][0] + cm[1][1])
print(f"\n   Spécificité (pour benign): {specificity:.4f} ({specificity*100:.2f}%)")

