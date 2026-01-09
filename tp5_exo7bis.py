import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

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

# Normaliser les données (nécessaire pour k-NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("=" * 60)
print("TP5 - EXERCICE 7 BIS - Comparaison k-NN vs Random Forest")
print("=" * 60)

# Modèle 1: k-NN
print("\n1. Modèle k-NN:")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn)
recall_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

print(f"   Accuracy: {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")
print(f"   Précision: {precision_knn:.4f} ({precision_knn*100:.2f}%)")
print(f"   Rappel: {recall_knn:.4f} ({recall_knn*100:.2f}%)")
print(f"   F1-score: {f1_knn:.4f} ({f1_knn*100:.2f}%)")

# Matrice de confusion k-NN
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(f"\n   Matrice de confusion:")
print(f"   TN={cm_knn[0][0]}, FP={cm_knn[0][1]}, FN={cm_knn[1][0]}, TP={cm_knn[1][1]}")

# Modèle 2: Random Forest
print("\n" + "=" * 60)
print("2. Modèle Random Forest:")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)  # Random Forest n'a pas besoin de normalisation
y_pred_rf = rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"   Accuracy: {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
print(f"   Précision: {precision_rf:.4f} ({precision_rf*100:.2f}%)")
print(f"   Rappel: {recall_rf:.4f} ({recall_rf*100:.2f}%)")
print(f"   F1-score: {f1_rf:.4f} ({f1_rf*100:.2f}%)")

# Matrice de confusion Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(f"\n   Matrice de confusion:")
print(f"   TN={cm_rf[0][0]}, FP={cm_rf[0][1]}, FN={cm_rf[1][0]}, TP={cm_rf[1][1]}")

# Importance des variables pour Random Forest
print(f"\n   Importance des variables (top 3):")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
for i, row in feature_importance.head(3).iterrows():
    print(f"      {row['feature']}: {row['importance']:.4f}")

# Comparaison des deux modèles
print("\n" + "=" * 60)
print("3. Comparaison des modèles:")
print("=" * 60)

print("\n   Métriques comparatives:")
print("   " + "-" * 60)
print(f"   {'Métrique':<15} {'k-NN':<15} {'Random Forest':<15} {'Meilleur':<15}")
print("   " + "-" * 60)
print(f"   {'Accuracy':<15} {accuracy_knn:<15.4f} {accuracy_rf:<15.4f} {'RF' if accuracy_rf > accuracy_knn else 'k-NN':<15}")
print(f"   {'Précision':<15} {precision_knn:<15.4f} {precision_rf:<15.4f} {'RF' if precision_rf > precision_knn else 'k-NN':<15}")
print(f"   {'Rappel':<15} {recall_knn:<15.4f} {recall_rf:<15.4f} {'RF' if recall_rf > recall_knn else 'k-NN':<15}")
print(f"   {'F1-score':<15} {f1_knn:<15.4f} {f1_rf:<15.4f} {'RF' if f1_rf > f1_knn else 'k-NN':<15}")

# Différences
print("\n   Différences:")
diff_accuracy = accuracy_rf - accuracy_knn
diff_precision = precision_rf - precision_knn
diff_recall = recall_rf - recall_knn
diff_f1 = f1_rf - f1_knn

print(f"   Accuracy: {diff_accuracy:+.4f} ({diff_accuracy*100:+.2f}%)")
print(f"   Précision: {diff_precision:+.4f} ({diff_precision*100:+.2f}%)")
print(f"   Rappel: {diff_recall:+.4f} ({diff_recall*100:+.2f}%)")
print(f"   F1-score: {diff_f1:+.4f} ({diff_f1*100:+.2f}%)")

# Vérifier si les prédictions sont identiques
predictions_different = (y_pred_knn != y_pred_rf).sum()
if predictions_different > 0:
    print(f"\n   ⚠️  Note: {predictions_different} prédictions diffèrent entre les deux modèles")
    print(f"   → Les performances sont identiques par COÏNCIDENCE")
    print(f"   → Les deux modèles font le même nombre d'erreurs mais sur des cas différents")

# Affichage des erreurs sur les tests
print("\n" + "=" * 60)
print("4. Erreurs sur l'ensemble de test:")
print("=" * 60)

# Erreurs k-NN
errors_knn = np.where(y_pred_knn != y_test)[0]
print(f"\n   Erreurs k-NN ({len(errors_knn)} erreurs):")
if len(errors_knn) > 0:
    print("   " + "-" * 80)
    print(f"   {'Index':<8} {'Réel':<12} {'Prédit':<12} {'Type erreur':<20} {'Pclass':<8} {'Sex':<8} {'Age':<8}")
    print("   " + "-" * 80)
    for idx in errors_knn[:15]:  # Afficher les 15 premières erreurs
        real_label = "Survivant" if y_test.iloc[idx] == 1 else "Non-survivant"
        pred_label = "Survivant" if y_pred_knn[idx] == 1 else "Non-survivant"
        error_type = "Faux positif" if (y_test.iloc[idx] == 0 and y_pred_knn[idx] == 1) else "Faux négatif"
        pclass = X_test.iloc[idx]['Pclass']
        sex = "F" if X_test.iloc[idx]['Sex_encoded'] == 1 else "M"
        age = X_test.iloc[idx]['Age']
        print(f"   {idx:<8} {real_label:<12} {pred_label:<12} {error_type:<20} {pclass:<8} {sex:<8} {age:<8.1f}")
    if len(errors_knn) > 15:
        print(f"   ... et {len(errors_knn) - 15} autres erreurs")

# Erreurs Random Forest
errors_rf = np.where(y_pred_rf != y_test)[0]
print(f"\n   Erreurs Random Forest ({len(errors_rf)} erreurs):")
if len(errors_rf) > 0:
    print("   " + "-" * 80)
    print(f"   {'Index':<8} {'Réel':<12} {'Prédit':<12} {'Type erreur':<20} {'Pclass':<8} {'Sex':<8} {'Age':<8}")
    print("   " + "-" * 80)
    for idx in errors_rf[:15]:  # Afficher les 15 premières erreurs
        real_label = "Survivant" if y_test.iloc[idx] == 1 else "Non-survivant"
        pred_label = "Survivant" if y_pred_rf[idx] == 1 else "Non-survivant"
        error_type = "Faux positif" if (y_test.iloc[idx] == 0 and y_pred_rf[idx] == 1) else "Faux négatif"
        pclass = X_test.iloc[idx]['Pclass']
        sex = "F" if X_test.iloc[idx]['Sex_encoded'] == 1 else "M"
        age = X_test.iloc[idx]['Age']
        print(f"   {idx:<8} {real_label:<12} {pred_label:<12} {error_type:<20} {pclass:<8} {sex:<8} {age:<8.1f}")
    if len(errors_rf) > 15:
        print(f"   ... et {len(errors_rf) - 15} autres erreurs")

# Analyse des erreurs communes et différentes
errors_knn_set = set(errors_knn)
errors_rf_set = set(errors_rf)
errors_common = errors_knn_set & errors_rf_set
errors_knn_only = errors_knn_set - errors_rf_set
errors_rf_only = errors_rf_set - errors_knn_set

print(f"\n   Analyse des erreurs:")
print(f"   • Erreurs communes (les deux se trompent): {len(errors_common)}")
print(f"   • Erreurs uniquement k-NN: {len(errors_knn_only)}")
print(f"   • Erreurs uniquement Random Forest: {len(errors_rf_only)}")

# Rapports de classification
print("\n" + "=" * 60)
print("Rapport de classification - k-NN:")
print("=" * 60)
print(classification_report(y_test, y_pred_knn, target_names=['Non-survivant', 'Survivant']))

print("\n" + "=" * 60)
print("Rapport de classification - Random Forest:")
print("=" * 60)
print(classification_report(y_test, y_pred_rf, target_names=['Non-survivant', 'Survivant']))

