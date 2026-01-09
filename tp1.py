from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
import re

from data import data as dataset
from new_mail import new_mail

df = pd.DataFrame(dataset)

# Fonction de preprocessing améliorée
def preprocess_text(text):
    # Normaliser les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    # Garder les caractères spéciaux importants pour le spam (!, ?, $, etc.)
    # Normaliser les multiples caractères spéciaux (!!! -> !)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    return text.strip()

# Appliquer le preprocessing
df['text'] = df['text'].apply(preprocess_text)

# Amélioration du vectorizer pour supporter anglais et français
# Test de différentes configurations pour trouver la meilleure
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,  # Pas de stop_words car on a du français et anglais
    ngram_range=(1, 2),  # Unigrammes et bigrammes (trigrammes peuvent être trop spécifiques)
    max_features=2500,  # Réduire pour éviter le sur-apprentissage
    min_df=2,  # Ignorer les mots qui apparaissent moins de 2 fois (réduit le bruit)
    max_df=0.88,  # Ignorer les mots trop communs
    sublinear_tf=True,  # Utiliser log pour réduire l'impact des fréquences élevées
    analyzer='word',
    token_pattern=r'(?u)\b\w+\b',  # Pattern pour capturer les mots (incluant accents)
    norm='l2'  # Normalisation L2 pour améliorer les performances
)

X_Vect = vectorizer.fit_transform(df["text"])
y = df["label"]

# Utiliser stratify pour maintenir la distribution des classes
X_train, X_test, y_train, y_test = train_test_split(
    X_Vect, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)


# Optimisation des hyperparamètres avec GridSearchCV pour MultinomialNB
print("Optimisation des hyperparamètres en cours...")
# Augmenter alpha pour plus de régularisation et réduire l'overfitting
param_grid_nb = {'alpha': [0.5, 1.0, 2.0, 5.0, 10.0]}
grid_search_nb = GridSearchCV(
    MultinomialNB(), 
    param_grid_nb, 
    cv=5, 
    scoring='f1',  # Utiliser F1 score pour équilibrer precision et recall
    n_jobs=-1
)
grid_search_nb.fit(X_train, y_train)
model_nb = grid_search_nb.best_estimator_

print(f"Meilleur alpha pour MultinomialNB: {grid_search_nb.best_params_['alpha']}")

y_pred = model_nb.predict(X_test)

# Trouver le meilleur seuil pour réduire les faux positifs
from sklearn.metrics import precision_recall_curve
y_proba = model_nb.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
# Trouver le seuil qui maximise F1 score
f1_scores = 2 * (precision * recall) / (precision + recall)
f1_scores = np.nan_to_num(f1_scores)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
print(f"Meilleur seuil de probabilité pour MultinomialNB: {best_threshold:.3f}")

print("--------------------------------")
print("MultinomialNB (Optimisé)")
print("--------------------------------")

# Calculer les scores sur training et test
train_pred = model_nb.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Écart (overfitting): {train_acc - test_acc:.4f} ({abs(train_acc - test_acc)*100:.2f}%)")
print("\nClassification Report (Test):\n", classification_report(y_test, y_pred))
print("Confusion Matrix (Test):\n", confusion_matrix(y_test, y_pred))

def testemail(email, model, threshold=0.5):
    """
    Prédit si un email est spam ou non.
    Utilise un seuil de probabilité pour réduire les faux positifs.
    """
    new_mail = [email]
    new_mail_vect = vectorizer.transform(new_mail)
    
    # Utiliser les probabilités si disponibles
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(new_mail_vect)[0]
        # Si la probabilité de spam est supérieure au seuil, c'est du spam
        # Ajuster le seuil pour réduire les faux positifs (seuil plus élevé = moins de spam détecté)
        prediction = 1 if proba[1] >= threshold else 0
    else:
        prediction = model.predict(new_mail_vect)[0]
    
    return prediction

def test_emails_and_show_errors(model, model_name, threshold=0.55):
    """
    Teste les emails et affiche les erreurs pour un modèle donné.
    """
    total_emails = 0
    score = 0
    errors = []
    
    for i, email in enumerate(new_mail["emails"]):
        isSpam = testemail(email, model, threshold=threshold)
        total_emails += 1
        correct = isSpam == new_mail['isSpam'][i]
        score += 1 if correct else 0
        if not correct:
            errors.append({
                'email': email,
                'predicted': isSpam,
                'actual': new_mail['isSpam'][i],
                'type': 'Faux positif' if isSpam == 1 and new_mail['isSpam'][i] == 0 else 'Faux négatif'
            })
    
    print(f"Total Emails: {total_emails}")
    print(f"Score: {score}")
    print(f"Accuracy: {score / total_emails:.2f}")
    if errors:
        print(f"\nErreurs ({len(errors)}):")
        for err in errors[:10]:  # Afficher les 10 premières erreurs
            print(f"  [{err['type']}] '{err['email'][:60]}...' -> Prédit: {err['predicted']}, Réel: {err['actual']}")
    else:
        print("\nAucune erreur !")
    return errors

# Test sur les nouveaux emails avec analyse des erreurs
print("\nAnalyse des erreurs sur les nouveaux emails:")
threshold_nb = max(best_threshold, 0.55)  # Seuil minimum de 0.55 pour réduire les faux positifs
errors_nb = test_emails_and_show_errors(model_nb, "MultinomialNB", threshold=threshold_nb)
print("--------------------------------")

# Optimisation de Logistic Regression
from sklearn.linear_model import LogisticRegression

param_grid_lr = {'C': [0.1, 1.0, 10.0], 'solver': ['lbfgs', 'liblinear']}
grid_search_lr = GridSearchCV(
    LogisticRegression(max_iter=1000), 
    param_grid_lr, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1
)
grid_search_lr.fit(X_train, y_train)
model_lr = grid_search_lr.best_estimator_

print(f"Meilleurs paramètres pour Logistic Regression: {grid_search_lr.best_params_}")

y_pred = model_lr.predict(X_test)

print("--------------------------------")
print("Logistic Regression (Optimisé)")
print("--------------------------------")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Test sur les nouveaux emails avec analyse des erreurs
print("\nAnalyse des erreurs sur les nouveaux emails:")
errors_lr = test_emails_and_show_errors(model_lr, "Logistic Regression", threshold=0.55)
print("--------------------------------")

# Optimisation de SVM
from sklearn.svm import SVC

param_grid_svm = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
grid_search_svm = GridSearchCV(
    SVC(probability=True), 
    param_grid_svm, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1
)
grid_search_svm.fit(X_train, y_train)
model_svm = grid_search_svm.best_estimator_

print(f"Meilleurs paramètres pour SVM: {grid_search_svm.best_params_}")

y_pred = model_svm.predict(X_test)

print("--------------------------------")
print("SVM (Optimisé)")
print("--------------------------------")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Test sur les nouveaux emails avec analyse des erreurs
print("\nAnalyse des erreurs sur les nouveaux emails:")
errors_svm = test_emails_and_show_errors(model_svm, "SVM", threshold=0.55)
print("--------------------------------")

# Ensemble Voting Classifier pour combiner les meilleurs modèles
print("\n--------------------------------")
print("Voting Classifier (Ensemble)")
print("--------------------------------")

voting_clf = VotingClassifier(
    estimators=[
        ('nb', model_nb),
        ('lr', model_lr),
        ('svm', model_svm)
    ],
    voting='soft'  # Utiliser les probabilités pour un meilleur résultat
)
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_voting))
print("Classification Report:\n", classification_report(y_test, y_pred_voting))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_voting))

# Test sur les nouveaux emails avec analyse des erreurs
print("\nAnalyse des erreurs sur les nouveaux emails:")
errors_voting = test_emails_and_show_errors(voting_clf, "Voting Classifier", threshold=0.55)
print("--------------------------------")