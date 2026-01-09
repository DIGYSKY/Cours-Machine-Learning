import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

texts = [
    "Service rapide et efficace",             # positif
    "Très satisfait de mon achat",            # positif
    "Bonne expérience, je recommande",        # positif
    "Support client à l'écoute",              # positif
    "Livraison dans les délais",              # positif
    "Produit conforme à la description",      # positif
    "Merci pour votre professionnalisme",     # positif
    "Site facile à utiliser",                 # positif
    "Je reviendrai sans hésiter",             # positif
    "Bonne qualité et prix attractif",        # positif

    "Livraison en retard",                    # négatif
    "Produit arrivé endommagé",               # négatif
    "Service client difficile à joindre",     # négatif
    "Commande jamais reçue",                  # négatif
    "Produit non conforme à la photo",        # négatif
    "Déçu par la qualité du produit",         # négatif
    "Site peu clair et compliqué",            # négatif
    "Achat impossible à finaliser",           # négatif
    "Impossible d’obtenir un remboursement",  # négatif
    "Réponse tardive du service client",      # négatif
]

labels = [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]

positive_words = ["rapide", "efficace", "satisfait", "recommande", "professionnel", "facile", "reviendrai", "attractif"]
negative_words = ["retard", "endommage", "difficile", "jamais", "non", "déçu", "compliqué", "impossible"]

def extract_features(text):
    text = text.lower()

    nb_words = len(text.split())
    nb_positive_words = sum(1 for word in text.split() if word in positive_words)
    nb_negative_words = sum(1 for word in text.split() if word in negative_words)

    return [nb_words, nb_positive_words, nb_negative_words]

X = np.array([extract_features(text) for text in texts])
y = np.array(labels)

print("Matrice des features:")
print(X)
print("Labels:")
print(y)

print("=" * 60)
print("Entraînement du modèle")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

new_text = "Le service est rapide et efficace"
new_features = extract_features(new_text)
new_features = np.array([new_features])
prediction = model.predict(new_features)
print(f"Prédiction pour le texte '{new_text}': {'positif' if prediction[0] == 1 else 'négatif'}")