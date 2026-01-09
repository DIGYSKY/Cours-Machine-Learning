from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

digits = load_digits()

x = digits.images
y = digits.target

# Préparation pour afficher les deux images côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Première image : image originale
axes[0].imshow(x[0], cmap="gray")
axes[0].set_title(f"Digit original: {y[0]}")
axes[0].axis('off')

print(x[0])
X_norm = x / 16.0

x_samples = len(X_norm)
X_flat = X_norm.reshape(x_samples, -1)

print(X_flat.shape)

X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("accuracy: ", model.score(X_test, y_test))

# Deuxième image : prédiction du modèle
axes[1].imshow(X_test[0].reshape(8, 8), cmap="gray")
axes[1].set_title(f"Digit prédit: {y_pred[0]}")
axes[1].axis('off')

plt.tight_layout()
plt.show()