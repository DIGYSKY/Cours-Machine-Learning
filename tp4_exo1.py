import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

print("=" * 60)
print("TP4 - EXERCICE 1 - Découverte du dataset Breast Cancer")
print("=" * 60)

print("\n1. Dataset chargé avec succès!")

print("\n2. Informations sur le dataset:")
print(f"   Nombre d'échantillons: {cancer.data.shape[0]}")
print(f"   Nombre de variables explicatives: {cancer.data.shape[1]}")

print("\n3. Variable cible et classes:")
print(f"   Variable cible: 'target'")
print(f"   Nombre de classes: {len(cancer.target_names)}")
print(f"   Noms des classes:")
for i, class_name in enumerate(cancer.target_names):
    print(f"      {i}: {class_name}")

print("\n   Distribution des classes:")
class_counts = pd.Series(cancer.target).value_counts().sort_index()
for idx, count in class_counts.items():
    class_name = cancer.target_names[idx]
    percentage = (count / len(cancer.target)) * 100
    print(f"      Classe {idx} ({class_name}): {count} échantillons ({percentage:.1f}%)")

print("\n" + "=" * 60)
print("Aperçu des données:")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("Informations sur le dataset:")
print("=" * 60)
print(df.info())

