import pandas as pd
import numpy as np

print("=" * 60)
print("TP5 - EXERCICE 1 - Chargement et exploration des données")
print("=" * 60)

df = pd.read_csv('train.csv')

print("\n1. Fichier train.csv chargé avec succès!")

print("\n2. Premières lignes du dataset:")
print(df.head(10))

print("\n3. Variables du dataset:")
print(f"   Colonnes: {list(df.columns)}")

# Variable cible
target = 'Survived'
print(f"\n   Variable cible: '{target}'")
print(f"   Valeurs possibles: {sorted(df[target].unique())}")
print(f"   Distribution:")
for val in sorted(df[target].unique()):
    count = (df[target] == val).sum()
    percentage = (count / len(df)) * 100
    label = "survivant" if val == 1 else "non survivant"
    print(f"      {val} ({label}): {count} ({percentage:.1f}%)")

# Variables explicatives
explanatory_vars = [col for col in df.columns if col != target]
print(f"\n   Variables explicatives ({len(explanatory_vars)}):")
for i, var in enumerate(explanatory_vars, 1):
    print(f"      {i}. {var}")

# 4. Donner le nombre d'observations et de variables
print("\n4. Statistiques du dataset:")
print(f"   Nombre d'observations (lignes): {len(df)}")
print(f"   Nombre de variables (colonnes): {len(df.columns)}")
print(f"   Nombre de variables explicatives: {len(explanatory_vars)}")
print(f"   Nombre de variables cibles: 1")

print("\n5. Types de variables:")

# Variables numériques
numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
if target in numeric_vars:
    numeric_vars.remove(target)

print(f"\n   Variables numériques ({len(numeric_vars)}):")
for var in numeric_vars:
    missing = df[var].isna().sum()
    missing_pct = (missing / len(df)) * 100
    print(f"      - {var}: type {df[var].dtype}, valeurs manquantes: {missing} ({missing_pct:.1f}%)")

# Variables catégorielles
categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n   Variables catégorielles ({len(categorical_vars)}):")
for var in categorical_vars:
    missing = df[var].isna().sum()
    missing_pct = (missing / len(df)) * 100
    unique_count = df[var].nunique()
    print(f"      - {var}: type {df[var].dtype}, valeurs uniques: {unique_count}, valeurs manquantes: {missing} ({missing_pct:.1f}%)")
    if unique_count <= 10:  # Afficher les valeurs si peu nombreuses
        print(f"        Valeurs: {list(df[var].unique())}")

# Résumé des valeurs manquantes
print("\n" + "=" * 60)
print("Résumé des valeurs manquantes:")
print("=" * 60)
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0]
if len(missing_summary) > 0:
    print("\n   Variables avec valeurs manquantes:")
    for var, count in missing_summary.items():
        pct = (count / len(df)) * 100
        print(f"      {var}: {count} ({pct:.1f}%)")
else:
    print("\n   Aucune valeur manquante détectée.")

# Informations générales
print("\n" + "=" * 60)
print("Informations générales:")
print("=" * 60)
print(df.info())

