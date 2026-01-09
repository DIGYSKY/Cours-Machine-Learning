import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

print("=" * 60)
print("TP5 - EXERCICE 2 - Analyse exploratoire")
print("=" * 60)

# 1. Étudier la répartition des survivants et des non-survivants
print("\n1. Répartition des survivants et des non-survivants:")
print("=" * 60)

survival_counts = df['Survived'].value_counts().sort_index()
total = len(df)

print(f"\n   Total de passagers: {total}")
for val in sorted(df['Survived'].unique()):
    count = survival_counts[val]
    percentage = (count / total) * 100
    label = "Survivants" if val == 1 else "Non-survivants"
    print(f"   {label}: {count} ({percentage:.2f}%)")

# 2. Analyser l'influence de certaines variables sur la survie
print("\n" + "=" * 60)
print("2. Influence des variables sur la survie:")
print("=" * 60)

# 2.1 Influence du sexe
print("\n   2.1 Influence du SEXE sur la survie:")
sex_survival = pd.crosstab(df['Sex'], df['Survived'], margins=True)
print("\n   Tableau de contingence:")
print(sex_survival)

print("\n   Taux de survie par sexe:")
for sex in df['Sex'].unique():
    sex_data = df[df['Sex'] == sex]
    survival_rate = (sex_data['Survived'] == 1).sum() / len(sex_data) * 100
    print(f"      {sex}: {survival_rate:.2f}% ({sex_data['Survived'].sum()}/{len(sex_data)})")

# 2.2 Influence de la classe
print("\n   2.2 Influence de la CLASSE (Pclass) sur la survie:")
class_survival = pd.crosstab(df['Pclass'], df['Survived'], margins=True)
print("\n   Tableau de contingence:")
print(class_survival)

print("\n   Taux de survie par classe:")
for pclass in sorted(df['Pclass'].unique()):
    class_data = df[df['Pclass'] == pclass]
    survival_rate = (class_data['Survived'] == 1).sum() / len(class_data) * 100
    print(f"      Classe {pclass}: {survival_rate:.2f}% ({class_data['Survived'].sum()}/{len(class_data)})")

# 2.3 Influence de l'âge
print("\n   2.3 Influence de l'ÂGE sur la survie:")
print("\n   Statistiques d'âge par groupe de survie:")
age_stats = df.groupby('Survived')['Age'].agg(['count', 'mean', 'median', 'std'])
print(age_stats)

# Analyser par groupes d'âge
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                        labels=['Enfant (0-12)', 'Adolescent (13-18)', 
                                'Jeune adulte (19-35)', 'Adulte (36-60)', 'Senior (60+)'])

print("\n   Taux de survie par groupe d'âge:")
for age_group in df['AgeGroup'].cat.categories:
    group_data = df[df['AgeGroup'] == age_group]
    if len(group_data) > 0:
        survival_rate = (group_data['Survived'] == 1).sum() / len(group_data) * 100
        print(f"      {age_group}: {survival_rate:.2f}% ({group_data['Survived'].sum()}/{len(group_data)})")

# 3. Formuler des hypothèses à partir des observations
print("\n" + "=" * 60)
print("3. Hypothèses formulées à partir des observations:")
print("=" * 60)

# Calculer les taux pour les hypothèses
female_survival = (df[df['Sex'] == 'female']['Survived'] == 1).sum() / len(df[df['Sex'] == 'female']) * 100
male_survival = (df[df['Sex'] == 'male']['Survived'] == 1).sum() / len(df[df['Sex'] == 'male']) * 100

class1_survival = (df[df['Pclass'] == 1]['Survived'] == 1).sum() / len(df[df['Pclass'] == 1]) * 100
class3_survival = (df[df['Pclass'] == 3]['Survived'] == 1).sum() / len(df[df['Pclass'] == 3]) * 100

print("\n   Statistiques pour formulation d'hypothèses:")
print(f"   → Femmes: {female_survival:.2f}% vs Hommes: {male_survival:.2f}%")
print(f"   → Classe 1: {class1_survival:.2f}% vs Classe 3: {class3_survival:.2f}%")
if len(df[df['AgeGroup'] == 'Enfant (0-12)']) > 0:
    child_survival = (df[df['AgeGroup'] == 'Enfant (0-12)']['Survived'] == 1).sum() / len(df[df['AgeGroup'] == 'Enfant (0-12)']) * 100
    print(f"   → Enfants: {child_survival:.2f}%")

# Vérification de l'hypothèse 4
print("\n   Vérification de l'hypothèse 4:")
female_class1 = df[(df['Sex'] == 'female') & (df['Pclass'] == 1)]
male_class3 = df[(df['Sex'] == 'male') & (df['Pclass'] == 3)]

if len(female_class1) > 0:
    f1_survival = (female_class1['Survived'] == 1).sum() / len(female_class1) * 100
    print(f"   → Femmes classe 1: {f1_survival:.2f}% ({female_class1['Survived'].sum()}/{len(female_class1)})")

if len(male_class3) > 0:
    m3_survival = (male_class3['Survived'] == 1).sum() / len(male_class3) * 100
    print(f"   → Hommes classe 3: {m3_survival:.2f}% ({male_class3['Survived'].sum()}/{len(male_class3)})")

