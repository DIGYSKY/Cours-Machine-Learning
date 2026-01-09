import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder

#utiliser le dataset Iris (inclus dans Scikit-learn)
iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print (df.info(), "\n") 
print (df.head(), "\n")

# La target représente la classe de l'iris (0=setosa, 1=versicolor, 2=virginica)
print("Noms des classes:")
print(iris.target_names)
print("\nDistribution des classes (target / repartition):")
print(df['target'].value_counts().sort_index())
