# Importer les bibliothèques nécessaires
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

# Charger les données dans un dataframe
df = pd.read_csv('/content/drive/My Drive/KaDo.csv')

# Afficher le nombre de lignes et de colonnes avant la suppression
print("Nombre de lignes et de colonnes avant la suppression:", df.shape)

# Supprimer les lignes avec des champs manquants
df = df.dropna()

# Afficher le nombre de lignes et de colonnes après la suppression
print("Nombre de lignes et de colonnes après la suppression:", df.shape)

# Obtenir les dimensions du dataframe avant suppression
initial_rows = df.shape[0]

# Supprimer les lignes dupliquées
df = df.drop_duplicates()

# Obtenir les dimensions du dataframe après suppression
final_rows = df.shape[0]

# Calculer le nombre de lignes supprimées
deleted_rows = initial_rows - final_rows

# Afficher le nombre de lignes supprimées
print("Nombre de lignes dupliquées supprimées : ", deleted_rows)


# Afficher les tendances de vente par mois
print("tendances de vente par mois : ", df.groupby('MOIS_VENTE')['PRIX_NET'].sum())

# Afficher les produits les plus populaires
print("les produits les plus populaires : ", df.groupby('LIBELLE')['PRIX_NET'].count())


# Agruper les données par famille de produits et calculer les ventes totales pour chaque groupe
sales_by_family = df.groupby("FAMILLE").sum()["PRIX_NET"]
# Afficher les ventes totales par famille de produits
print("les ventes totales par famille de produits : ", sales_by_family)

# Créer le graphique à barres
sales_by_family.plot(kind="bar", xlabel="Famille de produits", ylabel="Ventes totales")

# Afficher le graphique
plt.show()


# Sélectionner les univers uniques
univers = df['UNIVERS'].unique()

# Initialiser un dataframe vide pour stocker les résultats
result = pd.DataFrame(columns=['UNIVERS', 'LIBELLE', 'MOIS_VENTE', '%_VENTE'])

# Pour chaque univers
for u in univers:
    # Sélectionner les lignes qui appartiennent à cet univers
    df_univers = df[df['UNIVERS'] == u]
    # Compter le nombre de fois où chaque libellé apparaît
    count = df_univers.groupby('LIBELLE')['PRIX_NET'].count()
    # Trier les libellés par ordre décroissant de fréquence
    count = count.sort_values(ascending=False)
    # Sélectionner le libellé le plus fréquent
    top_libelle = count.index[0]
    mois_vente = df_univers[df_univers['LIBELLE'] == top_libelle].groupby('MOIS_VENTE')['PRIX_NET'].count()
    mois_vente = mois_vente.sort_values(ascending=False).index[0]
    # Calculer le pourcentage de vente
    pourcentage = (count[0] / df_univers.shape[0]) * 100
    # Ajouter les résultats au dataframe
    result = result.append({'UNIVERS': u, 'LIBELLE': top_libelle, 'MOIS_VENTE': mois_vente, '%_VENTE': pourcentage}, ignore_index=True)

# Afficher le tableau
print(result)

df_reduit = df.head(10000)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

# Chargement des données
# Récupération des libellés de produits
product_labels = df_reduit['MAILLE'].values
# Pré-traitement des données
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('french'))
tokens = []
for label in product_labels:
    label = label.replace("_", " ") # remplacement des _ par des espaces    
    words = word_tokenize(label)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    if len(words)>0:
        tokens.append(" ".join(words))

#Conversion des tokens en une representation numerique
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tokens)

# Calcul de la matrice de similarité
similarity_matrix = cosine_similarity(X.toarray())

# Clustering des articles avec KMEANS
kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(similarity_matrix)

# Ajout des clusters aux données
df_reduit['cluster'] = clusters

# Affichage des clusters
print(df_reduit.groupby('cluster').size())


user_data = df_reduit[df_reduit["CLI_ID"] == 1490281]
user_clusters = user_data["cluster"].unique()
recommendation_data = df_reduit[df_reduit["cluster"].isin(user_clusters)]
#calcul de la distance entre les mots
distance = []
for i in range(recommendation_data.shape[0]):
    distance.append(cosine_similarity(vectorizer.transform([recommendation_data["MAILLE"].values[i]]), X)[0][0])

recommendation_data.loc[:,'distance'] = distance


# Suppression des lignes en double
recommendation_data = recommendation_data.drop_duplicates(subset=["MAILLE"])


# Tri des produits par distance et prix pour obtenir les 5 produits les plus similaires
recommendation_data = recommendation_data.sort_values(by=["distance","PRIX_NET"], ascending=[False, True]).head(5)

print(recommendation_data[["MAILLE","LIBELLE","PRIX_NET"]])
