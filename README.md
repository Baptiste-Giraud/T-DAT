# T-DAT
First Data project

-front
-back
-segmentation client
-aucun modele d'ia utiliser actuellement: nous avons 
Votre code utilise des techniques d'apprentissage automatique pour créer des recommendations pour les utilisateurs. Il utilise la similarité cosinus pour calculer la similarité entre les produits en utilisant un représentation numérique des libellés de produits obtenus à partir de la bibliothèque scikit-learn "CountVectorizer". Ensuite, il utilise l'algorithme de clustering K-Means pour regrouper les produits en clusters. Enfin, pour faire des recommendations pour un utilisateur donné, il utilise les clusters auxquels les produits achetés par cet utilisateur appartiennent pour filtrer les autres produits et recommande les 5 produits les plus similaires en termes de similarité cosinus et de prix.