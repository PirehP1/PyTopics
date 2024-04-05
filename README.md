
## Présentation du Script

Ce script est un exemple d'implémentation d'un modèle de *Topics Modelling*  pour l'analyse de texte.

**Import de Bibliothèques** : Le script commence par l'importation des bibliothèques nécessaires telles que os, pandas, pyLDAvis, gensim

##  Fonctions :

**remove_rows_containing_strings** : Supprime les lignes contenant des chaînes de caractères spécifiques dans une colonne donnée d'un DataFrame.
**tokenize_corpus** : Tokenise un corpus de texte en utilisant Gensim.
**create_segments** : Crée des segments de tailles identiques à partir des données d'un DataFrame.
**train_lda_model** : Entraîne un modèle de sujets LDA sur des segments tokenisés.
**remove_stopwords** : Supprime les stopwords d'un texte donné.
**load_stopwords** : Charge une liste de stopwords à partir d'un fichier.
**visualize_lda_model** : Visualise un modèle LDA avec PyLDAvis et enregistre la visualisation dans un fichier HTML.
**save_corpus_and_dictionary** : Enregistre le corpus et le dictionnaire Gensim dans des fichiers distincts.


Fonction Principale main : La fonction principale du script, main(), exécute le flux principal du programme. Elle effectue les étapes suivantes :

1. Lecture d'un fichier CSV contenant des données textuelles.
2. Suppression des lignes contenant des termes spécifiques.
3. Création de segments à partir des données.
4. Chargement des stopwords depuis un fichier et suppression des stopwords des segments tokenisés.
5. Entraînement d'un modèle LDA sur les segments tokenisés.
6. Sauvegarde du corpus et du dictionnaire Gensim.
7. Visualisation du modèle LDA et enregistrement de la visualisation dans un fichier HTML.


