import os
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel
from gensim.utils import tokenize as gensim_tokenize


def remove_rows_containing_strings(df, column_num, strings_to_remove):
    """
    Fonction pour enlever les lignes contenant des chaînes de caractères spécifiques dans une colonne donnée du DataFrame.
    """
    condition = df.iloc[:, column_num].str.contains('|'.join(strings_to_remove), na=False)
    return df[~condition]


def tokenize_corpus(corpus, language='english'):
    """
    Fonction pour tokeniser un corpus de texte en utilisant Gensim.
    """
    tokenized_corpus = []
    for sentence in corpus:
        tokens = list(gensim_tokenize(sentence))
        tokenized_corpus.append(tokens)
    return tokenized_corpus


def create_segments(df, segment_length, join_char):
    """
    Fonction pour créer des segments de tailles identiques à partir des données du DataFrame.
    Retourne un DataFrame contenant deux colonnes : "document" et "segment".
    """
    segments = []
    num_rows = len(df)

    for i in range(0, num_rows, segment_length):
        segment_df = df.iloc[i:i+segment_length]
        segment_text = join_char.join(segment_df.iloc[:, 1].tolist())
        segments.append(segment_text)

    # Tokenisation des segments
    tokenized_segments = tokenize_corpus(segments)

    # Création du DataFrame de sortie
    segments_df = pd.DataFrame({"document": [f"segment_{i+1}" for i in range(len(segments))], "segment": tokenized_segments})
    return segments_df


def train_lda_model(tokenized_segments, num_topics=5):
    """
    Fonction pour entraîner un modèle de sujets LDA sur les segments tokenisés.
    Retourne le modèle entraîné.
    """
    # Création du dictionnaire à partir des segments tokenisés
    dictionary = Dictionary(tokenized_segments)

    # Création du corpus
    corpus = [dictionary.doc2bow(segment) for segment in tokenized_segments]

    # Supprimer les hapax du corpus
    filtered_corpus = remove_hapax(corpus)

    # Entraînement du modèle LDA
    lda_model = LdaModel(filtered_corpus, num_topics=num_topics, id2word=dictionary)

    return lda_model


def remove_stopwords(text, stopwords):
    """
    Fonction pour supprimer les stopwords d'un texte donné.
    """
    filtered_text = [word for word in text if word not in stopwords]
    return filtered_text


def load_stopwords(file_path):
    """
    Fonction pour charger une liste de stopwords à partir d'un fichier.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = [line.strip() for line in file]
    return set(stopwords)


def visualize_lda_model(lda_model, corpus, dictionary, output_path):
    """
    Fonction pour visualiser un modèle LDA avec PyLDAvis et enregistrer la visualisation dans un fichier HTML.
    """
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, output_path)



def remove_hapax(corpus):
    """
    Fonction pour supprimer les hapax du corpus.
    """
    # Compter la fréquence de chaque mot dans le corpus
    word_frequency = {}
    for document in corpus:
        for word_id, frequency in document:
            if word_id in word_frequency:
                word_frequency[word_id] += frequency
            else:
                word_frequency[word_id] = frequency
    
    # Identifier les hapax
    hapax_ids = [word_id for word_id, frequency in word_frequency.items() if frequency == 1]
    
    # Supprimer les hapax du corpus
    filtered_corpus = [[(word_id, frequency) for word_id, frequency in document if word_id not in hapax_ids] for document in corpus]
    
    return filtered_corpus


def save_corpus_and_dictionary(corpus, dictionary, corpus_file_path, dictionary_file_path):
    """
    Fonction pour enregistrer le corpus et le dictionnaire Gensim dans des fichiers distincts.
    """
    # Enregistrement du corpus
    MmCorpus.serialize(corpus_file_path, corpus)

    # Enregistrement du dictionnaire
    dictionary.save(dictionary_file_path)



def main():
    # Chemin vers le fichier CSV
    path = ""
    csv_file_path = path + "indexTotalTabulaire.csv"

    # Lecture du fichier CSV en utilisant pandas avec la tabulation comme délimiteur
    df = pd.read_csv(csv_file_path, sep='\t')


    # Liste des termes à supprimer
    terms_remove = ["latex"]

    # Liste des termes à supprimer
    terms_2_remove = ["tabular", "{|}", "&"]

    # Suppression des lignes contenant les termes spécifiés dans la deuxième colonne (index 1)
    df = remove_rows_containing_strings(df, 2, terms_remove)
    df = remove_rows_containing_strings(df, 1, terms_2_remove)

    # Paramètres pour la création des segments
    segment_length = 40  # Longueur de chaque segment
    join_char = ' '     # Caractère de jointure entre les éléments de chaque segment

    # Création des segments
    segments_df = create_segments(df, segment_length, join_char)



    # Chargement des stopwords depuis un fichier
    stopwords = load_stopwords(path + "stopwords.txt")

    # Suppression des stopwords des segments tokenisés
    segments_df['segment'] = segments_df['segment'].apply(lambda x: remove_stopwords(x, stopwords))

    # Entraînement du modèle LDA sur les segments tokenisés
    tokenized_segments = segments_df['segment']
    dictionary = Dictionary(tokenized_segments)
    corpus = [dictionary.doc2bow(segment) for segment in tokenized_segments]
    lda_model = train_lda_model(tokenized_segments,)


    # On sauvegarde le corpus au format dédié par gensim 
    save_corpus_and_dictionary(corpus, dictionary, path + "corpus.mm", path + "dictionary.dict")


    # Visualisation du modèle LDA et enregistrement de la visualisation dans un fichier HTML
    visualize_lda_model(lda_model, corpus, dictionary, "lda_visualization.html")


if __name__ == "__main__":
    main()
