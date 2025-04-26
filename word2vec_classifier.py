import numpy as np
import re
import string
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if len(token) > 1]
    return tokens

def get_imdb_data():
    vocab_size = 10000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
    word_index = imdb.get_word_index()
    index_word = {i: word for word, i in word_index.items()}
    
    def decode_review(encoded_review):
        return ' '.join([index_word.get(i - 3, '?') for i in encoded_review if i > 3])
    
    train_texts = [decode_review(x) for x in X_train]
    test_texts = [decode_review(x) for x in X_test]
    
    all_texts = train_texts + test_texts
    return all_texts

def get_20newsgroups_data():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    all_texts = newsgroups_train.data + newsgroups_test.data
    return all_texts

def main():
    print("Chargement des données IMDB...")
    imdb_texts = get_imdb_data()
    
    print("Chargement des données 20newsgroups...")
    newsgroups_texts = get_20newsgroups_data()
    
    print(f"Nombre de textes IMDB: {len(imdb_texts)}")
    print(f"Nombre de textes 20newsgroups: {len(newsgroups_texts)}")
    
    all_texts = imdb_texts + newsgroups_texts
    print(f"Nombre total de textes: {len(all_texts)}")
    
    print("Prétraitement des textes...")
    tokenized_texts = [preprocess_text(text) for text in all_texts]
    
    print("Entraînement du modèle Word2Vec...")
    word2vec_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=300,
        window=5,
        min_count=5,
        workers=4,
        sg=1
    )
    
    print("Enregistrement du modèle...")
    word2vec_model.save("supreme_word2vec_model.model")
    print("Modèle enregistré sous 'supreme_word2vec_model.model'")
    
    print("Quelques mots similaires pour tester le modèle:")
    for word in ['movie', 'computer', 'science', 'good', 'bad']:
        if word in word2vec_model.wv:
            similars = word2vec_model.wv.most_similar(word, topn=5)
            print(f"Similaires à '{word}': {similars}")

if __name__ == "__main__":
    main()





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec, KeyedVectors
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
import nltk
import re
import string
import os
import time
import zipfile
import requests
from io import BytesIO

nltk.download('punkt')

CUSTOM_MODEL_PATH = "/Users/yanis/Desktop/boudra/supreme_word2vec_model.model"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_PATH = "glove.6B.300d.txt"

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if len(token) > 1]
        return tokens
    return []

def text_to_vector(text, model, vector_size=300, is_word2vec=True):
    tokens = preprocess_text(text)
    vectors = []
    for token in tokens:
        if is_word2vec and token in model.wv:
            vectors.append(model.wv[token])
        elif not is_word2vec and token in model:
            vectors.append(model[token])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(vector_size)

def prepare_imdb_data():
    print("Préparation des données IMDB...")
    vocab_size = 10000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
    word_index = imdb.get_word_index()
    index_word = {i: word for word, i in word_index.items()}
    
    def decode_review(encoded_review):
        return ' '.join([index_word.get(i - 3, '?') for i in encoded_review if i > 3])
    
    train_texts = [decode_review(x) for x in X_train]
    test_texts = [decode_review(x) for x in X_test]
    
    return train_texts, y_train, test_texts, y_test

def prepare_20newsgroups_data():
    print("Préparation des données 20newsgroups...")
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    return (newsgroups_train.data, newsgroups_train.target, 
            newsgroups_test.data, newsgroups_test.target)

def create_ann_model(input_dim, output_dim, hidden_layers=[256, 128]):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.3))
    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.3))
    model.add(Dense(output_dim, activation='softmax' if output_dim > 1 else 'sigmoid'))
    
    model.compile(
        loss='categorical_crossentropy' if output_dim > 1 else 'binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def evaluate_model(y_true, y_pred, is_binary=True):
    if not is_binary:
        y_true_argmax = np.argmax(y_true, axis=1)
        y_pred_argmax = np.argmax(y_pred, axis=1)
        
        accuracy = accuracy_score(y_true_argmax, y_pred_argmax)
        precision = precision_score(y_true_argmax, y_pred_argmax, average='weighted')
        recall = recall_score(y_true_argmax, y_pred_argmax, average='weighted')
        f1 = f1_score(y_true_argmax, y_pred_argmax, average='weighted')
    else:
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def vectorize_data(texts, model, vector_size=300, is_word2vec=True):
    vectors = []
    for text in texts:
        vectors.append(text_to_vector(text, model, vector_size, is_word2vec))
    return np.array(vectors)

def plot_comparison(custom_results, pretrained_results, title, pretrained_name="GloVe"):
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    data = {
        'Métrique': metrics,
        'Modèle custom': [custom_results[m] for m in metrics],
        f'Modèle {pretrained_name}': [pretrained_results[m] for m in metrics]
    }
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(x='Métrique', y='value', hue='variable', 
                 data=pd.melt(df, ['Métrique']))
    plt.title(title)
    plt.ylim(0, 1)
    
    for p in chart.patches:
        chart.annotate(f'{p.get_height():.3f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', xytext = (0, 5),
                   textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

def load_glove_model():
    if not os.path.exists(GLOVE_PATH):
        print(f"Téléchargement du modèle GloVe (cela peut prendre un certain temps)...")
        r = requests.get(GLOVE_URL, stream=True)
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extract("glove.6B.300d.txt")
    
    print("Chargement du modèle GloVe...")
    glove_model = {}
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_model[word] = vector
    return glove_model

def main():
    print("Chargement du modèle Word2Vec personnalisé...")
    custom_model = Word2Vec.load(CUSTOM_MODEL_PATH)
    
    print("Chargement du modèle GloVe pré-entraîné...")
    glove_model = load_glove_model()
    
    print("\n=== Classification IMDB ===")
    train_texts, train_labels, test_texts, test_labels = prepare_imdb_data()
    
    print("Vectorisation des données IMDB avec le modèle personnalisé...")
    X_train_custom = vectorize_data(train_texts, custom_model)
    X_test_custom = vectorize_data(test_texts, custom_model)
    
    print("Vectorisation des données IMDB avec le modèle GloVe...")
    X_train_glove = vectorize_data(train_texts, glove_model, is_word2vec=False)
    X_test_glove = vectorize_data(test_texts, glove_model, is_word2vec=False)
    
    print("Entraînement du modèle ANN sur IMDB avec vecteurs personnalisés...")
    imdb_model_custom = create_ann_model(300, 1, hidden_layers=[256, 128])
    imdb_model_custom.fit(X_train_custom, train_labels, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    
    print("Entraînement du modèle ANN sur IMDB avec vecteurs GloVe...")
    imdb_model_glove = create_ann_model(300, 1, hidden_layers=[256, 128])
    imdb_model_glove.fit(X_train_glove, train_labels, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    
    print("Évaluation des modèles IMDB...")
    custom_preds = imdb_model_custom.predict(X_test_custom)
    glove_preds = imdb_model_glove.predict(X_test_glove)
    
    imdb_custom_results = evaluate_model(test_labels, custom_preds, is_binary=True)
    imdb_glove_results = evaluate_model(test_labels, glove_preds, is_binary=True)
    
    print("\nRésultats IMDB avec modèle personnalisé:")
    for metric, value in imdb_custom_results.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nRésultats IMDB avec modèle GloVe:")
    for metric, value in imdb_glove_results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Classification 20newsgroups ===")
    train_texts, train_labels, test_texts, test_labels = prepare_20newsgroups_data()
    
    num_classes = len(np.unique(train_labels))
    train_labels_cat = to_categorical(train_labels, num_classes)
    test_labels_cat = to_categorical(test_labels, num_classes)
    
    print("Vectorisation des données 20newsgroups avec le modèle personnalisé...")
    X_train_custom = vectorize_data(train_texts, custom_model)
    X_test_custom = vectorize_data(test_texts, custom_model)
    
    print("Vectorisation des données 20newsgroups avec le modèle GloVe...")
    X_train_glove = vectorize_data(train_texts, glove_model, is_word2vec=False)
    X_test_glove = vectorize_data(test_texts, glove_model, is_word2vec=False)
    
    print("Entraînement du modèle ANN sur 20newsgroups avec vecteurs personnalisés...")
    news_model_custom = create_ann_model(300, num_classes, hidden_layers=[512, 256])
    news_model_custom.fit(X_train_custom, train_labels_cat, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    
    print("Entraînement du modèle ANN sur 20newsgroups avec vecteurs GloVe...")
    news_model_glove = create_ann_model(300, num_classes, hidden_layers=[512, 256])
    news_model_glove.fit(X_train_glove, train_labels_cat, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    
    print("Évaluation des modèles 20newsgroups...")
    custom_preds = news_model_custom.predict(X_test_custom)
    glove_preds = news_model_glove.predict(X_test_glove)
    
    news_custom_results = evaluate_model(test_labels_cat, custom_preds, is_binary=False)
    news_glove_results = evaluate_model(test_labels_cat, glove_preds, is_binary=False)
    
    print("\nRésultats 20newsgroups avec modèle personnalisé:")
    for metric, value in news_custom_results.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nRésultats 20newsgroups avec modèle GloVe:")
    for metric, value in news_glove_results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nCréation des graphiques de comparaison...")
    plot_comparison(imdb_custom_results, imdb_glove_results, "Comparaison Word2Vec sur IMDB", "GloVe")
    plot_comparison(news_custom_results, news_glove_results, "Comparaison Word2Vec sur 20newsgroups", "GloVe")
    
    print("\nRésumé des résultats:")
    print("\nIMDB:")
    custom_acc = imdb_custom_results['accuracy']
    glove_acc = imdb_glove_results['accuracy']
    diff = custom_acc - glove_acc
    print(f"Modèle personnalisé: {custom_acc:.4f}, Modèle GloVe: {glove_acc:.4f}")
    print(f"Différence: {diff:.4f} ({'meilleur' if diff > 0 else 'moins bon'} que GloVe)")
    
    print("\n20newsgroups:")
    custom_acc = news_custom_results['accuracy']
    glove_acc = news_glove_results['accuracy']
    diff = custom_acc - glove_acc
    print(f"Modèle personnalisé: {custom_acc:.4f}, Modèle GloVe: {glove_acc:.4f}")
    print(f"Différence: {diff:.4f} ({'meilleur' if diff > 0 else 'moins bon'} que GloVe)")

if __name__ == "__main__":
    main()
