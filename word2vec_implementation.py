"""
# **Implémentation from scratch de Word2Vec**

Implémentation des modèles Word2Vec (CBOW et Skip-gram) avec negative sampling
et sous-échantillonnage des mots fréquents. Inclut des outils d'évaluation et
de visualisation des embeddings.
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os
import string
import time
from collections import Counter, defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging
import pickle
import random
from tqdm import tqdm
import multiprocessing
from functools import partial
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Téléchargement des ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
try:
    nltk.download('punkt_tab')
except:
    # Fallback si punkt_tab n'est pas disponible
    pass

# ===================================
# 1. PARTIE THÉORIQUE ET INTRODUCTION
# ===================================

"""
Word2Vec génère des représentations vectorielles de mots basées sur l'hypothèse
distributionnelle que les mots apparaissant dans des contextes similaires ont
des significations similaires.

Architectures:
1. CBOW: prédit un mot à partir de son contexte
2. Skip-gram: prédit le contexte à partir d'un mot

Avantages:
- Capture la sémantique des mots
- Permet des opérations vectorielles (roi - homme + femme ≈ reine)
- Plus compact que les représentations one-hot
- Généralisation à de nouveaux contextes

Base mathématique:
- Optimisation par descente de gradient
- Techniques d'optimisation: negative sampling, subsampling
"""

# ===================================
# 2. PRÉPARATION DES DONNÉES
# ===================================

class TextPreprocessor:
    """Prétraitement des données textuelles."""
    
    def __init__(self, min_count=5, max_vocab_size=None, remove_stopwords=True, 
                 lemmatize=True, lowercase=True, window_size=5):
        """
        Initialise le préprocesseur de texte.
        
        Args:
            min_count: Occurrences minimales pour inclusion dans le vocabulaire
            max_vocab_size: Taille maximale du vocabulaire
            remove_stopwords: Suppression des stopwords
            lemmatize: Application de la lemmatisation
            lowercase: Conversion en minuscules
            window_size: Taille de la fenêtre contextuelle
        """
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.window_size = window_size
        
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        
        # Outils pour le prétraitement
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        
        # Pour le sous-échantillonnage
        self.word_frequencies = {}
        self.total_words = 0
        self.subsampling_threshold = 1e-5
        
    def normalize_text(self, text):
        """
        Normalise un texte par tokenisation et nettoyage.
        
        Args:
            text: Texte à normaliser
            
        Returns:
            Liste de tokens normalisés
        """
        if self.lowercase:
            text = text.lower()
        
        # Nettoyage du texte
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' NUM ', text)
        
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
            
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
            
        return tokens
    
    def build_vocab(self, texts):
        """
        Construit le vocabulaire à partir des textes.
        
        Args:
            texts: Liste de textes
            
        Returns:
            self pour chaînage
        """
        logger.info("Construction du vocabulaire...")
        
        for text in tqdm(texts):
            tokens = self.normalize_text(text)
            self.word_counts.update(tokens)
            self.total_words += len(tokens)
        
        # Filtrage par fréquence
        filtered_words = {word: count for word, count in self.word_counts.items() 
                         if count >= self.min_count}
        
        # Limitation de taille
        if self.max_vocab_size and len(filtered_words) > self.max_vocab_size:
            filtered_words = dict(sorted(filtered_words.items(), 
                                         key=lambda x: x[1], reverse=True)[:self.max_vocab_size])
        
        # Création des mappings
        self.word2idx = {word: idx for idx, word in enumerate(filtered_words.keys())}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        # Calcul des fréquences relatives
        self.word_frequencies = {word: count / self.total_words for word, count in filtered_words.items()}
        
        logger.info(f"Vocabulaire construit avec {self.vocab_size} mots uniques")
        return self
    
    def subsample_prob(self, word):
        """
        Calcule la probabilité de conserver un mot selon sa fréquence.
        
        Args:
            word: Mot à évaluer
            
        Returns:
            Probabilité de conserver le mot (entre 0 et 1)
        """
        if word not in self.word_frequencies:
            return 0
        
        frequency = self.word_frequencies[word]
        if frequency == 0:
            return 0
        
        t = self.subsampling_threshold
        ratio = t / frequency
        prob = (np.sqrt(ratio) + 1) * ratio
        
        return min(prob, 1.0)
    
    def generate_training_pairs(self, texts, model_type='cbow'):
        """
        Génère des paires d'entraînement à partir des textes.
        
        Args:
            texts: Liste de textes
            model_type: 'cbow' ou 'skipgram'
            
        Returns:
            Liste de paires (contexte, cible)
        """
        logger.info(f"Génération des paires d'entraînement pour {model_type}...")
        
        training_pairs = []
        
        for text in tqdm(texts):
            tokens = self.normalize_text(text)
            
            # Filtrage et sous-échantillonnage
            filtered_tokens = []
            for token in tokens:
                if token in self.word2idx:
                    prob = self.subsample_prob(token)
                    if prob >= 1.0 or random.random() <= prob:
                        filtered_tokens.append(token)
            
            # Génération des paires
            for i, target in enumerate(filtered_tokens):
                window_start = max(0, i - self.window_size)
                window_end = min(len(filtered_tokens), i + self.window_size + 1)
                
                context_words = [filtered_tokens[j] for j in range(window_start, window_end) 
                                if j != i and filtered_tokens[j] in self.word2idx]
                
                if not context_words:
                    continue
                
                target_idx = self.word2idx[target]
                context_indices = [self.word2idx[word] for word in context_words]
                
                if model_type == 'cbow':
                    training_pairs.append((context_indices, target_idx))
                else:  # skipgram
                    for context_idx in context_indices:
                        training_pairs.append((target_idx, context_idx))
                        
        logger.info(f"Génération terminée avec {len(training_pairs)} paires")
        return training_pairs
    
    def save(self, filepath):
        """Sauvegarde le préprocesseur dans un fichier."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath):
        """Charge un préprocesseur depuis un fichier."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# ===================================
# 3. MODÈLE CBOW
# ===================================

class CBOW:
    """Modèle Continuous Bag of Words prédisant un mot à partir de son contexte."""
    
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.025, negative_samples=5):
        """
        Initialise le modèle CBOW.
        
        Args:
            vocab_size: Taille du vocabulaire
            embedding_dim: Dimension des embeddings
            learning_rate: Taux d'apprentissage
            negative_samples: Nombre d'échantillons négatifs par exemple
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        
        # Initialisation des matrices
        self.W = np.random.normal(0, 0.01, (vocab_size, embedding_dim))
        self.W_prime = np.random.normal(0, 0.01, (vocab_size, embedding_dim))
        
        # Table pour le negative sampling
        self.prepare_negative_sampling_table()
        
    def prepare_negative_sampling_table(self, table_size=100000000):
        """
        Crée une table de distribution pour l'échantillonnage négatif.
        
        Args:
            table_size: Taille de la table
        """
        self.negative_sampling_table = np.arange(self.vocab_size)
    
    def negative_sampling(self, positive_example, num_samples):
        """
        Sélectionne des exemples négatifs pour l'entraînement.
        
        Args:
            positive_example: Indice du mot positif à éviter
            num_samples: Nombre d'échantillons à générer
            
        Returns:
            Liste d'indices de mots négatifs
        """
        negative_samples = []
        while len(negative_samples) < num_samples:
            neg = np.random.randint(0, self.vocab_size)
            if neg != positive_example and neg not in negative_samples:
                negative_samples.append(neg)
        return negative_samples
    
    def forward(self, context_indices):
        """
        Calcule la représentation vectorielle du contexte.
        
        Args:
            context_indices: Indices des mots du contexte
            
        Returns:
            Vecteur de contexte moyen
        """
        context_vectors = np.array([self.W[idx] for idx in context_indices])
        context_vector = np.mean(context_vectors, axis=0)
        
        return context_vector
    
    def sigmoid(self, x):
        """Fonction sigmoïde optimisée pour éviter les overflow."""
        return 1 / (1 + np.exp(-x))
    
    def compute_loss_and_gradients(self, context_vector, target_idx, negative_indices):
        """
        Calcule la perte et les gradients pour un exemple.
        
        Args:
            context_vector: Vecteur de contexte moyen
            target_idx: Indice du mot cible
            negative_indices: Indices des mots négatifs
            
        Returns:
            Tuple (perte, gradients)
        """
        # Vecteurs cible et négatifs
        target_vector = self.W_prime[target_idx]
        negative_vectors = self.W_prime[negative_indices]
        
        # Calcul des scores
        target_score = np.dot(context_vector, target_vector)
        target_prob = self.sigmoid(target_score)
        
        negative_scores = np.dot(negative_vectors, context_vector)
        negative_probs = self.sigmoid(-negative_scores)
        
        # Calcul de la perte
        loss = -np.log(target_prob) - np.sum(np.log(negative_probs))
        
        # Calcul des gradients
        target_gradient = context_vector * (target_prob - 1)
        negative_gradients = np.outer(1 - negative_probs, context_vector)
        
        context_gradient = (target_prob - 1) * target_vector
        for i, neg_idx in enumerate(negative_indices):
            context_gradient += (1 - negative_probs[i]) * negative_vectors[i]
            
        return loss, {
            'target': (target_idx, target_gradient),
            'negatives': list(zip(negative_indices, negative_gradients)),
            'context': context_gradient
        }
    
    def update_weights(self, gradients, context_indices):
        """
        Met à jour les poids du modèle.
        
        Args:
            gradients: Dictionnaire des gradients
            context_indices: Indices des mots du contexte
        """
        # Mise à jour pour le mot cible
        target_idx, target_gradient = gradients['target']
        self.W_prime[target_idx] -= self.learning_rate * target_gradient
        
        # Mise à jour pour les mots négatifs
        for neg_idx, neg_gradient in gradients['negatives']:
            self.W_prime[neg_idx] -= self.learning_rate * neg_gradient
        
        # Mise à jour pour les mots contextuels
        context_gradient = gradients['context']
        for idx in context_indices:
            self.W[idx] -= self.learning_rate * context_gradient
    
    def train_pair(self, context_indices, target_idx):
        """
        Entraîne le modèle sur une paire (contexte, cible).
        
        Args:
            context_indices: Indices des mots du contexte
            target_idx: Indice du mot cible
            
        Returns:
            Perte pour cette paire
        """
        context_vector = self.forward(context_indices)
        negative_indices = self.negative_sampling(target_idx, self.negative_samples)
        loss, gradients = self.compute_loss_and_gradients(context_vector, target_idx, negative_indices)
        self.update_weights(gradients, context_indices)
        
        return loss
    
    def train(self, training_pairs, epochs=5, batch_size=256, verbose=True):
        """
        Entraîne le modèle sur un ensemble de paires.
        
        Args:
            training_pairs: Liste de paires (contexte, cible)
            epochs: Nombre d'époques
            batch_size: Taille des lots
            verbose: Affichage de la progression
            
        Returns:
            Historique des pertes
        """
        losses = []
        total_batches = len(training_pairs) // batch_size + (1 if len(training_pairs) % batch_size > 0 else 0)
        
        for epoch in range(epochs):
            epoch_loss = 0
            start_time = time.time()
            
            random.shuffle(training_pairs)
            
            for batch_idx in tqdm(range(0, len(training_pairs), batch_size), 
                                   desc=f"Époque {epoch+1}/{epochs}",
                                   disable=not verbose):
                batch_pairs = training_pairs[batch_idx:batch_idx + batch_size]
                
                batch_loss = 0
                for context_indices, target_idx in batch_pairs:
                    loss = self.train_pair(context_indices, target_idx)
                    batch_loss += loss
                
                epoch_loss += batch_loss / len(batch_pairs)
            
            epoch_loss /= total_batches
            losses.append(epoch_loss)
            
            end_time = time.time()
            if verbose:
                logger.info(f"Époque {epoch+1}/{epochs}, Perte: {epoch_loss:.4f}, Temps: {end_time - start_time:.2f}s")
        
        return losses
    
    def get_word_embedding(self, word_idx):
        """Récupère l'embedding d'un mot."""
        return self.W[word_idx]
    
    def get_embeddings(self):
        """Récupère tous les embeddings."""
        return self.W
    
    def save(self, filepath):
        """Sauvegarde le modèle dans un fichier."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath):
        """Charge un modèle depuis un fichier."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# ===================================
# 4. MODÈLE SKIP-GRAM
# ===================================

class SkipGram:
    """Modèle Skip-gram prédisant le contexte à partir d'un mot cible."""
    
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.025, negative_samples=5):
        """
        Initialise le modèle Skip-gram.
        
        Args:
            vocab_size: Taille du vocabulaire
            embedding_dim: Dimension des embeddings
            learning_rate: Taux d'apprentissage
            negative_samples: Nombre d'échantillons négatifs par exemple
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        
        self.W = np.random.normal(0, 0.01, (vocab_size, embedding_dim))
        self.W_prime = np.random.normal(0, 0.01, (vocab_size, embedding_dim))
        
        self.prepare_negative_sampling_table()
        
    def prepare_negative_sampling_table(self, table_size=100000000):
        """
        Prépare une table pour l'échantillonnage négatif.
        
        Args:
            table_size: Taille de la table
        """
        self.negative_sampling_table = np.arange(self.vocab_size)
    
    def negative_sampling(self, positive_example, num_samples):
        """
        Sélectionne des exemples négatifs pour l'entraînement.
        
        Args:
            positive_example: Indice du mot positif
            num_samples: Nombre d'échantillons
            
        Returns:
            Liste d'indices négatifs
        """
        negative_samples = []
        while len(negative_samples) < num_samples:
            neg = np.random.randint(0, self.vocab_size)
            if neg != positive_example and neg not in negative_samples:
                negative_samples.append(neg)
        return negative_samples
    
    def sigmoid(self, x):
        """Fonction sigmoïde."""
        return 1 / (1 + np.exp(-x))
    
    def train_pair(self, target_idx, context_idx):
        """
        Entraîne le modèle sur une paire (cible, contexte).
        
        Args:
            target_idx: Indice du mot cible
            context_idx: Indice du mot contextuel
            
        Returns:
            Perte pour cette paire
        """
        target_vector = self.W[target_idx]
        context_vector = self.W_prime[context_idx]
        
        negative_indices = self.negative_sampling(context_idx, self.negative_samples)
        negative_vectors = self.W_prime[negative_indices]
        
        context_score = np.dot(target_vector, context_vector)
        context_prob = self.sigmoid(context_score)
        
        negative_scores = np.dot(target_vector, negative_vectors.T)
        negative_probs = self.sigmoid(-negative_scores)
        
        loss = -np.log(context_prob) - np.sum(np.log(negative_probs))
        
        # Gradients
        context_gradient = target_vector * (context_prob - 1)
        
        negative_gradients = []
        for i, neg_idx in enumerate(negative_indices):
            neg_gradient = target_vector * (1 - negative_probs[i])
            negative_gradients.append((neg_idx, neg_gradient))
        
        target_gradient = (context_prob - 1) * context_vector
        for i, neg_idx in enumerate(negative_indices):
            target_gradient += (1 - negative_probs[i]) * negative_vectors[i]
        
        # Mise à jour
        self.W_prime[context_idx] -= self.learning_rate * context_gradient
        
        for neg_idx, neg_gradient in negative_gradients:
            self.W_prime[neg_idx] -= self.learning_rate * neg_gradient
        
        self.W[target_idx] -= self.learning_rate * target_gradient
        
        return loss
    
    def train(self, training_pairs, epochs=5, batch_size=256, verbose=True):
        """
        Entraîne le modèle sur un ensemble de paires.
        
        Args:
            training_pairs: Liste de paires (cible, contexte)
            epochs: Nombre d'époques
            batch_size: Taille des lots
            verbose: Affichage de la progression
            
        Returns:
            Historique des pertes
        """
        losses = []
        total_batches = len(training_pairs) // batch_size + (1 if len(training_pairs) % batch_size > 0 else 0)
        
        for epoch in range(epochs):
            epoch_loss = 0
            start_time = time.time()
            
            random.shuffle(training_pairs)
            
            for batch_idx in tqdm(range(0, len(training_pairs), batch_size), 
                                   desc=f"Époque {epoch+1}/{epochs}",
                                   disable=not verbose):
                batch_pairs = training_pairs[batch_idx:batch_idx + batch_size]
                
                batch_loss = 0
                for target_idx, context_idx in batch_pairs:
                    loss = self.train_pair(target_idx, context_idx)
                    batch_loss += loss
                
                epoch_loss += batch_loss / len(batch_pairs)
            
            epoch_loss /= total_batches
            losses.append(epoch_loss)
            
            end_time = time.time()
            if verbose:
                logger.info(f"Époque {epoch+1}/{epochs}, Perte: {epoch_loss:.4f}, Temps: {end_time - start_time:.2f}s")
        
        return losses
    
    def get_word_embedding(self, word_idx):
        """Récupère l'embedding d'un mot."""
        return self.W[word_idx]
    
    def get_embeddings(self):
        """Récupère tous les embeddings."""
        return self.W
    
    def save(self, filepath):
        """Sauvegarde le modèle dans un fichier."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath):
        """Charge un modèle depuis un fichier."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# ===================================
# 5. OPTIMISATIONS AVANCÉES
# ===================================

class NegativeSamplingTable:
    """Table d'échantillonnage négatif basée sur les fréquences des mots."""
    
    def __init__(self, word_frequencies, table_size=100000000):
        """
        Initialise la table d'échantillonnage.
        
        Args:
            word_frequencies: Dictionnaire {mot: fréquence}
            table_size: Taille de la table
        """
        self.table_size = table_size
        self.table = self._create_table(word_frequencies)
        self.vocab_size = len(word_frequencies)
        
    def _create_table(self, word_frequencies):
        """
        Crée la table de distribution pour l'échantillonnage.
        
        Args:
            word_frequencies: Dictionnaire {mot: fréquence}
            
        Returns:
            Table d'échantillonnage
        """
        logger.info("Création de la table d'échantillonnage négatif...")
        
        words = list(word_frequencies.keys())
        frequencies = np.array(list(word_frequencies.values()))
        
        # Fréquences à la puissance 3/4
        pow_frequencies = np.power(frequencies, 0.75)
        
        pow_sum = np.sum(pow_frequencies)
        probs = pow_frequencies / pow_sum if pow_sum > 0 else pow_frequencies
        
        table = np.zeros(self.table_size, dtype=np.int32)
        
        p = 0
        i = 0
        for word_idx, word in enumerate(words):
            segment_size = int(probs[word_idx] * self.table_size)
            for _ in range(segment_size):
                if i < self.table_size:
                    table[i] = word_idx
                    i += 1
        
        while i < self.table_size:
            table[i] = random.randint(0, len(words) - 1)
            i += 1
            
        logger.info(f"Table d'échantillonnage créée avec {self.table_size} entrées")
        return table
    
    def sample(self, positive_examples=None, n_samples=5):
        """
        Échantillonne des exemples négatifs.
        
        Args:
            positive_examples: Exemples positifs à éviter
            n_samples: Nombre d'échantillons
            
        Returns:
            Liste d'indices de mots négatifs
        """
        if positive_examples is None:
            positive_examples = []
        elif not isinstance(positive_examples, list):
            positive_examples = [positive_examples]
            
        samples = []
        while len(samples) < n_samples:
            idx = random.randint(0, self.table_size - 1)
            sample = self.table[idx]
            if sample not in positive_examples and sample not in samples:
                samples.append(sample)
                
        return samples

# ===================================
# 6. VISUALISATION ET ÉVALUATION
# ===================================

class Word2VecEvaluator:
    """Évaluation et visualisation des embeddings Word2Vec."""
    
    def __init__(self, model, preprocessor):
        """
        Initialise l'évaluateur.
        
        Args:
            model: Modèle Word2Vec
            preprocessor: Préprocesseur de texte
        """
        self.model = model
        self.preprocessor = preprocessor
        self.embeddings = model.get_embeddings()
        
    def word_similarity(self, word1, word2):
        """
        Calcule la similarité cosinus entre deux mots.
        
        Args:
            word1: Premier mot
            word2: Deuxième mot
            
        Returns:
            Similarité cosinus
        """
        if word1 not in self.preprocessor.word2idx or word2 not in self.preprocessor.word2idx:
            return None
        
        idx1 = self.preprocessor.word2idx[word1]
        idx2 = self.preprocessor.word2idx[word2]
        
        vec1 = self.embeddings[idx1].reshape(1, -1)
        vec2 = self.embeddings[idx2].reshape(1, -1)
        
        return cosine_similarity(vec1, vec2)[0][0]
    
    def most_similar(self, word, n=10):
        """
        Trouve les mots les plus similaires à un mot donné.
        
        Args:
            word: Mot de référence
            n: Nombre de mots à retourner
            
        Returns:
            Liste de tuples (mot, similarité)
        """
        if word not in self.preprocessor.word2idx:
            return []
        
        word_idx = self.preprocessor.word2idx[word]
        word_vec = self.embeddings[word_idx].reshape(1, -1)
        
        similarities = cosine_similarity(word_vec, self.embeddings)[0]
        
        most_similar = []
        for idx in similarities.argsort()[::-1]:
            if idx != word_idx:
                similar_word = self.preprocessor.idx2word[idx]
                similarity = similarities[idx]
                most_similar.append((similar_word, similarity))
                if len(most_similar) >= n:
                    break
                    
        return most_similar
    
    def evaluate_analogies(self, analogies):
        """
        Évalue le modèle sur des tâches d'analogie.
        
        Args:
            analogies: Liste de tuples (a, b, c, d) où a:b::c:d
            
        Returns:
            Précision sur les analogies
        """
        correct = 0
        
        for a, b, c, d in analogies:
            if (a not in self.preprocessor.word2idx or 
                b not in self.preprocessor.word2idx or 
                c not in self.preprocessor.word2idx or 
                d not in self.preprocessor.word2idx):
                continue
                
            a_idx = self.preprocessor.word2idx[a]
            b_idx = self.preprocessor.word2idx[b]
            c_idx = self.preprocessor.word2idx[c]
            
            # d ≈ c + (b - a)
            target_vec = self.embeddings[c_idx] + (self.embeddings[b_idx] - self.embeddings[a_idx])
            
            similarities = cosine_similarity(target_vec.reshape(1, -1), self.embeddings)[0]
            
            similarities[a_idx] = -np.inf
            similarities[b_idx] = -np.inf
            similarities[c_idx] = -np.inf
            
            predicted_idx = np.argmax(similarities)
            predicted_word = self.preprocessor.idx2word[predicted_idx]
            
            if predicted_word == d:
                correct += 1
                
        if len(analogies) == 0:
            return 0
        else:
            return correct / len(analogies)
    
    def visualize_embeddings(self, words=None, n=100, method='tsne', n_components=2, random_state=42):
        """
        Visualise les embeddings dans un espace réduit.
        
        Args:
            words: Liste de mots à visualiser
            n: Nombre de mots à visualiser
            method: Méthode de réduction ('tsne' ou 'pca')
            n_components: Nombre de composantes (2 ou 3)
            random_state: Graine aléatoire
        """
        if words is None:
            words = []
            word_counts = sorted(self.preprocessor.word_counts.items(), key=lambda x: x[1], reverse=True)
            for word, _ in word_counts:
                if word in self.preprocessor.word2idx:
                    words.append(word)
                    if len(words) >= n:
                        break
        else:
            words = [word for word in words if word in self.preprocessor.word2idx]
            
        if not words:
            logger.warning("Aucun mot à visualiser.")
            return
            
        word_indices = [self.preprocessor.word2idx[word] for word in words]
        word_vectors = self.embeddings[word_indices]
        
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=random_state)
        else:  # method == 'pca'
            reducer = PCA(n_components=n_components, random_state=random_state)
            
        reduced_vectors = reducer.fit_transform(word_vectors)
        
        plt.figure(figsize=(12, 10))
        
        if n_components == 2:
            plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.7)
            for i, word in enumerate(words):
                plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
        else:  # n_components == 3
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], reduced_vectors[:, 2], alpha=0.7)
            for i, word in enumerate(words):
                ax.text(reduced_vectors[i, 0], reduced_vectors[i, 1], reduced_vectors[i, 2], word)
                
        plt.title(f"Visualisation des embeddings Word2Vec ({method.upper()})")
        plt.grid(True)
        plt.show()
        
    def document_to_vector(self, document):
        """
        Convertit un document en vecteur par moyenne des embeddings.
        
        Args:
            document: Texte du document
            
        Returns:
            Vecteur du document
        """
        tokens = self.preprocessor.normalize_text(document)
        tokens = [token for token in tokens if token in self.preprocessor.word2idx]
        
        if not tokens:
            return np.zeros(self.model.embedding_dim)
            
        word_indices = [self.preprocessor.word2idx[token] for token in tokens]
        word_vectors = self.embeddings[word_indices]
        
        return np.mean(word_vectors, axis=0)
        
    def save_embeddings_to_file(self, filepath):
        """
        Sauvegarde les embeddings dans un fichier texte.
        
        Args:
            filepath: Chemin du fichier
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{self.preprocessor.vocab_size} {self.model.embedding_dim}\n")
            
            for idx in range(self.preprocessor.vocab_size):
                word = self.preprocessor.idx2word[idx]
                embedding = self.embeddings[idx]
                vector_str = ' '.join(str(x) for x in embedding)
                f.write(f"{word} {vector_str}\n")
                
        logger.info(f"Embeddings sauvegardés dans {filepath}")
    
# ===================================
# 7. APPLICATION PRATIQUE
# ===================================

class TextClassifier:
    """Classification de texte utilisant des embeddings Word2Vec."""
    
    def __init__(self, evaluator, classifier=None):
        """
        Initialise le classificateur.
        
        Args:
            evaluator: Évaluateur Word2Vec
            classifier: Classificateur (défaut: MLPClassifier)
        """
        self.evaluator = evaluator
        
        if classifier is None:
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                max_iter=200,
                random_state=42
            )
        else:
            self.classifier = classifier
            
    def prepare_data(self, texts, labels):
        """
        Prépare les données pour la classification.
        
        Args:
            texts: Liste de textes
            labels: Liste d'étiquettes
            
        Returns:
            Tuple (X, y)
        """
        X = np.array([self.evaluator.document_to_vector(text) for text in texts])
        y = np.array(labels)
        
        return X, y
    
    def train(self, train_texts, train_labels):
        """
        Entraîne le classificateur.
        
        Args:
            train_texts: Textes d'entraînement
            train_labels: Étiquettes d'entraînement
            
        Returns:
            self pour chaînage
        """
        X_train, y_train = self.prepare_data(train_texts, train_labels)
        self.classifier.fit(X_train, y_train)
        
        return self
    
    def evaluate(self, test_texts, test_labels):
        """
        Évalue le classificateur.
        
        Args:
            test_texts: Textes de test
            test_labels: Étiquettes de test
            
        Returns:
            Rapport de classification
        """
        X_test, y_test = self.prepare_data(test_texts, test_labels)
        y_pred = self.classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Précision: {accuracy:.4f}")
        logger.info(f"Rapport de classification:\n{report}")
        
        return accuracy, report
    
    def predict(self, texts):
        """
        Prédit les étiquettes pour des textes.
        
        Args:
            texts: Liste de textes
            
        Returns:
            Prédictions
        """
        X = np.array([self.evaluator.document_to_vector(text) for text in texts])
        return self.classifier.predict(X)
    
    def save(self, filepath):
        """Sauvegarde le classificateur dans un fichier."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filepath):
        """Charge un classificateur depuis un fichier."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# ===================================
# 8. PIPELINE COMPLET
# ===================================

def load_20newsgroups():
    """
    Charge le dataset 20 Newsgroups.
    
    Returns:
        Tuple (train_texts, train_labels, test_texts, test_labels)
    """
    from sklearn.datasets import fetch_20newsgroups
    
    train_data = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    test_data = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    return train_data.data, train_data.target, test_data.data, test_data.target

def load_imdb():
    """
    Charge le dataset IMDb.
    
    Returns:
        Tuple (train_texts, train_labels, test_texts, test_labels)
    """
    try:
        df = pd.read_csv('imdb_dataset.csv')
    except FileNotFoundError:
        logger.error("Dataset IMDb non trouvé. Téléchargez-le depuis https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        return None, None, None, None
    
    label_map = {'positive': 1, 'negative': 0}
    df['label'] = df['sentiment'].map(label_map)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    return (train_df['review'].tolist(), train_df['label'].tolist(),
            test_df['review'].tolist(), test_df['label'].tolist())

def main_pipeline(dataset='20newsgroups', model_type='cbow', embedding_dim=100,
                  window_size=5, min_count=5, negative_samples=5, epochs=5,
                  subsampling=True, subsampling_threshold=1e-5):
    """
    Pipeline complet pour l'entraînement et l'évaluation d'un modèle Word2Vec.
    
    Args:
        dataset: Nom du dataset ('20newsgroups' ou 'imdb')
        model_type: Type de modèle ('cbow' ou 'skipgram')
        embedding_dim: Dimension des embeddings
        window_size: Taille de la fenêtre contextuelle
        min_count: Seuil minimal d'occurrences pour un mot
        negative_samples: Nombre d'échantillons négatifs
        epochs: Nombre d'époques
        subsampling: Application du sous-échantillonnage
        subsampling_threshold: Seuil pour le sous-échantillonnage
    """
    # 1. Chargement des données
    logger.info(f"Chargement du dataset {dataset}...")
    if dataset == '20newsgroups':
        train_texts, train_labels, test_texts, test_labels = load_20newsgroups()
    else:  # dataset == 'imdb'
        train_texts, train_labels, test_texts, test_labels = load_imdb()
        
    if train_texts is None:
        return
    
    # 2. Prétraitement
    logger.info("Prétraitement des données...")
    preprocessor = TextPreprocessor(
        min_count=min_count,
        window_size=window_size,
        remove_stopwords=True,
        lemmatize=True
    )
    
    if subsampling:
        preprocessor.subsampling_threshold = subsampling_threshold
    else:
        preprocessor.subsampling_threshold = 1e-10
    
    preprocessor.build_vocab(train_texts)
    training_pairs = preprocessor.generate_training_pairs(train_texts, model_type=model_type)
    
    # 3. Création et entraînement du modèle
    logger.info(f"Entraînement du modèle {model_type}...")
    if model_type == 'cbow':
        model = CBOW(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=embedding_dim,
            negative_samples=negative_samples
        )
    else:  # model_type == 'skipgram'
        model = SkipGram(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=embedding_dim,
            negative_samples=negative_samples
        )
    
    losses = model.train(training_pairs, epochs=epochs)
    
    # 4. Évaluation
    logger.info("Évaluation du modèle...")
    evaluator = Word2VecEvaluator(model, preprocessor)
    
    for word in ['computer', 'science', 'technology', 'movie', 'good', 'bad']:
        if word in preprocessor.word2idx:
            logger.info(f"Mots similaires à '{word}':")
            similar_words = evaluator.most_similar(word, n=5)
            for similar_word, similarity in similar_words:
                logger.info(f"  {similar_word}: {similarity:.4f}")
    
    # 5. Classification
    logger.info("Entraînement du classificateur...")
    classifier = TextClassifier(evaluator)
    classifier.train(train_texts, train_labels)
    
    logger.info("Évaluation du classificateur...")
    accuracy, report = classifier.evaluate(test_texts, test_labels)
    
    # 6. Visualisation
    logger.info("Visualisation des embeddings...")
    common_words = []
    word_counts = sorted(preprocessor.word_counts.items(), key=lambda x: x[1], reverse=True)
    for word, _ in word_counts:
        if word in preprocessor.word2idx and len(word) > 2:
            common_words.append(word)
            if len(common_words) >= 100:
                break
                
    evaluator.visualize_embeddings(words=common_words, method='tsne')
    
    # 7. Sauvegarde
    logger.info("Sauvegarde des résultats...")
    os.makedirs('results', exist_ok=True)
    
    model.save(f'results/{model_type}_model.pkl')
    preprocessor.save(f'results/{model_type}_preprocessor.pkl')
    evaluator.save_embeddings_to_file(f'results/{model_type}_embeddings.txt')
    classifier.save(f'results/{model_type}_classifier.pkl')
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f"Évolution de la perte pendant l'entraînement ({model_type})")
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.grid(True)
    plt.savefig(f'results/{model_type}_loss.png')
    
    logger.info("Pipeline terminé avec succès!")

# ===================================
# EXÉCUTION DU PROGRAMME
# ===================================

if __name__ == "__main__":
    main_pipeline(
        dataset='20newsgroups',
        model_type='cbow',
        embedding_dim=100,
        window_size=5,
        min_count=5,
        negative_samples=10,
        epochs=5,
        subsampling=True,
        subsampling_threshold=1e-5
    )
    
    main_pipeline(
        dataset='20newsgroups',
        model_type='skipgram',
        embedding_dim=100,
        window_size=5,
        min_count=5,
        negative_samples=10,
        epochs=5,
        subsampling=True,
        subsampling_threshold=1e-5
    )