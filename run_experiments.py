"""
# **Script d'exploration et visualisation pour Word2Vec**

Outil d'exploration et visualisation des modèles Word2Vec
entraînés avec word2vec_implementation.py.

Prérequis: Exécution préalable de word2vec_implementation.py pour
générer les modèles.
"""

import os
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ajout du répertoire courant au path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import des modules
try:
    from word2vec_implementation import (
        TextPreprocessor, CBOW, SkipGram, Word2VecEvaluator, TextClassifier
    )
except ImportError as e:
    logger.error(f"Erreur d'importation: {e}")
    logger.error("Vérifiez que word2vec_implementation.py est accessible")
    sys.exit(1)

def load_model(model_path):
    """
    Charge un modèle Word2Vec depuis un fichier.
    
    Args:
        model_path: Chemin du fichier modèle
        
    Returns:
        Modèle Word2Vec
    """
    logger.info(f"Chargement du modèle depuis {model_path}...")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {e}")
        return None

def load_preprocessor(preprocessor_path):
    """
    Charge un préprocesseur depuis un fichier.
    
    Args:
        preprocessor_path: Chemin du fichier préprocesseur
        
    Returns:
        Préprocesseur TextPreprocessor
    """
    logger.info(f"Chargement du préprocesseur depuis {preprocessor_path}...")
    try:
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        return preprocessor
    except Exception as e:
        logger.error(f"Erreur lors du chargement du préprocesseur: {e}")
        return None

def explore_analogies(evaluator):
    """Test de quelques analogies classiques."""
    analogies = [
        ('man', 'woman', 'king', 'queen'),
        ('france', 'paris', 'italy', 'rome'),
        ('man', 'woman', 'uncle', 'aunt'),
        ('good', 'better', 'bad', 'worse')
    ]
    
    # Filtrage des analogies valides
    valid_analogies = []
    for a, b, c, d in analogies:
        if (a in evaluator.preprocessor.word2idx and
            b in evaluator.preprocessor.word2idx and
            c in evaluator.preprocessor.word2idx and
            d in evaluator.preprocessor.word2idx):
            valid_analogies.append((a, b, c, d))
    
    if not valid_analogies:
        logger.warning("Aucune des analogies prédéfinies n'est valide avec ce modèle.")
        return
    
    logger.info(f"Test de {len(valid_analogies)} analogies...")
    
    for a, b, c, d in valid_analogies:
        # Calcul de l'analogie: d ≈ c + (b - a)
        a_idx = evaluator.preprocessor.word2idx[a]
        b_idx = evaluator.preprocessor.word2idx[b]
        c_idx = evaluator.preprocessor.word2idx[c]
        
        a_vec = evaluator.embeddings[a_idx]
        b_vec = evaluator.embeddings[b_idx]
        c_vec = evaluator.embeddings[c_idx]
        
        target_vec = c_vec + (b_vec - a_vec)
        
        # Calcul des similarités
        similarities = np.dot(evaluator.embeddings, target_vec) / (
            np.linalg.norm(evaluator.embeddings, axis=1) * np.linalg.norm(target_vec)
        )
        
        # Exclusion des mots de l'analogie
        similarities[a_idx] = -np.inf
        similarities[b_idx] = -np.inf
        similarities[c_idx] = -np.inf
        
        # Top 5 résultats
        top_indices = similarities.argsort()[-5:][::-1]
        
        logger.info(f"Analogie: {a}:{b}::{c}:{d}")
        for idx in top_indices:
            word = evaluator.preprocessor.idx2word[idx]
            sim = similarities[idx]
            is_correct = "✓" if word == d else " "
            logger.info(f"  {is_correct} {word}: {sim:.4f}")

def interactive_exploration(evaluator):
    """Interface interactive pour explorer les embeddings."""
    print("=== Exploration interactive des embeddings Word2Vec ===")
    
    while True:
        print("\nOptions:")
        print("1. Trouver les mots similaires")
        print("2. Calculer la similarité entre deux mots")
        print("3. Résoudre une analogie (a:b::c:?)")
        print("4. Visualiser un ensemble de mots")
        print("0. Quitter")
        
        choice = input("\nVotre choix: ")
        
        if choice == '0':
            break
            
        elif choice == '1':
            word = input("Entrez un mot: ")
            if word not in evaluator.preprocessor.word2idx:
                print(f"Le mot '{word}' n'est pas dans le vocabulaire.")
                continue
                
            n = int(input("Nombre de mots similaires à afficher: "))
            similar_words = evaluator.most_similar(word, n=n)
            
            print(f"\nMots similaires à '{word}':")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
                
        elif choice == '2':
            word1 = input("Entrez le premier mot: ")
            word2 = input("Entrez le deuxième mot: ")
            
            if word1 not in evaluator.preprocessor.word2idx:
                print(f"Le mot '{word1}' n'est pas dans le vocabulaire.")
                continue
                
            if word2 not in evaluator.preprocessor.word2idx:
                print(f"Le mot '{word2}' n'est pas dans le vocabulaire.")
                continue
                
            similarity = evaluator.word_similarity(word1, word2)
            print(f"\nSimilarité entre '{word1}' et '{word2}': {similarity:.4f}")
            
        elif choice == '3':
            a = input("Entrez le mot a: ")
            b = input("Entrez le mot b: ")
            c = input("Entrez le mot c: ")
            
            if a not in evaluator.preprocessor.word2idx:
                print(f"Le mot '{a}' n'est pas dans le vocabulaire.")
                continue
                
            if b not in evaluator.preprocessor.word2idx:
                print(f"Le mot '{b}' n'est pas dans le vocabulaire.")
                continue
                
            if c not in evaluator.preprocessor.word2idx:
                print(f"Le mot '{c}' n'est pas dans le vocabulaire.")
                continue
                
            # Calcul de l'analogie: d ≈ c + (b - a)
            a_idx = evaluator.preprocessor.word2idx[a]
            b_idx = evaluator.preprocessor.word2idx[b]
            c_idx = evaluator.preprocessor.word2idx[c]
            
            a_vec = evaluator.embeddings[a_idx]
            b_vec = evaluator.embeddings[b_idx]
            c_vec = evaluator.embeddings[c_idx]
            
            target_vec = c_vec + (b_vec - a_vec)
            
            # Calcul des similarités
            similarities = np.dot(evaluator.embeddings, target_vec) / (
                np.linalg.norm(evaluator.embeddings, axis=1) * np.linalg.norm(target_vec)
            )
            
            # Exclusion des mots de l'analogie
            similarities[a_idx] = -np.inf
            similarities[b_idx] = -np.inf
            similarities[c_idx] = -np.inf
            
            # Top 5 résultats
            top_indices = similarities.argsort()[-5:][::-1]
            
            print(f"\nRésolution de l'analogie: {a}:{b}::{c}:?")
            print("  Résultats:")
            for idx in top_indices:
                word = evaluator.preprocessor.idx2word[idx]
                sim = similarities[idx]
                print(f"  {word}: {sim:.4f}")
                
        elif choice == '4':
            words_input = input("Entrez une liste de mots séparés par des espaces: ")
            words = words_input.split()
            
            # Filtrage des mots hors vocabulaire
            words = [word for word in words if word in evaluator.preprocessor.word2idx]
            
            if not words:
                print("Aucun des mots spécifiés n'est dans le vocabulaire.")
                continue
                
            method = input("Méthode de réduction (tsne/pca): ").lower()
            if method not in ['tsne', 'pca']:
                method = 'tsne'
                
            evaluator.visualize_embeddings(words=words, method=method)
            
        else:
            print("Option invalide, veuillez réessayer.")

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Exploration des modèles Word2Vec")
    
    parser.add_argument("--model_type", type=str, default="cbow",
                        choices=["cbow", "skipgram"],
                        help="Type de modèle à explorer")
    
    parser.add_argument("--interactive", action="store_true",
                        help="Activer le mode d'exploration interactive")
    
    args = parser.parse_args()
    
    # Vérification du dossier results
    if not os.path.exists('results'):
        logger.error("Dossier 'results' non trouvé. Exécutez d'abord word2vec_implementation.py")
        return
    
    # Chemins des fichiers
    model_path = f'results/{args.model_type}_model.pkl'
    preprocessor_path = f'results/{args.model_type}_preprocessor.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        logger.error(f"Modèle ou préprocesseur non trouvé pour {args.model_type}.")
        logger.error("Exécutez word2vec_implementation.py pour créer les modèles.")
        return
    
    # Chargement des fichiers
    model = load_model(model_path)
    preprocessor = load_preprocessor(preprocessor_path)
    
    if model is None or preprocessor is None:
        return
    
    # Création de l'évaluateur
    evaluator = Word2VecEvaluator(model, preprocessor)
    
    # Informations sur le modèle
    logger.info(f"Modèle: {args.model_type}")
    logger.info(f"Taille du vocabulaire: {preprocessor.vocab_size}")
    logger.info(f"Dimension des embeddings: {model.embedding_dim}")
    
    # Exploration des mots similaires
    for word in ['computer', 'science', 'technology', 'good', 'bad']:
        if word in preprocessor.word2idx:
            logger.info(f"Mots similaires à '{word}':")
            similar_words = evaluator.most_similar(word, n=5)
            for similar_word, similarity in similar_words:
                logger.info(f"  {similar_word}: {similarity:.4f}")
    
    # Test d'analogies
    explore_analogies(evaluator)
    
    # Mode interactif
    if args.interactive:
        interactive_exploration(evaluator)
    else:
        # Visualisation d'embeddings
        common_words = []
        word_counts = sorted(preprocessor.word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in word_counts[:50]:
            if word in preprocessor.word2idx and len(word) > 2:
                common_words.append(word)
                if len(common_words) >= 30:
                    break
        
        if len(common_words) < 5:
            logger.warning("Pas assez de mots pour la visualisation (minimum 5 requis)")
        else:
            # Ajustement de la perplexité
            perplexity = min(30, len(common_words) - 1)
            logger.info(f"Visualisation de {len(common_words)} embeddings (perplexité={perplexity})...")
            try:
                evaluator.visualize_embeddings(words=common_words, method='tsne')
            except Exception as e:
                logger.error(f"Erreur lors de la visualisation: {e}")
                logger.info("Tentative avec PCA...")
                evaluator.visualize_embeddings(words=common_words, method='pca')

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Erreur: {e}")
        import traceback
        logger.error(traceback.format_exc())