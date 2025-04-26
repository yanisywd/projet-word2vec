# Implémentation from scratch de Word2Vec

## Word2Vec Implementation

### Project Files

**word2vec_implementation.py:**
- Implements Word2Vec models (CBOW and Skip-gram) from scratch
- Includes text preprocessing, vocabulary building
- Implements negative sampling and word subsampling
- Provides training pipeline for 20newsgroups and IMDB datasets
- Includes text classification functionality

**word2vec_explorer.py:**
- Loads trained models from word2vec_implementation.py
- Visualizes word embeddings using t-SNE and PCA
- Provides interactive exploration of word similarities
- Tests word analogies
- Supports command-line arguments for model selection

**word2vec_evaluation.py:**
- Contains functions for evaluating the performance of Word2Vec models
- Includes metrics for accuracy, loss, and similarity
- Provides detailed reports on model performance
- Supports comparison between CBOW and Skip-gram models
- Generates visualizations for evaluation results

**word2vec_classifier.py:**
- Implements neural network classification using custom Word2Vec embeddings
- Compares classification performance with pre-trained GloVe embeddings
- Processes and vectorizes IMDB and 20newsgroups datasets
- Creates and trains ANN models with optimized architectures
- Evaluates models using accuracy, precision, recall and F1-score
- Generates comparative visualizations between embedding models
- Provides comprehensive performance analysis and reporting
