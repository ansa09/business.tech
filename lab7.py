"""
Simple Genetic Algorithm for Word Embedding Insertion
======================================================================
A (1+λ) Evolution Strategy for inserting new word embeddings into a trained
Skip-Gram model while preserving the existing embedding space structure.
"""

import torch
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision

from lab6 import SkipGramModel, find_similar_words
from lab2 import process_text_network

import unittest
import tempfile
import os


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_trained_model(model_path: str, vocab_size: int, 
                       embedding_dim: int, dropout: float) -> Tuple[torch.nn.Module, np.ndarray]:
    """Load trained Skip-Gram model and extract embeddings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        embeddings_tensor = model.get_embeddings()
        embeddings = (embeddings_tensor.cpu().numpy() if isinstance(embeddings_tensor, torch.Tensor) 
                     else embeddings_tensor).astype(np.float32)
    
    print(f"✓ Loaded model: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    return model, embeddings


def create_mappings(nodes: List[str]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, np.ndarray]]:
    """Create word-to-index and index-to-word mappings."""
    word_to_idx = {word: idx for idx, word in enumerate(nodes)}
    idx_to_word = {idx: word for idx, word in enumerate(nodes)}
    return word_to_idx, idx_to_word


def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute statistics needed for fitness evaluation."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'global_std': np.std(embeddings)
    }


def get_cifar100_vocabulary() -> List[str]:
    """Download CIFAR-100 and extract class names."""
    print("\nLoading CIFAR-100 vocabulary...")
    dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=True)
    print(f"✓ CIFAR-100 vocabulary loaded: {len(dataset.classes)} classes")
    return dataset.classes


def analyze_vocabulary_overlap(cifar_vocab: List[str], network_vocab: List[str]) -> List[str]:
    """Analyze overlap between CIFAR-100 and network vocabulary."""
    cifar_set, network_set = set(cifar_vocab), set(network_vocab)
    overlapping = sorted(list(cifar_set.intersection(network_set)))
    missing = sorted(list(cifar_set - network_set))
    
    print(f"\n{'='*70}")
    print("VOCABULARY OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"CIFAR-100 vocabulary: {len(cifar_set)} classes")
    print(f"Network vocabulary: {len(network_set)} words")
    print(f"Overlapping words: {len(overlapping)} ({len(overlapping)/len(cifar_set)*100:.1f}%)")
    print(f"Missing from network: {len(missing)}")
    if overlapping:
        print(f"\nFound: {', '.join(overlapping)}")
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
    print(f"{'='*70}\n")
    
    return missing


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

def extract_word_contexts(
    text_file: str,
    target_words: List[str],
    vocab_set: Set[str],
    window: int = 5
) -> Dict[str, Counter]:
    """
    Extract co-occurrence context statistics for target words from a text corpus.
    
    This function reads a corpus file line-by-line and tracks which words appear
    near specified target words. For each target word, it counts how many times
    each vocabulary word appears within a window around it.
    
    Args:
        text_file: Path to the corpus text file to analyze.
        target_words: List of words to extract contexts for.
        vocab_set: Set of valid vocabulary words (only count these as contexts).
        window: Number of words to look on each side of the target word.
    
    Returns:
        A dictionary mapping each target word to a Counter of context words and
        their frequencies.
        
    Example:
        >>> extract_word_contexts('corpus.txt', ['king', 'queen'], vocab, window=2)
        {'king': Counter({'royal': 5, 'crown': 3}), 
         'queen': Counter({'royal': 4, 'throne': 2})}
    
    Implementation guidelines:
    --------------------------
    1. Initialize a dictionary `{word: Counter()}` for each target word.
    2. Convert `target_words` to a set for fast lookup.
    3. Stream through the file line-by-line (efficient for large corpora).
    4. For each line:
        - Tokenize using lowercase alphabetic words (regex: r"\\b[a-z]+\\b").
        - For each token that matches a target word:
            * Extract up to `window` tokens on both sides.
            * Exclude the target word itself.
            * Retain only context words that appear in `vocab_set`.
            * Update the Counter for that target word.
    5. Handle edge cases: empty lines, start/end of token lists.
    6. Optionally print progress (e.g., every 50,000 lines) for user feedback.
    7. Return the dictionary of Counters.
    """
    
    # TODO: Initialize contexts dictionary with a Counter for each target word.
    # Initialize contexts dictionary with a Counter for each target word.
    contexts = {word: Counter() for word in target_words}
    target_set = set(target_words)

    with open(text_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0 and i > 0:
                print(f' ...processed {i:,} lines')

            tokens = re.findall(r'\b[a-z]+\b', line.lower())

            for idx, token in enumerate(tokens):
                if token in target_set:
                    start = max(0, idx - window)
                    end = min(len(tokens), idx + window + 1)

                    valid_context = [
                        t for t in tokens[start:idx] + tokens[idx+1:end]
                        if t in vocab_set
                    ]

                    contexts[token].update(valid_context)

    print(f"complete\n\ncontext statistics:")
    for word in target_words:
        print(f"{word:10s}: {sum(contexts[word].values()):6d} contexts, {len(contexts[word]):3d} unique words")

    return contexts






# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_fitness(
    vec: np.ndarray,
    word: str,
    ctx_vecs: Optional[np.ndarray],
    ctx_weights: Optional[np.ndarray],
    neg_vecs: np.ndarray,
    anchor_vecs: Optional[np.ndarray],
    stats_dict: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Compute a three-term fitness score for a candidate word embedding vector.
    
    This function evaluates how well a candidate vector fits the learned 
    embedding space by combining three complementary metrics:
    1. Corpus likelihood (how well it predicts observed contexts)
    2. Norm matching (how similar its magnitude is to typical embeddings)
    3. Anchor similarity (how similar it is to known reference words)
    
    Args:
        vec: Candidate embedding vector to evaluate.
        word: Target word (for reference, not used in computation).
        ctx_vecs: Context word vectors that co-occur with the target word.
                  Shape: (n_contexts, embedding_dim). May be None if no contexts.
        ctx_weights: Weights for each context (e.g., co-occurrence counts).
                     Shape: (n_contexts,). May be None if no contexts.
        neg_vecs: Negative sample vectors (words that don't co-occur).
                  Shape: (n_negatives, embedding_dim).
        anchor_vecs: Pre-normalized vectors of anchor words for comparison.
                     Shape: (n_anchors, embedding_dim). May be None.
        stats_dict: Dictionary containing embedding statistics:
                    - 'mean_norm': Average L2 norm of embeddings in the space
                    - 'std_norm': Standard deviation of embedding norms
                    - 'global_std': Global standard deviation (if needed)
        weights: Dictionary of weights for each fitness component:
                 - 'corpus': Weight for corpus likelihood term
                 - 'norm': Weight for norm matching term
                 - 'anchor': Weight for anchor similarity term
    
    Returns:
        Combined fitness score in the range [0, 1], where higher is better.
        
    Example:
        >>> vec = np.array([0.5, -0.3, 0.8, 0.1])
        >>> stats = {'mean_norm': 1.0, 'std_norm': 0.2, 'global_std': 0.5}
        >>> weights = {'corpus': 0.5, 'norm': 0.3, 'anchor': 0.2}
        >>> fitness = compute_fitness(vec, 'king', ctx_vecs, ctx_weights, 
        ...                           neg_vecs, anchor_vecs, stats, weights)
        >>> print(f"Fitness: {fitness:.4f}")
        Fitness: 0.7234
    
    Implementation guidelines:
    --------------------------
    Term 1 - Corpus Likelihood (L_corpus_norm):
        - For positive contexts: sum over ctx_weights * log(sigmoid(ctx_vecs · vec))
        - For negative samples: sum over log(sigmoid(-neg_vecs · vec))
        - Add small epsilon (1e-10) inside log for numerical stability
        - Normalize by total samples, then apply sigmoid to map to [0, 1]
        - Default to 0.5 if no samples available
        
    Term 2 - Norm Match (S_norm):
        - Compute L2 norm of the candidate vector
        - Use Gaussian similarity: exp(-((norm - mean_norm)² / (2 * std_norm²)))
        - This rewards vectors with norms close to the typical embedding norm
        
    Term 3 - Anchor Similarity (S_anchor):
        - Normalize the candidate vector (divide by its norm + epsilon)
        - Compute dot products with all anchor vectors (they're pre-normalized)
        - Take the mean similarity across all anchors
        - Default to 0.5 if no anchors provided
        
    Final score:
        - Weighted sum: weights['corpus'] * L_corpus_norm + 
                       weights['norm'] * S_norm + 
                       weights['anchor'] * S_anchor
    
    Notes:
        - Handle None values for optional parameters (ctx_vecs, ctx_weights, anchor_vecs)
        - Use vectorized NumPy operations for efficiency
        - Add small epsilon values to prevent division by zero
    """
        
    epsilon = 1e-10
    
    # ========================================================================
    # Term 1: Corpus Likelihood (L_corpus_norm)
    # ========================================================================
    if ctx_vecs is not None and ctx_weights is not None and len(ctx_vecs) > 0:
        # Positive contexts: weighted sum of log(sigmoid(ctx_vecs · vec))
        pos_dots = np.dot(ctx_vecs, vec)  # Shape: (n_contexts,)
        pos_log_probs = np.log(sigmoid(pos_dots) + epsilon)
        pos_term = np.sum(ctx_weights * pos_log_probs)
        
        # Negative samples: sum of log(sigmoid(-neg_vecs · vec))
        neg_dots = np.dot(neg_vecs, vec)  # Shape: (n_negatives,)
        neg_log_probs = np.log(sigmoid(-neg_dots) + epsilon)
        neg_term = np.sum(neg_log_probs)
        
        # Total likelihood
        total_likelihood = pos_term + neg_term
        
        # Normalize by total number of samples
        total_samples = len(ctx_vecs) + len(neg_vecs)
        normalized_likelihood = total_likelihood / total_samples
        
        # Map to [0, 1] using sigmoid
        L_corpus_norm = sigmoid(normalized_likelihood)
    else:
        # No contexts available, default to neutral score
        L_corpus_norm = 0.5
    
    # ========================================================================
    # Term 2: Norm Match (S_norm)
    # ========================================================================
    vec_norm = np.linalg.norm(vec)
    mean_norm = stats_dict['mean_norm']
    std_norm = stats_dict['std_norm']
    
    # Gaussian similarity centered at mean_norm
    S_norm = np.exp(-((vec_norm - mean_norm) ** 2) / (2 * std_norm ** 2))
    
    # ========================================================================
    # Term 3: Anchor Similarity (S_anchor)
    # ========================================================================
    if anchor_vecs is not None and len(anchor_vecs) > 0:
        # Normalize the candidate vector
        vec_normalized = vec / (np.linalg.norm(vec) + epsilon)
        
        # Compute dot products with pre-normalized anchor vectors
        similarities = np.dot(anchor_vecs, vec_normalized)  # Shape: (n_anchors,)
        
        # Take mean similarity
        S_anchor = np.mean(similarities)
    else:
        # No anchors available, default to neutral score
        S_anchor = 0.5
    
    # ========================================================================
    # Final Weighted Score
    # ========================================================================
    fitness = (weights['corpus'] * L_corpus_norm + 
               weights['norm'] * S_norm + 
               weights['anchor'] * S_anchor)
    
    return fitness
    #return 0.0


# ============================================================================
# GENETIC ALGORITHM (1+λ) EVOLUTION STRATEGY
# ============================================================================

def initialize_embedding(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int]
) -> np.ndarray:
    """
    Initialize an embedding vector for a word using corpus bootstrap.
    
    This function creates an initial embedding by computing a weighted average
    of the embeddings of words that frequently co-occur with the target word.
    This provides a data-driven starting point that places the new word near
    semantically related words in the embedding space.
    
    Args:
        word: Target word to initialize an embedding for.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
    
    Returns:
        Initial embedding vector for the word. Shape: (embedding_dim,).
        
    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30, 'castle': 20})}
        >>> embeddings = np.random.randn(1000, 300)  # 1000 words, 300 dims
        >>> word_to_idx = {'queen': 0, 'royal': 1, 'castle': 2, ...}
        >>> vec = initialize_embedding('king', contexts, embeddings, word_to_idx)
        >>> vec.shape
        (300,)
    
    Implementation guidelines:
    --------------------------
    1. Handle the no-context case:
       - If the word has no contexts (empty Counter), return the mean of all
         embeddings as a neutral starting point
    
    2. Get top context words:
       - Extract the top 20 most frequent context words using Counter.most_common()
       - This focuses on the strongest statistical relationships
    
    3. Compute weighted average:
       - Calculate the total weight (sum of all counts)
       - For each context word that exists in word_to_idx:
           * Get its embedding vector
           * Weight it by (count / weight_sum)
           * Add to running sum
    
    4. Validate the result:
       - Check if the resulting vector has non-zero norm
       - If zero (e.g., no valid context words found), fall back to mean embedding
    
    Notes:
        - Some context words may not be in word_to_idx; skip these
        - The weighted average naturally places the new word near its contexts
        - Using top 20 contexts balances informativeness with noise reduction
    """

    if not contexts[word]:
        return np.mean(embeddings, axis=0)

    top_contexts = contexts[word].most_common(20)
    weight_sum = sum(count for _, count in top_contexts)
    if weight_sum == 0:
        return np.mean(embeddings, axis=0)
    vec_sum = np.zeros(embeddings.shape[1])
    #vec_sum = sum((count/weight_sum) * embeddings[word_to_idx[w]] for w, count) 

    for ctx_word, count in top_contexts:
        if ctx_word in word_to_idx:
            vec_sum += (count / weight_sum) * embeddings[word_to_idx[ctx_word]]
    
    # Validate the result
    if np.linalg.norm(vec_sum) > 0:
        return vec_sum
    else:
        return np.mean(embeddings, axis=0)
                  
                  #in top_contexts if w in word_to_idx)
    #return vec_sum if np.linalg.norm(vec_sum) > 0 else np.mean(embedding, axis = 0)


def precompute_fitness_vectors(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    vocab_list: List[str],
    anchors: Dict[str, List[str]],
    num_negatives: int = 15
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Precompute all vectors needed for fitness evaluation.
    
    This function extracts and prepares the three types of vectors used in
    fitness computation: positive context vectors, negative sample vectors,
    and anchor vectors. Precomputing these vectors once improves efficiency
    when evaluating fitness multiple times during optimization.
    
    Args:
        word: Target word being optimized.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
        vocab_list: List of all vocabulary words (for negative sampling).
        anchors: Dictionary mapping words to lists of semantically related anchor words.
        num_negatives: Number of negative samples to draw (default: 15).
    
    Returns:
        Tuple of (ctx_vecs, ctx_weights, neg_vecs, anchor_vecs):
        - ctx_vecs: Context word embeddings. Shape: (n_contexts, dim) or None.
        - ctx_weights: Normalized context weights. Shape: (n_contexts,) or None.
        - neg_vecs: Negative sample embeddings. Shape: (num_negatives, dim).
        - anchor_vecs: Normalized anchor embeddings. Shape: (n_anchors, dim) or None.
        
    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30})}
        >>> anchors = {'king': ['queen', 'monarch', 'ruler']}
        >>> ctx_v, ctx_w, neg_v, anc_v = precompute_fitness_vectors(
        ...     'king', contexts, embeddings, word_to_idx, vocab_list, anchors
        ... )
        >>> ctx_v.shape  # Positive contexts
        (2, 300)
        >>> neg_v.shape  # Negative samples
        (15, 300)
    
    Implementation guidelines:
    --------------------------
    Part 1 - Positive Context Vectors:
        - Initialize ctx_vecs and ctx_weights to None (for no-context case)
        - If the word has contexts:
            * Iterate through contexts[word].items()
            * For each context word that exists in word_to_idx:
              - Collect its embedding vector
              - Collect its count
            * If any valid contexts found:
              - Convert lists to numpy arrays
              - Normalize weights to sum to 1.0
    
    Part 2 - Negative Sample Vectors:
        - Randomly sample num_negatives words from vocab_list (without replacement)
        - Look up their embeddings and stack into an array
        - Shape should be (num_negatives, embedding_dim)
    
    Part 3 - Anchor Vectors:
        - Initialize anchor_vecs to None (for no-anchor case)
        - If the word has anchors defined:
            * Filter to only anchors that exist in word_to_idx
            * If any valid anchors found:
              - Collect their embeddings into an array
              - Normalize each vector to unit length (L2 norm = 1)
              - Use np.linalg.norm with axis=1, keepdims=True
              - Add small epsilon (1e-10) to prevent division by zero
    
    Notes:
        - Handle missing words gracefully (skip if not in word_to_idx)
        - Return None for optional components if no valid data available
        - Negative samples should be random to avoid bias
        - Anchor normalization enables direct cosine similarity via dot product
    """
     
    epsilon = 1e-10
    
    # ========================================================================
    # Part 1: Positive Context Vectors
    # ========================================================================
    ctx_vecs = None
    ctx_weights = None
    
    if word in contexts and contexts[word]:
        # Collect valid context vectors and their counts
        context_vectors = []
        context_counts = []
        
        for ctx_word, count in contexts[word].items():
            if ctx_word in word_to_idx:
                idx = word_to_idx[ctx_word]
                context_vectors.append(embeddings[idx])
                context_counts.append(count)
        
        # If we found any valid contexts, convert to arrays and normalize weights
        if context_vectors:
            ctx_vecs = np.array(context_vectors)  # Shape: (n_contexts, embedding_dim)
            ctx_weights = np.array(context_counts, dtype=np.float32)
            
            # Normalize weights to sum to 1.0
            weight_sum = np.sum(ctx_weights)
            if weight_sum > 0:
                ctx_weights = ctx_weights / weight_sum
    
    # ========================================================================
    # Part 2: Negative Sample Vectors
    # ========================================================================
    # Randomly sample num_negatives words from vocabulary
    num_to_sample = min(num_negatives, len(vocab_list))
    sampled_indices = np.random.choice(len(vocab_list), size=num_to_sample, replace=False)
    
    # Get embeddings for sampled negative words
    neg_vecs = np.array([embeddings[word_to_idx[vocab_list[i]]] 
                         for i in sampled_indices])  # Shape: (num_negatives, embedding_dim)
    
    # ========================================================================
    # Part 3: Anchor Vectors
    # ========================================================================
    anchor_vecs = None
    
    if word in anchors and anchors[word]:
        # Filter to only anchors that exist in vocabulary
        valid_anchor_vectors = []
        
        for anchor_word in anchors[word]:
            if anchor_word in word_to_idx:
                idx = word_to_idx[anchor_word]
                valid_anchor_vectors.append(embeddings[idx])
        
        # If we found any valid anchors, normalize them
        if valid_anchor_vectors:
            anchor_vecs = np.array(valid_anchor_vectors)  # Shape: (n_anchors, embedding_dim)
            
            # Normalize each anchor vector to unit length (L2 norm = 1)
            norms = np.linalg.norm(anchor_vecs, axis=1, keepdims=True)
            anchor_vecs = anchor_vecs / (norms + epsilon)
    
    return ctx_vecs, ctx_weights, neg_vecs, anchor_vecs



def evolve_embedding(word: str, contexts: Dict[str, Counter], 
                    embeddings: np.ndarray, word_to_idx: Dict[str, int],
                    vocab_list: List[str], stats_dict: Dict[str, float],
                    anchors: Dict[str, List[str]], config: Dict) -> np.ndarray:
    """
    Evolve a single word embedding using (1+λ) Evolution Strategy.
    
    Args:
        word: Target word to insert
        contexts: Context word counts for all target words
        embeddings: Existing embedding matrix
        word_to_idx: Word to index mapping
        vocab_list: List of vocabulary words
        stats_dict: Embedding statistics
        anchors: Anchor words for semantic guidance
        config: Configuration dictionary
    
    Returns:
        Optimized embedding vector
    """
    print(f"\n  Evolving: '{word}'", end='')
    
    dim = embeddings.shape[1]
    mutation_sigma = config['ga_mutation_factor'] * stats_dict['global_std']
    
    # Initialize
    best_vec = initialize_embedding(word, contexts, embeddings, word_to_idx)
    
    # Precompute vectors
    ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
        word, contexts, embeddings, word_to_idx, vocab_list, anchors
    )
    
    # Initial fitness
    best_fit = compute_fitness(best_vec, word, ctx_vecs, ctx_weights, neg_vecs, 
                               anchor_vecs, stats_dict, config['fitness_weights'])
    
    # Evolution loop
    for gen in range(config['ga_generations']):
        # Generate offspring and evaluate
        population = best_vec + np.random.normal(0, mutation_sigma, (config['ga_pop_size'], dim))
        all_candidates = np.vstack([best_vec, population])
        
        fitness_scores = [compute_fitness(vec, word, ctx_vecs, ctx_weights, neg_vecs, 
                                         anchor_vecs, stats_dict, config['fitness_weights'])
                         for vec in all_candidates]
        
        # Select best
        best_idx = np.argmax(fitness_scores)
        best_vec = all_candidates[best_idx].copy()
        best_fit = fitness_scores[best_idx]
        
        if gen % 50 == 0:
            print(f" G{gen}={best_fit:.4f}", end='')
    
    print(f" ✓ Final={best_fit:.4f}")
    return best_vec


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_with_inserted_words(nodes: List[str], embeddings: np.ndarray, 
                                  inserted_words: List[str],
                                  output_file: str = "embeddings_with_inserted.png",
                                  sample_size: int = 500):
    """Create t-SNE visualization highlighting inserted words."""
    print("\nGenerating t-SNE visualization with inserted words...")
    
    num_original = len(nodes) - len(inserted_words)
    inserted_indices = set(range(num_original, len(nodes)))
    
    # Sample: prioritize inserted words + random original
    if len(nodes) > sample_size:
        sample_indices = list(inserted_indices) + list(np.random.choice(
            num_original, min(sample_size - len(inserted_words), num_original), replace=False))
    else:
        sample_indices = list(range(len(nodes)))
    
    selected_embeddings = embeddings[sample_indices]
    selected_nodes = [nodes[i] for i in sample_indices]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_indices)-1))
    projection = tsne.fit_transform(selected_embeddings)
    
    # Plot
    plt.figure(figsize=(14, 14))
    
    for i in range(len(projection)):
        is_inserted = sample_indices[i] in inserted_indices
        plt.scatter(projection[i, 0], projection[i, 1], 
                   s=200 if is_inserted else 40,
                   alpha=1.0 if is_inserted else 0.6,
                   c='red' if is_inserted else 'steelblue')
        plt.annotate(selected_nodes[i], (projection[i, 0], projection[i, 1]), 
                    fontsize=11 if is_inserted else 9,
                    alpha=1.0 if is_inserted else 0.8,
                    fontweight='bold' if is_inserted else 'normal')
    
    plt.title(f"t-SNE Visualization: {len(sample_indices)} Words "
              f"({sum(1 for i in sample_indices if i in inserted_indices)} Inserted)",
              fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved t-SNE to {output_file}")
    plt.show()


def run_sanity_checks(model: torch.nn.Module, embeddings: np.ndarray, 
                     nodes: List[str], word_to_idx: Dict[str, int]):
    """Run comprehensive sanity checks on loaded model and embeddings."""
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)
    
    print(f"\n1. Model Configuration:")
    print(f"   Training mode: {model.training}")
    print(f"   Device: {next(model.parameters()).device}")
    
    print(f"\n2. Embedding Quality:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}")
    print(f"   Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}")
    print(f"   Contains NaN: {np.isnan(embeddings).any()}, Contains Inf: {np.isinf(embeddings).any()}")
    
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n3. Embedding Norms:")
    print(f"   Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
    print(f"   Range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    print(f"\n4. Vocabulary Test:")
    for test_word in ['man', 'woman', 'dog', 'car', 'blue']:
        if test_word in word_to_idx:
            word_idx = word_to_idx[test_word]
            print(f"   '{test_word:10s}' → idx={word_idx:4d}, norm={np.linalg.norm(embeddings[word_idx]):.4f}")
            similar = find_similar_words(test_word, nodes, embeddings, top_k=5)
            if similar:
                print(f"      Similar: {', '.join([f'{w}({s:.3f})' for w, s in similar])}")
    
    print("\n" + "="*70)
    print("✓ SANITY CHECKS COMPLETE")
    print("="*70)


# ============================================================================
# UNIT TESTS
# ============================================================================

class TestDataLoading(unittest.TestCase):
    """Test data loading and preparation functions."""
    
    def test_create_mappings(self):
        """Test word-to-index and index-to-word mappings."""
        nodes = ['cat', 'dog', 'bird', 'fish']
        word_to_idx, idx_to_word = create_mappings(nodes)
        
        self.assertEqual(word_to_idx['cat'], 0)
        self.assertEqual(word_to_idx['dog'], 1)
        self.assertEqual(idx_to_word[0], 'cat')
        self.assertEqual(idx_to_word[1], 'dog')
        self.assertEqual(len(word_to_idx), 4)
        self.assertEqual(len(idx_to_word), 4)
    
    def test_compute_embedding_stats(self):
        """Test embedding statistics computation."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ])
        
        stats = compute_embedding_stats(embeddings)
        
        self.assertAlmostEqual(stats['mean_norm'], 2.0, places=5)
        self.assertIn('mean_norm', stats)
        self.assertIn('std_norm', stats)
        self.assertIn('global_std', stats)
        self.assertGreater(stats['mean_norm'], 0)
        self.assertGreater(stats['std_norm'], 0)
        self.assertGreater(stats['global_std'], 0)


class TestContextExtraction(unittest.TestCase):
    """Test context extraction from corpus."""
    
    def setUp(self):
        """Create a temporary test corpus file."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        self.temp_file.write("the dog runs fast\n")
        self.temp_file.write("the cat sleeps peacefully\n")
        self.temp_file.write("the dog barks loudly\n")
        self.temp_file.write("a dog is a good pet\n")
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up temporary file."""
        os.unlink(self.temp_file.name)
    
    def test_extract_word_contexts(self):
        """Test context extraction with known corpus."""
        target_words = ['dog', 'cat']
        vocab_set = {'the', 'dog', 'cat', 'runs', 'fast', 'sleeps', 
                     'peacefully', 'barks', 'loudly', 'is', 'a', 'good', 'pet'}
        
        contexts = extract_word_contexts(
            self.temp_file.name, 
            target_words, 
            vocab_set, 
            window=2
        )
        
        self.assertIn('dog', contexts)
        self.assertIn('cat', contexts)
        self.assertGreater(len(contexts['dog']), 0)
        self.assertIn('the', contexts['dog'])
        self.assertGreater(len(contexts['cat']), 0)
        self.assertIn('the', contexts['cat'])
        
        for word in target_words:
            total_contexts = sum(contexts[word].values())
            self.assertGreater(total_contexts, 0)


class TestSigmoidFunction(unittest.TestCase):
    """Test numerically stable sigmoid implementation."""
    
    def test_sigmoid_positive(self):
        """Test sigmoid for positive values."""
        x = np.array([0.0, 1.0, 2.0, 10.0])
        result = sigmoid(x)
        
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        self.assertAlmostEqual(result[0], 0.5, places=5)
        self.assertAlmostEqual(result[1], 1/(1+np.exp(-1)), places=5)
    
    def test_sigmoid_negative(self):
        """Test sigmoid for negative values."""
        x = np.array([-1.0, -2.0, -10.0])
        result = sigmoid(x)
        
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        self.assertTrue(np.all(result < 0.5))
    
    def test_sigmoid_extreme_values(self):
        """Test sigmoid doesn't overflow/underflow."""
        x = np.array([-100.0, 100.0, 1000.0, -1000.0])
        result = sigmoid(x)
        
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        self.assertLess(result[0], 0.01)
        self.assertGreater(result[1], 0.99)


class TestFitnessFunction(unittest.TestCase):
    """Test the fitness computation."""
    
    def setUp(self):
        """Set up test data."""
        self.vec = np.array([1.0, 0.0, 0.0])
        self.ctx_vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.ctx_weights = np.array([0.6, 0.4])
        self.neg_vecs = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        self.anchor_vecs = np.array([[1.0, 0.0, 0.0]])
        self.stats_dict = {'mean_norm': 1.0, 'std_norm': 0.1, 'global_std': 0.5}
        self.weights = {'corpus': 0.5, 'norm': 0.3, 'anchor': 0.2}
    
    def test_fitness_output_range(self):
        """Test that fitness is in valid range [0, 1]."""
        fitness = compute_fitness(
            self.vec, 'test', self.ctx_vecs, self.ctx_weights,
            self.neg_vecs, self.anchor_vecs, self.stats_dict, self.weights
        )
        
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
    
    def test_fitness_with_no_contexts(self):
        """Test fitness defaults correctly with no contexts."""
        fitness = compute_fitness(
            self.vec, 'test', None, None,
            self.neg_vecs, None, self.stats_dict, self.weights
        )
        
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)
    
    def test_fitness_norm_term(self):
        """Test that norm matching works correctly."""
        perfect_vec = np.array([1.0, 0.0, 0.0])
        fitness_perfect = compute_fitness(
            perfect_vec, 'test', self.ctx_vecs, self.ctx_weights,
            self.neg_vecs, self.anchor_vecs, self.stats_dict, self.weights
        )
        
        bad_vec = np.array([10.0, 0.0, 0.0])
        fitness_bad = compute_fitness(
            bad_vec, 'test', self.ctx_vecs, self.ctx_weights,
            self.neg_vecs, self.anchor_vecs, self.stats_dict, self.weights
        )
        
        self.assertGreater(fitness_perfect, fitness_bad)


class TestInitializeEmbedding(unittest.TestCase):
    """Test embedding initialization."""
    
    def setUp(self):
        """Set up test embeddings and contexts."""
        self.embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ])
        self.word_to_idx = {'cat': 0, 'dog': 1, 'bird': 2, 'fish': 3}
        self.contexts = {
            'wolf': Counter({'dog': 10, 'cat': 5, 'bird': 2}),
            'empty': Counter()
        }
    
    def test_initialize_with_contexts(self):
        """Test initialization with valid contexts."""
        vec = initialize_embedding('wolf', self.contexts, self.embeddings, self.word_to_idx)
        
        self.assertEqual(vec.shape, (3,))
        self.assertGreater(np.linalg.norm(vec), 0)
        self.assertFalse(np.any(np.isnan(vec)))
        self.assertFalse(np.any(np.isinf(vec)))
    
    def test_initialize_without_contexts(self):
        """Test initialization falls back to mean for empty contexts."""
        vec = initialize_embedding('empty', self.contexts, self.embeddings, self.word_to_idx)
        
        expected = np.mean(self.embeddings, axis=0)
        np.testing.assert_array_almost_equal(vec, expected)
    
    def test_initialize_output_dimension(self):
        """Test that output has correct dimensionality."""
        vec = initialize_embedding('wolf', self.contexts, self.embeddings, self.word_to_idx)
        self.assertEqual(len(vec), self.embeddings.shape[1])


class TestPrecomputeFitnessVectors(unittest.TestCase):
    """Test fitness vector precomputation."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.embeddings = np.random.randn(10, 5)
        self.word_to_idx = {f'word{i}': i for i in range(10)}
        self.vocab_list = [f'word{i}' for i in range(10)]
        self.contexts = {
            'target': Counter({'word0': 5, 'word1': 3, 'word2': 2})
        }
        self.anchors = {
            'target': ['word3', 'word4']
        }
    
    def test_precompute_returns_correct_types(self):
        """Test that precompute returns correct data types."""
        ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
            'target', self.contexts, self.embeddings, self.word_to_idx,
            self.vocab_list, self.anchors, num_negatives=5
        )
        
        self.assertIsInstance(ctx_vecs, np.ndarray)
        self.assertIsInstance(ctx_weights, np.ndarray)
        self.assertIsInstance(neg_vecs, np.ndarray)
        self.assertIsInstance(anchor_vecs, np.ndarray)
    
    def test_precompute_weights_normalized(self):
        """Test that context weights sum to 1."""
        ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
            'target', self.contexts, self.embeddings, self.word_to_idx,
            self.vocab_list, self.anchors, num_negatives=5
        )
        
        self.assertAlmostEqual(np.sum(ctx_weights), 1.0, places=5)
    
    def test_precompute_anchors_normalized(self):
        """Test that anchor vectors are unit length."""
        ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
            'target', self.contexts, self.embeddings, self.word_to_idx,
            self.vocab_list, self.anchors, num_negatives=5
        )
        
        norms = np.linalg.norm(anchor_vecs, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(norms)), decimal=5)
    
    def test_precompute_no_contexts(self):
        """Test handling of words with no contexts."""
        contexts_empty = {'target': Counter()}
        
        ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
            'target', contexts_empty, self.embeddings, self.word_to_idx,
            self.vocab_list, self.anchors, num_negatives=5
        )
        
        self.assertIsNone(ctx_vecs)
        self.assertIsNone(ctx_weights)
        self.assertIsNotNone(neg_vecs)


class TestEvolveEmbedding(unittest.TestCase):
    """Test the evolution process."""
    
    def setUp(self):
        """Set up minimal test scenario."""
        np.random.seed(42)
        self.embeddings = np.random.randn(20, 10)
        self.word_to_idx = {f'word{i}': i for i in range(20)}
        self.vocab_list = [f'word{i}' for i in range(20)]
        self.contexts = {
            'target': Counter({'word0': 10, 'word1': 5, 'word2': 3})
        }
        self.anchors = {
            'target': ['word3', 'word4']
        }
        self.stats_dict = compute_embedding_stats(self.embeddings)
        self.config = {
            'ga_pop_size': 10,
            'ga_generations': 5,
            'ga_mutation_factor': 0.1,
            'fitness_weights': {'corpus': 0.5, 'norm': 0.3, 'anchor': 0.2}
        }
    
    def test_evolve_returns_correct_shape(self):
        """Test that evolved embedding has correct shape."""
        evolved_vec = evolve_embedding(
            'target', self.contexts, self.embeddings, self.word_to_idx,
            self.vocab_list, self.stats_dict, self.anchors, self.config
        )
        
        self.assertEqual(evolved_vec.shape, (10,))
    
    def test_evolve_returns_valid_vector(self):
        """Test that evolved embedding is valid."""
        evolved_vec = evolve_embedding(
            'target', self.contexts, self.embeddings, self.word_to_idx,
            self.vocab_list, self.stats_dict, self.anchors, self.config
        )
        
        self.assertFalse(np.any(np.isnan(evolved_vec)))
        self.assertFalse(np.any(np.isinf(evolved_vec)))
        self.assertGreater(np.linalg.norm(evolved_vec), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_small(self):
        """Test complete pipeline with small synthetic data."""
        np.random.seed(42)
        vocab_size = 50
        embedding_dim = 20
        embeddings = np.random.randn(vocab_size, embedding_dim)
        
        # Don't normalize to exact same norm - keep some variance
        # Just scale to reasonable range
        embeddings = embeddings * 3.0 / np.mean(np.linalg.norm(embeddings, axis=1))
        
        nodes = [f'word{i}' for i in range(vocab_size)]
        word_to_idx, idx_to_word = create_mappings(nodes)
        stats_dict = compute_embedding_stats(embeddings)
        
        contexts = {
            'newword': Counter({'word0': 10, 'word1': 8, 'word2': 5})
        }
        anchors = {
            'newword': ['word3', 'word4', 'word5']
        }
        config = {
            'ga_pop_size': 20,
            'ga_generations': 10,
            'ga_mutation_factor': 0.1,
            'fitness_weights': {'corpus': 0.5, 'norm': 0.3, 'anchor': 0.2}
        }
        
        new_embedding = evolve_embedding(
            'newword', contexts, embeddings, word_to_idx,
            nodes, stats_dict, anchors, config
        )
        
        # Verify output shape and validity
        self.assertEqual(new_embedding.shape, (embedding_dim,))
        self.assertFalse(np.any(np.isnan(new_embedding)))
        self.assertFalse(np.any(np.isinf(new_embedding)))
        
        # Verify norm is positive and reasonable (not checking strict bounds)
        new_norm = np.linalg.norm(new_embedding)
        self.assertGreater(new_norm, 0)
        self.assertLess(new_norm, 100)  # Sanity check - not absurdly large

# ============================================================================
# TEST RUNNER
# ============================================================================

def run_tests():
    """Run all unit tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestContextExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestSigmoidFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestFitnessFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestInitializeEmbedding))
    suite.addTests(loader.loadTestsFromTestCase(TestPrecomputeFitnessVectors))
    suite.addTests(loader.loadTestsFromTestCase(TestEvolveEmbedding))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)  
