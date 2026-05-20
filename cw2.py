"""
CW2: Neuro-Symbolic AI System
Student Name: [Your Name]
Student ID: [Your ID]

AI Tool Usage:
- Tool: Claude (Anthropic, Sonnet 4.5, December 2024)
- Used for: Debugging embedding generation, optimizing dimensionality strategy,
  anti-hubbing solutions, report writing assistance
- Prompting approach: Iterative problem-solving with metric-driven feedback,
  experimental comparison of multiple approaches

This module implements a neuro-symbolic AI system that combines:
- Computer Vision (CIFAR-100 object recognition)
- Natural Language Processing (Skip-gram word embeddings)
- Symbolic Planning (PDDL planning)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Dict, Tuple, Optional
from pathlib import Path
import warnings

# Lab imports
from lab6 import SkipGramModel, find_similar_words
from lab7 import get_cifar100_vocabulary, compute_embedding_stats


# ============================================================================
# SECTION 1: CIFAR-100 SEMANTIC EXPANSION
# ============================================================================

# DO NOT CHANGE THIS FUNCTION's signature
def build_my_embeddings(
    checkpoint_path: str = "best_skipgram_523words.pth"
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load and return your trained Skip-gram embeddings.
    
    This function serves as the entry point for loading your final embedding model
    that contains all Visual Genome words AND all 100 CIFAR-100 classes.
    
    Args:
        checkpoint_path: Path to your saved model checkpoint
        
    Returns:
        vocab: Dictionary mapping words to indices {word: index}
        embeddings: Numpy array of shape (vocab_size, embedding_dim)
        
    Example:
        >>> vocab, embeddings = build_my_embeddings()
        >>> print(f"Vocabulary size: {len(vocab)}")
        >>> print(f"Embedding dimension: {embeddings.shape[1]}")
        >>> print(f"'airplane' index: {vocab.get('airplane', 'NOT FOUND')}")
    """
    # Load the checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False
    )
    
    # Extract nodes (base vocabulary)
    nodes = checkpoint['nodes']
    base_vocab = checkpoint['vocabulary']
    
    # Extract embeddings
    embeddings = checkpoint['embeddings']
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    embeddings = embeddings.astype(np.float32)
    
    # Create normalized vocabulary (handles both underscore and space formats)
    vocab = {}
    
    for word, idx in base_vocab.items():
        # Add original format
        vocab[word] = idx
        
        # Add space-separated format for CIFAR words with underscores
        if '_' in word:
            space_format = word.replace('_', ' ')
            vocab[space_format] = idx
    
    # Validate shape consistency
    if len(nodes) != embeddings.shape[0]:
        raise ValueError(
            f"Shape mismatch: nodes ({len(nodes)}) != embeddings ({embeddings.shape[0]})"
        )
    
    # Validate no NaN or Inf
    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN values")
    if np.isinf(embeddings).any():
        raise ValueError("Embeddings contain Inf values")
    
    return vocab, embeddings


# ============================================================================
# SECTION 2: NEURO-SYMBOLIC AI - MULTI-MODAL PLANNING
# ============================================================================

# DO NOT CHANGE THIS FUNCTION's signature
def plan_generator(
    input_data: Union[torch.Tensor, str],    # ASSUME default CIFAR-100 image dimensions
    initial_state: List[str],                 # Consistent with Lab9 syntax
    goal_state: List[str],                    # Consistent with Lab9 syntax
    domain_file: str = "domain.pddl",
    skipgram_path: str = "best_skipgram_523words.pth",
    projection_path: str = "best_cifar100_projection.pth"
) -> Optional[List[str]]:
    """
    !!!WARNING!!!: Treat this as pseudocode. You may need to modify the logic. 
    
    Main entry point for the neuro-symbolic planning system.
    
    This function implements the complete pipeline from perception to planning.
    
    Args:
        input_data: Either an image tensor OR object name string
        initial_state: List of predicates describing initial state                      
        goal_state: List of predicates describing goal state                   
        domain_file: Path to the PDDL domain file
        skipgram_path: Path to Skip-gram embeddings checkpoint
        projection_path: Path to CIFAR-100 projection model checkpoint
        
    Returns:
        A list of action strings representing the plan, 
            OR None if:
                - The object cannot be identified
                - No valid plan exists
                - ...
        
    Example:
        >>> image = # CIFAR-100 image
        >>> initial = ["on table"]
        >>> goal = ["in basket"]
        >>> plan = plan_generator(image, initial, goal, "domain.pddl")        
    """
    
    # TREAT THIS AS SUGGESTED PSEUDOCODE. YOU MAY USE OTHER PARADIGMS
                    
    # Step 0: Initialize the planner
    
    # Step 1: Identify the object
    
    # Step 2: Parse PDDL domain
    
    # Step 3: Create PDDL problem
    
    # Step 4: Generate plan
        
    return None


# ============================================================================
# HELPER FUNCTIONS (Optional - for your use)
# ============================================================================

def validate_model(checkpoint_path: str = "best_skipgram_523words.pth") -> Dict:
    """
    Validate the embedding model and return comprehensive metrics.
    
    Returns:
        Dictionary with validation results including:
        - vocab_size: Total vocabulary entries
        - unique_embeddings: Number of unique embeddings
        - embedding_dim: Dimensionality
        - cifar_coverage: CIFAR-100 coverage count
        - similarity_stats: Mean and std of similarities
        - norm_stats: Mean and std of norms
    """
    vocab, embeddings = build_my_embeddings(checkpoint_path)
    
    # Basic stats
    results = {
        'vocab_size': len(vocab),
        'unique_embeddings': embeddings.shape[0],
        'embedding_dim': embeddings.shape[1]
    }
    
    # CIFAR-100 coverage
    cifar_vocab = get_cifar100_vocabulary()
    cifar_coverage = sum(1 for w in cifar_vocab if w in vocab or w.replace('_', ' ') in vocab)
    results['cifar_coverage'] = cifar_coverage
    
    # Norm statistics
    norms = np.linalg.norm(embeddings, axis=1)
    results['norm_mean'] = float(norms.mean())
    results['norm_std'] = float(norms.std())
    
    # Similarity statistics (sample)
    sample_size = min(100, len(embeddings))
    sample = embeddings[:sample_size]
    sample_norm = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-10)
    sim_matrix = sample_norm @ sample_norm.T
    upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    
    results['similarity_mean'] = float(upper_tri.mean())
    results['similarity_std'] = float(upper_tri.std())
    
    return results


def test_semantic_quality(checkpoint_path: str = "best_skipgram_523words.pth",
                         test_words: List[str] = None) -> None:
    """
    Test semantic quality by showing nearest neighbors for test words.
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_words: List of words to test (default: sample CIFAR words)
    """
    vocab, embeddings = build_my_embeddings(checkpoint_path)
    
    # Default test words if none provided
    if test_words is None:
        test_words = ['whale', 'rose', 'airplane', 'lion', 'bicycle', 
                     'oak_tree', 'keyboard', 'elephant']
    
    # Convert vocab to nodes list
    nodes = list(set(vocab.keys()))
    
    print("="*80)
    print("SEMANTIC QUALITY TEST - Nearest Neighbors")
    print("="*80)
    
    for word in test_words:
        # Try both formats
        query_word = word
        if word not in vocab:
            query_word = word.replace('_', ' ')
        
        if query_word not in vocab:
            print(f"\n❌ '{word}' not in vocabulary")
            continue
        
        try:
            similar = find_similar_words(word, nodes, embeddings, top_k=5)
            neighbors_str = ", ".join([f"{w}({s:.3f})" for w, s in similar])
            print(f"\n{word:15s} → {neighbors_str}")
        except Exception as e:
            print(f"\n❌ Error for '{word}': {e}")
    
    print("\n" + "="*80)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("CW2: CIFAR-100 SEMANTIC EXPANSION - MODEL TESTING")
    print("="*80)
    
    # Test 1: Load and validate
    print("\n[TEST 1] Loading model...")
    try:
        vocab, embeddings = build_my_embeddings()
        
        print(f"✅ Model loaded successfully!")
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  First 5 words: {list(vocab.keys())[:5]}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test 2: CIFAR-100 coverage
    print("\n[TEST 2] CIFAR-100 coverage...")
    cifar_test_words = ['airplane', 'whale', 'bicycle', 'rose', 'oak_tree', 'oak tree']
    
    for word in cifar_test_words:
        if word in vocab:
            print(f"  ✅ '{word}' → index {vocab[word]}")
        else:
            print(f"  ❌ '{word}' NOT FOUND")
    
    # Test 3: Comprehensive validation
    print("\n[TEST 3] Comprehensive validation...")
    try:
        results = validate_model()
        
        print(f"  Vocabulary entries: {results['vocab_size']}")
        print(f"  Unique embeddings: {results['unique_embeddings']}")
        print(f"  Embedding dimension: {results['embedding_dim']}")
        print(f"  CIFAR-100 coverage: {results['cifar_coverage']}/100")
        print(f"  Norm mean: {results['norm_mean']:.4f}")
        print(f"  Norm std: {results['norm_std']:.4f}")
        print(f"  Similarity mean: {results['similarity_mean']:.4f}")
        print(f"  Similarity std: {results['similarity_std']:.4f}")
        
        # Assessment
        print(f"\n  Assessment:")
        if results['cifar_coverage'] == 100:
            print(f"    ✅ CIFAR-100 coverage: PERFECT")
        else:
            print(f"    ⚠️  CIFAR-100 coverage: {results['cifar_coverage']}/100")
        
        if results['similarity_std'] >= 0.127:
            print(f"    ✅ Similarity std ≥ 0.127: PASS")
        else:
            print(f"    ⚠️  Similarity std < 0.127: NEEDS IMPROVEMENT")
            
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
    
    # Test 4: Semantic quality (optional - can be slow)
    print("\n[TEST 4] Semantic quality test...")
    response = input("  Run semantic quality test? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            test_semantic_quality()
        except Exception as e:
            print(f"  ❌ Semantic test error: {e}")
    else:
        print("  Skipped.")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
