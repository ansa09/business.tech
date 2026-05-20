"""
CW1: Networks and Pathfinding on Literary Text Networks

This module implements graph search algorithms on a text network derived from 
George Orwell's Nineteen Eighty-Four. Each unique word is represented as a node 
and each transition between consecutive words forms a directed edge.

Author: [Your Name]
Date: [Current Date]
"""

# =============================================================================
# IMPORTS
import heapq
import numpy as np

# =============================================================================
# Import lab modules (as completed in previous labs)

# * - * - * - * - * - * - * - * - * - * - * - * 
# TODO: modify these imports as needed
# * - * - * - * - * - * - * - * - * - * - * - * 
from lab2 import *
from lab3 import *
from lab4 import *  
# Students may import additional functionality from labs as needed
# ONLY modules/functions used in labs 1-4 are allowed
# IMPORTANT! NO external libraries beyond what was used in the labs
import heapq
import random
import math
import networkx as nx
import numpy as np
from collections import deque, Counter
import unittest
from unittest.mock import patch, MagicMock
import sys
from io import StringIO

# *  *  *  *  *  *  *  *  *  *  *  *  
#  ** ** ** ** ** ** ** ** ** ** ** *
#  IMPORTANT NOTE ON PATH DEFINITIONS
#  ** ** ** ** ** ** ** ** ** ** ** * 
# *  *  *  *  *  *  *  *  *  *  *  *  
# In this coursework, paths must not contain loops or repeated nodes.
#
# Even though the network is built from a text (where words naturally repeat),
# this assignment focuses on *graph search algorithms* rather than text order.
#
# Therefore:
#   - A valid path must visit each node (word) at most once.
#   - Any solution that revisits nodes (i.e., contains cycles) will be penalized.
#   - You should explicitly prevent loops in your search algorithm logic.
#
# Think of this as a pure pathfinding problem in a directed graph, *not* as a
# simple traversal of the text sequence.
#
# This rule applies to ALL tasks (longest path, most expensive path, quotes, etc.).
# =============================================================================

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# Add any helper functions you need here
#Helper function for [task 3]
'''

def find_most_expensive_path_words(text_network):
    
    import random
    
    
    nodes = text_network["nodes"]
    distance_matrix = text_network["distance_matrix"]
    G = text_network["graph"]
    
    n = len(nodes)
    max_path_cost = -float("inf")
    best_start = None
    best_end = None
    
    random.seed(42)
    sample_size = min(100, n)
    sampled_idx = random.sample(range(n), sample_size)
    
    print(f"Sampling {sample_size} nodes to find most expensive path...")
    
    for count, start_idx in enumerate(sampled_idx):
        if count % 10 == 0:
            print(f"  Processed {count}/{sample_size} nodes...")
        
        # Find maximum distances from this start node
        max_dist = np.full(n, -np.inf)
        max_dist[start_idx] = 0.0
        
        pq = [(0, start_idx)]
        visited = set()
        
        iterations = 0
        max_iterations = 1000
        
        while pq and iterations < max_iterations:
            iterations += 1
            
            neg_dist, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            current_dist = -neg_dist
            
            # FIXED: Only explore actual graph neighbors
            current_node = nodes[u]
            for neighbor in G.neighbors(current_node):
                v = nodes.index(neighbor)
                
                if v in visited:
                    continue
                
                edge_cost = distance_matrix[u][v]
                
                if np.isinf(edge_cost) or edge_cost <= 0:
                    continue
                
                new_dist = current_dist + edge_cost
                
                if new_dist > max_dist[v]:
                    max_dist[v] = new_dist
                    heapq.heappush(pq, (-new_dist, v))
        
        # Find the node with maximum distance from start_idx
        # Only consider reachable nodes
        valid_dists = [(i, d) for i, d in enumerate(max_dist) if d > -np.inf and d > 0]
        if valid_dists:
            goal_local_idx = max(valid_dists, key=lambda x: x[1])[0]
            goal_local_cost = max_dist[goal_local_idx]
            
            if goal_local_cost > max_path_cost:
                max_path_cost = goal_local_cost
                best_start = nodes[start_idx]
                best_end = nodes[goal_local_idx]
                print(f'    New best: start_word="{best_start}", end_word="{best_end}", cost={max_path_cost:.2f}')
    
    print(f"\n=== FINAL RESULT ===")
    print(f'Best pair: "{best_start}" → "{best_end}"')
    print(f"Maximum path cost: {max_path_cost:.2f}")
    
    return (best_start, best_end)
'''
#Helper fuction for [task 4]
'''
def find_expensive_quote_endpoints(text_network, max_search_length=20):
    """
    Helper to identify which start/end words produce the most expensive quote.
    This is useful if you need to hard-code the values.
    """
    try:
        with open("1984.txt", "r", encoding="utf-8") as file:
            raw_text = file.read()
    except FileNotFoundError:
        return (None, None)
    
    from lab2 import tokenize_text
    tokens = tokenize_text(raw_text)
    
    nodes = text_network["nodes"]
    distance_matrix = text_network["distance_matrix"]
    valid_tokens = [t for t in tokens if t in nodes]
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    max_cost = -float("inf")
    best_start = None
    best_end = None
    best_length = 0
    
    print(f"Scanning for expensive quote endpoints (max length: {max_search_length})...")
    
    for start_pos in range(len(valid_tokens) - 1):
        current_cost = 0.0
        
        for end_pos in range(start_pos + 1, min(start_pos + max_search_length, len(valid_tokens))):
            sequence = valid_tokens[start_pos:end_pos + 1]
            
            try:
                prev_word = sequence[-2]
                curr_word = sequence[-1]
                edge_cost = distance_matrix[node_to_idx[prev_word]][node_to_idx[curr_word]]
                
                if np.isinf(edge_cost) or np.isnan(edge_cost):
                    break
                
                current_cost += edge_cost
                
                if current_cost > max_cost:
                    max_cost = current_cost
                    best_start = sequence[0]
                    best_end = sequence[-1]
                    best_length = len(sequence)
                    
            except (KeyError, IndexError):
                break
    
    print(f"\nBest endpoints found:")
    print(f"  start_word='{best_start}'")
    print(f"  end_word='{best_end}'")
    print(f"  Length: {best_length} words")
    print(f"  Cost: {max_cost:.4f}")
    
    return (best_start, best_end)
'''



# =============================================================================
# TASK 1: LONGEST PATH [5 marks]
# =============================================================================
'''
    Return the longest possible path in the text network.

    Description
    -----------
    This function should take the two words in the text network that are
    connected by the *longest possible path* (by number of edges) and return
    that path as a list of words.

    IMPORTANT
    ----------
    - You must first determine (by analysis or experimentation) which two words
      in the network are connected by the *longest path*.
    - Once found, **hard-code those two words** as the default values of
      `start_word` and `end_word` above.
    - Do NOT modify the function signature otherwise.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step

    start_word : str, optional
        The first word (source node) of the longest path.
        By default, this should be set manually to the word
        identified as the start of the longest path.

    end_word : str, optional
        The final word (target node) of the longest path.
        By default, this should be set manually to the word
        identified as the end of the longest path.

    Returns
    -------
    list
        The longest path as a list of words (nodes), e.g.:
            ['it', 'was', 'a', 'bright', 'cold', 'day', 'in', 'April']
        Returns an empty list [] if no path can be found or inputs are invalid.

    Notes
    -----
    - The “longest path” refers to the path with the most edges between two
      connected words in the directed network.
    - You should use the graph search algorithms introduced in previous labs
      (e.g., breadth-first or depth-first search). Do not import new libraries.
    - Efficiency matters for ranking, but correctness is the priority.

    TODO
    ----
    1. Identify the two words in the text network connected by the longest path.
    2. Replace the placeholders above with those two words.
    3. Implement the search algorithm to return that path as a list of words.
    """

    # TODO: Implement your algorithm here to find the path
    # Example steps (you may modify as needed):
    # 1. Retrieve the graph from text_network
    # 2. Use a search algorithm (e.g., DFS or BFS) to find a path from start_word to end_word
    # 3. Return the resulting list of words representing that path

'''

def print_long_path(text_network, start_word='sort', end_word='the'):
    graph = text_network['graph']
    
    # Helper function to check that the words are valid, filtering out unhelpful nodes.
    def is_valid_word(word):
        if len(word) <= 2 or word.isdigit():
            return False
        if word.startswith('<') and word.endswith('>'):
            return False
        return True
    #Filtering through the nodes in the graph to find out how many nodes there are and which are useable
    all_nodes = list(graph.nodes())
    filtered_nodes = [n for n in all_nodes if is_valid_word(n)]
    
    #print(f"Total nodes: {len(all_nodes)}, Filtered: {len(filtered_nodes)}")
    
    # Validating the start and end words are in the graph
    if start_word is not None and start_word not in graph:
        print(f"ERROR: start_word '{start_word}' not in graph")
        return []
    if end_word is not None and end_word not in graph:
        print(f"ERROR: end_word '{end_word}' not in graph")
        return []
    
    # If both start and end are specified, search between them
    if start_word is not None and end_word is not None:
        print(f"Searching from '{start_word}' to '{end_word}'")
        search_nodes = [start_word]
    elif start_word is not None:
        print(f"Searching from '{start_word}'")
        search_nodes = [start_word]
    else:
        # For every valid word the number of connections is being determined
        node_degrees = [(n, graph.degree(n)) for n in filtered_nodes]
        node_degrees.sort(key=lambda x: x[1])
        search_nodes = [n for n, d in node_degrees[:5000]]
        #print(f"Selected {len(search_nodes)} low-degree candidates")
    #initialising the search variables
    longest_path = []
    nodes_explored = 0
    MAX_EXPLORATIONS = 10000000
    #Implementing an iterative depth-first search function
    def dfs_iterative(start_node, target_node=None, max_depth=1000):
        nonlocal nodes_explored
        
        best_path = [start_node]
        stack = [(start_node, [start_node], {start_node})]
        
        while stack and nodes_explored < MAX_EXPLORATIONS:
            nodes_explored += 1
            
            node, path, visited = stack.pop()
            
            # Update best path
            if len(path) > len(best_path):
                best_path = path[:]
            
            # If we the target is reached it returns immediately
            if target_node and node == target_node:
                return best_path
            
            if len(path) >= max_depth:
                continue
            
            # Get valid neighboring nodes which have not been visited yet - limited to 5
            neighbors = [n for n in graph.neighbors(node) 
                        if n not in visited and is_valid_word(n)]
            
            neighbors = sorted(neighbors, key=lambda n: graph.degree(n))[:5]
            #Expanding the search for neighboring nodes which haven't been explored yet and adding those to the next stack
            for neighbor in reversed(neighbors):
                new_visited = visited | {neighbor}
                new_path = path + [neighbor]
                stack.append((neighbor, new_path, new_visited))
        
        return best_path
    
    # Search with the specified start word, if one is given otherwise try 100 different start nodes
    num_tries = 1 if start_word is not None else 100
    # Runs dfs for each candidate
    for i, start_node in enumerate(search_nodes[:num_tries]):
        if nodes_explored >= MAX_EXPLORATIONS:
            print(f"  Reached total exploration limit")
            break
        
        #if i % 5 == 0 and num_tries > 1:
            #print(f"  Tried {i}/{num_tries} nodes, best: {len(longest_path)}, explored: {nodes_explored}")
        
        path = dfs_iterative(start_node, target_node=end_word, max_depth=1000)
        
        # Check if path ends at the required end word (if specified)
        if end_word is not None and len(path) > 0 and path[-1] != end_word:
            continue  # Skip paths that don't reach the target
        #Checking if a longer path is found, then replacing 
        if len(path) > len(longest_path):
            longest_path = path
            #print(f"    NEW BEST from '{start_node}': {len(path)} nodes")
            
            # When a path with both endpoints matching - its complete, goal reached 
            if start_word and end_word and len(path) > 0:
                if path[0] == start_word and path[-1] == end_word:
                    #print(f"    Found path from '{start_word}' to '{end_word}'")
                    break
    
    #print(f"\n{'='*50}")
    #print(f"FINAL: Longest path has {len(longest_path)} nodes")
    
    if longest_path:
        print(f"Start: '{longest_path[0]}'")
        print(f"End: '{longest_path[-1]}'")
        print(f"Length: {len(longest_path)} words")
        print(f"Path: {longest_path}")
        
        #print(f"\nFirst 30: {' -> '.join(longest_path[:30])}")
        #if len(longest_path) > 60:
            #print(f"Middle 30: {' -> '.join(longest_path[len(longest_path)//2-15:len(longest_path)//2+15])}")
            #print(f"Last 30: {' -> '.join(longest_path[-30:])}")
        
        # Verify no cycles - path is acyclical
        if len(longest_path) != len(set(longest_path)):
            print("\n⚠️  WARNING: Path contains repeated nodes!")
        else:
            print(f"\n✓ Verified: No repeated nodes (no cycles)")
    else:
        print("No path found!")
    
    return longest_path

# =============================================================================
# TASK 2: LONGEST QUOTE [5 marks]
# =============================================================================
'''
    Return the longest literal quote in the text network.

    Description
    -----------
    This function should return the *longest contiguous sequence of words*
    that appears exactly as in the original text (i.e., a literal quote).

    IMPORTANT
    ----------
    - You must first determine (by analysis or experimentation) which two words
      mark the start and end of the longest contiguous quote.
    - Once found, **hard-code those two words** as the default values of
      `start_word` and `end_word` above.
    - Do NOT modify the function signature otherwise.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    start_word : str, optional
        The first word (source node) of the longest literal quote.

    end_word : str, optional
        The final word (target node) of the longest literal quote.

    Returns
    -------
    list
        The longest literal quote as a list of words (nodes), e.g.:
            ['"it', 'was', 'a', 'bright', 'cold', 'day', 'in', 'April"']
        Returns an empty list [] if no quote can be found or inputs are invalid.

    Notes
    -----
    - The quote must appear *exactly* as in the original text.
    - Use your text network structure to trace contiguous word sequences.
    - You may reuse traversal or search logic from previous tasks.

    TODO
    ----
    1. Identify the start and end words for the longest literal quote.
    2. Replace the placeholders above with those two words.
    3. Implement the search logic to return that sequence.
    """

    # TODO: Implement your quote-finding logic here
    # HINT: You can iterate through word transitions to find contiguous sequences

'''


    
def print_long_quote(text_network, start_word='side', end_word='the'):
    
    # Loading 1984.txt
    try:
        with open("1984.txt", "r", encoding="utf-8") as file:
            raw_text = file.read()
    except FileNotFoundError:
        print("Error: Could not find 1984.txt")
        return []
    #Tokeniszing into a list of words
    from lab2 import tokenize_text
    original_tokens = tokenize_text(raw_text)
    
    # Helper function to check all words are valid
    def is_valid_word(word):
        if len(word) <= 2 or word.isdigit():
            return False
        if word.startswith('<') and word.endswith('>'):
            return False
        return True
    
    if not isinstance(text_network, dict):
        print("Error: text_network must be a dictionary")
        return []
    #Checking the format making sure the stucture is correct
    if 'adjacency_counts' not in text_network:
        print("Error: text_network must contain 'adjacency_counts' key")
        return []
    
    adjacency_counts = text_network['adjacency_counts']
    nodes = text_network['nodes']
    
    # Building a dictionary of the allowed next words based on above adjacency count
    transitions = {}
    for (word1, word2), count in adjacency_counts.items():
        if word1 not in transitions:
            transitions[word1] = {}
        transitions[word1][word2] = count
    
    #print(f"Searching for longest CONTIGUOUS literal quote in {len(original_tokens)} tokens...")
    
    longest_quote = []
    
    # Scanning through the text and trying each position as a starting point
    for start_pos in range(len(original_tokens)):
        if not is_valid_word(original_tokens[start_pos]):
            continue
        if original_tokens[start_pos] not in transitions:
            continue
        
        # Building a contiguous sequence, starting from the selected word and adding a word at a time 
        current_quote = [original_tokens[start_pos]]
        visited = {original_tokens[start_pos]}
        pos = start_pos + 1
        
        # Follow consecutive words in the text
        while pos < len(original_tokens):
            next_word = original_tokens[pos]
            
            # Stops if invalid
            if not is_valid_word(next_word) or next_word not in nodes:
                break
            
            # Stops if it would create a cycle
            if next_word in visited:
                break
            
            # Stop if no edge exists in the graph
            current_word = current_quote[-1]
            if current_word not in transitions or next_word not in transitions[current_word]:
                break
            
            
            current_quote.append(next_word)
            visited.add(next_word)
            pos += 1
        
        # tracking the longest quote
        if len(current_quote) > len(longest_quote):
            longest_quote = current_quote[:]
            #print(f"  New best: {len(longest_quote)} words at position {start_pos}")
        
        # Progress
        #if start_pos % 10000 == 0:
            #print(f"  Checked {start_pos} positions, best: {len(longest_quote)} words")
    
    print(f"\nAs list:")
    print(longest_quote)
    
    # Verify no cycles
    if len(longest_quote) != len(set(longest_quote)):
        print(f"\n❌ ERROR: Quote contains cycles!")
    else:
        print(f"\n✓ Verified: No repeated words (no cycles)")
    
    # Verify no <RARE> tokens
    rare_count = sum(1 for word in longest_quote if word.startswith('<') and word.endswith('>'))
    if rare_count > 0:
        print(f"⚠️ WARNING: Found {rare_count} <RARE> tokens!")
    else:
        print(f"✓ Verified: No <RARE> tokens")
    
    if longest_quote:
        #print(f"\nDiscovered quote:")
        print(f"  Start word: '{longest_quote[0]}'")
        print(f"  End word: '{longest_quote[-1]}'")
        print(f"  Length: {len(longest_quote)} words")
        print(f"  Quote: {' '.join(longest_quote)}")
    
    return longest_quote

# =============================================================================
# TASK 3: MOST EXPENSIVE PATH [5 marks]
# =============================================================================



def print_expensive_path(text_network, start_word="remember", end_word="impossible"):
    """
    Return the most expensive path between two words in the text network.

    Description
    -----------
    This function should return the *most expensive path* (i.e., the path
    with the highest cumulative cost) between two connected words in the network.

    IMPORTANT
    ----------
    - You must first determine (by analysis or experimentation) which two words
      are connected by the most expensive path.
    - Once found, **hard-code those two words** as the default values of
      `start_word` and `end_word` above.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    start_word : str, optional
        The first word (source node) of the most expensive path.

    end_word : str, optional
        The final word (target node) of the most expensive path.

    Returns
    -------
    tuple
        (path, total_cost)
        where `path` is a list of words and `total_cost` is a numeric value.

    Notes
    -----
    - You should use your path-cost computation from previous labs.
    - Use the parameter `distance_mode="inverted"` when computing costs.

    TODO
    ----
    1. Identify and hard-code the start and end nodes of the most expensive path.
    2. Implement the search algorithm to find that path.
    3. Compute and return the total path cost.
    """

    # TODO: Implement cost-based path search here (using distance_mode="inverted")
    
    #if start_word is None or end_word is None:
        #print("Auto-detecting most expensive path...")
        #start_word, end_word = find_most_expensive_path_words(text_network)
        #print(f"Using: start_word='{start_word}', end_word='{end_word}'")

    #Pulling out the data from the text network and making a distance matrix based on this, ensurung distance_mode="inverted" as per requirements.
    nodes = text_network["nodes"]
    adjacency_counts = text_network["adjacency_counts"]
    from lab2 import compute_distance_matrix
    distance_matrix, _ = compute_distance_matrix(nodes, adjacency_counts, distance_mode="inverted")
    G = text_network["graph"]
    #If words used which are not in the data set, algorithm will stop
    if start_word not in nodes or end_word not in nodes:
        print(f"Error: Start '{start_word}' or end '{end_word}' not in network")
        return ([], 0.0)
    #mapping words to numbers
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    #Priority queue set up for search
    tie = 0
    pq = [(0.0, tie, start_word, [start_word], frozenset([start_word]))]
    #Checking the best path to the end word and the best cost to each node
    best_path_to_goal = None
    best_cost_to_goal = -np.inf
    
    # Loose tracking - allow multiple good paths to each node
    best_to_node = {}
    
    iterations = 0
    max_iterations = 200000
    max_path_length = 250
    
    #print(f"Searching for most expensive path (max {max_iterations} iterations)...")
    #progress_interval = 20000
    #Main function for the search, using a hybrid Dijikstra (reverse) and BFS (using weights which bfs usually doesn't).
    while pq and iterations < max_iterations:
        iterations += 1
        
        # Progress updates
        #if iterations % progress_interval == 0:
            #print(f"  Iteration {iterations}/{max_iterations}: queue size={len(pq)}, best cost={best_cost_to_goal:.2f}")
        
        neg_cost, _, current_node, path, visited = heapq.heappop(pq)
        current_cost = -neg_cost
        
        # Prune very long paths, if the path is really long it will stop - this was causing issues to the search
        if len(path) > max_path_length:
            continue
        
        # Loose pruning: keep paths within 70% of best to this node
        if current_node in best_to_node:
            if current_cost < best_to_node[current_node] * 0.7:
                continue
        best_to_node[current_node] = max(best_to_node.get(current_node, 0), current_cost)
        
        # Goal check, when the goal is reached the global best is updated, and search continues
        if current_node == end_word:
            if current_cost > best_cost_to_goal:
                best_cost_to_goal = current_cost
                best_path_to_goal = path[:]
                #print(f"  *** NEW BEST PATH! Cost: {best_cost_to_goal:.2f}, Length: {len(path)} nodes ***")
            continue
        
        # Explore neighbors, cost is computed and pushed into priority queue.
        for neighbor in G.neighbors(current_node):
            if neighbor in visited:
                continue
            
            current_idx = node_to_idx[current_node]
            neighbor_idx = node_to_idx[neighbor]
            edge_cost = distance_matrix[current_idx][neighbor_idx]
            
            if np.isinf(edge_cost) or edge_cost <= 0:
                continue
            
            new_cost = current_cost + edge_cost
            new_path = path + [neighbor]
            new_visited = visited | frozenset([neighbor])
            
            tie += 1
            heapq.heappush(pq, (-new_cost, tie, neighbor, new_path, new_visited))
    
    if best_path_to_goal is None:
        print(f"No path found after {iterations} iterations")
        return ([], 0.0)
    # Verify no <RARE> tokens, otherwise this will give a false path
    rare_count = sum(1 for word in best_path_to_goal if word.startswith('<') and word.endswith('>'))
    if rare_count > 0:
        print(f"⚠️ WARNING: Found {rare_count} <RARE> tokens in result!")
    else:
        print(f"✓ Verified: No <RARE> tokens in quote")
    
    #print(f"\nSearch complete! Iterations: {iterations}, Final best cost: {best_cost_to_goal:.2f}, Path length: {len(best_path_to_goal)}")
    print(f"Start word: '{best_path_to_goal[0]}'")
    print(f"End word: '{best_path_to_goal[-1]}'")
    print(f"Path: {(best_path_to_goal)}")
    print(f"Cost: {(best_cost_to_goal)}")
    return (best_path_to_goal, best_cost_to_goal)


# =============================================================================
# TASK 4: MOST EXPENSIVE QUOTE [5 marks]
# =============================================================================


def print_expensive_quote(text_network, start_word='perhaps', end_word='it'):
    """
    Return the most expensive literal quote in the text network.

    Description
    -----------
    This function should return the literal quote (contiguous word sequence)
    that has the *highest total cost* according to the network’s edge weights.

    IMPORTANT
    ----------
    - You must first determine (by analysis or experimentation) which two words
      mark the start and end of the most expensive literal quote.
    - Once found, **hard-code those two words** as the default values of
      `start_word` and `end_word` above.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    start_word : str, optional
        The first word (source node) of the most expensive literal quote.

    end_word : str, optional
        The final word (target node) of the most expensive literal quote.

    Returns
    -------
    tuple
        (quote, total_cost)
        where `quote` is a list of words (the literal quote) and
        `total_cost` is the numeric cost of that quote.

    Notes
    -----
    - The quote must appear exactly as in the original text.
    - You may reuse your traversal logic and cost function from previous tasks.

    TODO
    ----
    1. Identify the start and end words of the most expensive literal quote.
    2. Replace the placeholders above.
    3. Implement the logic to compute and return the quote and total cost.
    """
    #Pulling out all data from the text network into memeory as a string
    try:
        with open("1984.txt", "r", encoding="utf-8") as file:
            raw_text = file.read()
    except FileNotFoundError:
        print("Error: Could not find 1984.txt")
        return ([], 0.0)
    #Text is turned to a string
    from lab2 import tokenize_text
    original_tokens = tokenize_text(raw_text)
    
    #Weighted directed graph
    nodes = text_network["nodes"]
    adjacency_counts = text_network["adjacency_counts"]
    from lab2 import compute_distance_matrix
    distance_matrix, _ = compute_distance_matrix(nodes, adjacency_counts, distance_mode="inverted")
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    #Tracking highest cost, best quote, and limits infinite loops.
    max_cost = -float("inf")
    best_quote = []
    best_start_pos = -1
    max_search_length = 500
    
    # Find all positions where valid sequences can start, only starting at words which exist in the graph
    valid_starts = []
    for i in range(len(original_tokens) - 1):
        if original_tokens[i] in node_to_idx:
            valid_starts.append(i)
    
    # Search from each valid starting position and keep stepping forward (a greedy rule) checking edge exists in the graph and has a positive weight.
    for start_idx in valid_starts:
        current_cost = 0.0
        current_quote = [original_tokens[start_idx]]
        visited = {original_tokens[start_idx]}  #Ensuring no cycles
        
        pos = start_idx + 1
        
        while pos < len(original_tokens) and len(current_quote) < max_search_length:
            curr_token = original_tokens[pos]
            
            # Stop if token not in network
            if curr_token not in node_to_idx:
                break
            
            # Stop if would create cycle
            if curr_token in visited:
                break
            #If the edge is valid then add the cost and append the word.
            prev_token = current_quote[-1]
            prev_idx = node_to_idx[prev_token]
            curr_idx = node_to_idx[curr_token]
            edge_cost = distance_matrix[prev_idx][curr_idx]
            
            # Stop if invalid edge
            if np.isinf(edge_cost) or np.isnan(edge_cost) or edge_cost < 0:
                break
            
            current_cost += edge_cost
            current_quote.append(curr_token)
            visited.add(curr_token)  # Mark as visited
            
            # Update best if this sequence is better (and has at least 2 words)
            if current_cost > max_cost and len(current_quote) >= 2:
                max_cost = current_cost
                best_quote = current_quote[:]
                best_start_pos = start_idx
            
            pos += 1
        
        # Progress indicator - used when searching for best path and words, used in most the above tasks as well when helper function is built in main function.
        #if start_idx % 10000 == 0:
            #progress = (valid_starts.index(start_idx) / len(valid_starts)) * 100
            #print(f"  Progress: {progress:.1f}% (best cost so far: {max_cost:.2f}, length: {len(best_quote)})")

    #More validation 
    #if not best_quote:
        #print("Could not find any valid quote")
        #return ([], 0.0)
    
    # Output results
    print(f"\n{'='*60}")
    print(f"Start word: '{best_quote[0]}'")
    print(f"End word: '{best_quote[-1]}'")
    print(f"Quote length: {len(best_quote)} words")
    print(f"Total cost: {max_cost:.4f}")
    
    # Verify no cycles
    if len(best_quote) != len(set(best_quote)):
        print(f"❌ ERROR: Quote contains cycles!")
    else:
        print(f"✓ Verified: No repeated words (no cycles)")
    
    return (best_quote, max_cost)

# =============================================================================
# TASK 5: HEURISTIC SEARCH [30 marks total]
# =============================================================================

# -------------------------------------------------------------------------
# Part (a): Sentence Completion [10 marks]
# -------------------------------------------------------------------------

    """
    Complete a sentence by filling the <CONTENT> placeholder using heuristic search.

    Description
    -----------
    This function should take a sentence containing the token <CONTENT> and use
    a heuristic search algorithm (inspired by A*) to generate a coherent sequence
    of words to replace that token.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    prompt : str
        A string containing the <CONTENT> token to be completed.

    Returns
    -------
    list
        The completed sentence as a list of words.

    Notes
    -----
    - The heuristic should guide the search toward semantically or syntactically
      plausible completions.
    - You may design your own heuristic (to be explained in report.pdf).

    TODO
    ----
    1. Parse the input sentence and identify the <CONTENT> region.
    2. Implement a heuristic search to fill that region with words.
    3. Return the full completed sentence as a list of words.
    """

    # TODO: Implement heuristic sentence completion



def complete_sentence(text_network, prompt="please believe my eyes <CONTENT>."):
    
    
    #Building transitions
    adjacency_counts = text_network.get('adjacency_counts', {})
    
    if not adjacency_counts:
        return prompt.replace("<CONTENT>", "").split()
    
    # Build clean transitions - ignores rare tokens, loops and reduces weight if next word in punctuation
    transitions = {}
    for (word1, word2), count in adjacency_counts.items():
        # Filter noise
        if '<RARE>' in (word1, word2):
            continue
        if word1 == word2:
            continue
        
        # Downweight punctuation endings
        weight = count
        if word2 in {'.', ',', '!', '?', ';', ':'}:
            weight = max(1, count // 4)
        
        if word1 not in transitions:
            transitions[word1] = {}
        transitions[word1][word2] = weight
    
    if not transitions:
        return prompt.replace("<CONTENT>", "").split()
    
    #Splitting words before and after content and punctuation; this is so the function knows what word to begin with, punctuation to add and potential goal. 
    if "<CONTENT>" not in prompt:
        return prompt.split()
    
    parts = prompt.split("<CONTENT>")
    before_words = parts[0].strip().split() if parts[0].strip() else []
    after_text = parts[1].strip() if len(parts) > 1 else ""
    
    # Extract goal and punctuation
    after_words = []
    trailing_punct = []
    
    if after_text:
        if after_text in {'.', ',', '!', '?', ';', ':'}:
            trailing_punct.append(after_text)
        else:
            for token in after_text.split():
                if token in {'.', ',', '!', '?', ';', ':'}:
                    trailing_punct.append(token)
                else:
                    clean = token.strip('.,!?;:').lower()
                    if clean and len(clean) > 1:
                        after_words.append(clean)
                    for char in token:
                        if char in '.,!?;:' and char not in trailing_punct:
                            trailing_punct.append(char)
    
   
    # Find start word
    start_word = None
    if before_words:
        for word in reversed(before_words):
            clean = word.strip('.,!?;:').lower()
            if clean in transitions:
                start_word = clean
                break
    #If there isn't a sensible word, choose the most content rich word 
    if not start_word and transitions:
        # Pick word with most content neighbors
        start_word = max(transitions.keys(), 
                        key=lambda w: sum(1 for n in transitions[w] 
                                        if n not in {'.', ',', '!', '?'}))
    
    if not start_word:
        return before_words + trailing_punct
    
    
    # Find goal word
    goal_word = None
    for word in after_words:
        if word in transitions:
            goal_word = word
            break
    
    #Limitations and requirements to meet to help sentence coherence 
    MIN_CONTENT_WORDS = 4
    MAX_PATH_LENGTH = 10
    
    # Strongly avoiding these- otherwise output includes too many repititions. 
    FILLER_WORDS = {
        'the', 'a', 'an', 'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with',
        'and', 'or', 'but', 'be', 'is', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'shall'
    }
    
    
    # Beam Search - more robust than pure greedy
    def beam_search_generate(start, goal, beam_width=5):
       
        # Each beam item: (score, path, current_word, visited_set)
        # Score = negative of path quality (lower is better for heapq library)
        #Rewarding high transition weights, punishing filler words, and preventing cycles
        beams = [(0.0, [], start, {start})]
        
        best_complete_path = None
        best_complete_score = float('inf')
        
        for depth in range(MAX_PATH_LENGTH):
            candidates = []
            
            for score, path, current, visited in beams:
                # If we reached goal with enough words, save it
                if goal and current == goal and len(path) >= MIN_CONTENT_WORDS:
                    if score < best_complete_score:
                        best_complete_path = path + [current]
                        best_complete_score = score
                    continue
                
                # If path is long enough and no goal, search stops
                if not goal and len(path) >= MIN_CONTENT_WORDS:
                    if score < best_complete_score:
                        best_complete_path = path + [current]
                        best_complete_score = score
                
                # Expand neighbors
                neighbors = transitions.get(current, {})
                
                for next_word, weight in neighbors.items():
                    # No loops ensured
                    if next_word in visited:
                        continue
                    
                    # Skip punctuation unless at end
                    if next_word in {'.', ',', '!', '?', ';', ':'}:
                        continue
                    
                    # Calculating score
                    # Punishing filler words heavily
                    word_penalty = 10.0 if next_word in FILLER_WORDS else 0.0
                    
                    # Reward high weight transitions
                    transition_score = 1.0 / (weight + 1)
                    
                    # Bonus for reaching goal
                    goal_bonus = -100.0 if next_word == goal else 0.0
                    
                    new_score = score + transition_score + word_penalty + goal_bonus
                    
                    new_path = path + [next_word]
                    new_visited = visited | {next_word}
                    
                    candidates.append((new_score, new_path, next_word, new_visited))
            
            if not candidates:
                break
            
            # Keep best beam_width candidates
            candidates.sort(key=lambda x: x[0])
            beams = candidates[:beam_width]
            
            # Early stopping if we found goal
            if best_complete_path and goal:
                break
        
        # Return best path found
        if best_complete_path:
            return best_complete_path
        
        # Fallback, return best incomplete path
        if beams:
            return beams[0][1] + [beams[0][2]]
        
        return []
    
    # Generating the content
    generated = beam_search_generate(start_word, goal_word, beam_width=5)
    
    # Remove start word if it was already in before_words
    if generated and generated[0] == start_word:
        generated = generated[1:]
    
    # Filter out any remaining filler words if we have enough content - avoiding non sensical sentences
    if len(generated) > MIN_CONTENT_WORDS:
        # Keep some filler for grammar - not too many
        content_count = sum(1 for w in generated if w not in FILLER_WORDS)
        if content_count >= MIN_CONTENT_WORDS:
            # Allow max 30% filler words - inline with general english standard language.
            max_filler = max(2, len(generated) // 3)
            filtered = []
            filler_count = 0
            for w in generated:
                if w in FILLER_WORDS:
                    filler_count += 1
                    if filler_count <= max_filler:
                        filtered.append(w)
                else:
                    filtered.append(w)
            generated = filtered
        result = before_words + generated

        # Add remaining after_words if goal was reached
        if goal_word and generated and generated[-1] == goal_word:
            if len(after_words) > 1:
                result.extend(after_words[1:])

        result.extend(trailing_punct)

        # Removing consecutive duplicates- avoiding non sensical sentences
        final_result = []
        prev_word = None
        for word in result:
            if word != prev_word:  # Only add if different from previous
                final_result.append(word)
                prev_word = word

        return final_result



  

# -------------------------------------------------------------------------
# Part (b): Sentence Starting [10 marks]
# -------------------------------------------------------------------------

def start_sentence(text_network, prompt="two <CONTENT> can ask for a solution."):
    """
    Generate a plausible sentence beginning using heuristic search.

    Description
    -----------
    This function should take a sentence containing the token <CONTENT> and
    replace that token with a coherent sequence of words that could plausibly
    precede the given phrase.

    Parameters
    ----------
    text_network : dict
        A dictionary produced by your text-processing step.

    prompt : str
        A string containing the <CONTENT> token to be expanded.

    Returns
    -------
    list
        The full generated sentence as a list of words.

    Notes
    -----
    - The heuristic should guide the search toward plausible predecessors.
    - You may base this on linguistic, statistical, or semantic principles.

    TODO
    ----
    1. Parse the input sentence and identify the <CONTENT> token.
    2. Implement a heuristic search to generate preceding words.
    3. Return the reconstructed full sentence as a list of words.
    """

    # TODO: Implement heuristic sentence starting
    
    # Building reverse transition network generating words that come before the phrase.
    adjacency_counts = text_network.get('adjacency_counts', {})
    
    if not adjacency_counts:
        return prompt.replace("<CONTENT>", "").split()
    
    # Build reverse transitions
    reverse_transitions = {}
    for (word1, word2), count in adjacency_counts.items():
        # Filtering for clean output
        if '<RARE>' in (word1, word2):
            continue
        if word1 == word2:
            continue
        
        # Ensuring word2 > word1
        if word2 not in reverse_transitions:
            reverse_transitions[word2] = {}
        reverse_transitions[word2][word1] = count
    
    if not reverse_transitions:
        return prompt.replace("<CONTENT>", "").split()
    
    # Parse prompt - splitting again around content like in part a
    if "<CONTENT>" not in prompt:
        return prompt.split()
    
    parts = prompt.split("<CONTENT>")
    before_text = parts[0].strip()
    after_text = parts[1].strip() if len(parts) > 1 else ""
    
    # Preserve original structure
    after_words_original = after_text.split() if after_text else []
    
    # Clean after_words for searching
    after_clean = []
    for word in after_words_original:
        clean = word.strip('.,!?;:\'"').lower()
        if clean and len(clean) > 1:
            after_clean.append(clean)
    
    # Extract any prefix words before <CONTENT>
    prefix_words = before_text.split() if before_text else []
    
    
    # Find start word (first word after <CONTENT>)
    start_word = None
    for word in after_clean:
        if word in reverse_transitions:
            start_word = word
            break
    
    if not start_word:
        return prefix_words + after_words_original
    
    
    # Find goal word 
    goal_word = None
    if prefix_words:
        # Clean prefix words
        prefix_clean = []
        for word in prefix_words:
            clean = word.strip('.,!?;:\'"').lower()
            if clean:
                prefix_clean.append(clean)
        
        # Find goal from end of prefix
        for word in reversed(prefix_clean):
            if word in reverse_transitions and word != start_word:
                goal_word = word
                break
    
    #Same as part a - #Limitations and requirements to meet to help sentence coherence, strongly avoiding these- otherwise output includes too many repititions. 
    MIN_CONTENT_WORDS = 1
    MAX_TOTAL_WORDS = 5
    
    # Comprehensive filler word list
    FILLER_WORDS = {
        'the', 'a', 'an', 'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with',
        'and', 'or', 'but', 'if', 'as', 'be', 'is', 'was', 'were', 'been', 'being', 'are',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'shall', 
        'these', 'those', 'it', 'he', 'she', 'they', 'them', 'their', 'there',
        'his', 'her', 'its', 'our', 'your', 'my', 'me', 'us', 'we', 'you',
        'who', 'what', 'where', 'when', 'why', 'how', 'which', 'from',
        'no', 'not', 'one', 'two', 'three', 'made', 'said', 'thing', 'some',
        'make', 'out', 'time', 'about', 'into', 'so', 'up', 'down', 'then',
        'than', 'now', 'only', 'just', 'like', 'such', 'get', 'all', 'any'
    }
    
    
    # Greedy bfs (backward generation) - This looks at the predecessor choices, scores them and picks the next previous word, this is making the chain point backwards to a prefered start.
    def greedy_search_backwards(start, goal):
        
        if not start:
            return []
            
        path = []
        visited = {start}
        current = start
        
        for step in range(MAX_TOTAL_WORDS):
            # Geting predecessors
            predecessors = reverse_transitions.get(current, {})
            if not predecessors:
                break
            
            # Filter out visited nodes and punctuation
            candidates = []
            for prev_word, weight in predecessors.items():
                if prev_word in visited:
                    continue
                if prev_word in {'.', ',', '!', '?', ';', ':', '"', "'"}:
                    continue
                
                # Score each candidate
                is_content = prev_word not in FILLER_WORDS
                
                # Favor content words
                content_bonus = 100.0 if is_content else 0.0
                
                # Favor high-frequency transitions
                frequency_score = weight * 10.0
                
                # Bonus if this is the goal
                goal_bonus = 1000.0 if prev_word == goal else 0.0
                
                # Bonus for direct connection to goal
                lookahead_bonus = 0.0
                if goal and goal != prev_word:
                    if goal in reverse_transitions.get(prev_word, {}):
                        lookahead_bonus = 200.0
                
                total_score = (content_bonus + frequency_score + 
                             goal_bonus + lookahead_bonus)
                
                candidates.append((total_score, prev_word))
            
            if not candidates:
                break
            
            # Pick the best candidate
            candidates.sort(reverse=True)
            best_score, best_word = candidates[0]
            
            path.append(best_word)
            visited.add(best_word)
            current = best_word
            
            # Stop if we reached goal
            if goal and current == goal:
                break
            
            # Early stop if we have enough content and no specific goal
            content_count = sum(1 for w in path if w not in FILLER_WORDS)
            if content_count >= 2 and not goal:
                break
        
        return path
    

    # Generate path - this will be in reverse order, newest to oldest
    generated_path = greedy_search_backwards(start_word, goal_word)
    
    # Reverse to get chronological order - oldest to newest
    generated = list(reversed(generated_path))
    
    # If generated is empty or only has goal word, at least one content word is needed
    if len(generated) == 0 or (len(generated) == 1 and generated[0] == goal_word):
        # Try to get at least one good predecessor
        predecessors = reverse_transitions.get(start_word, {})
        if predecessors:
            # Pick best content word
            best_word = None
            best_score = -1
            for word, weight in predecessors.items():
                if word in FILLER_WORDS or word in {'.', ',', '!', '?', ';', ':'}:
                    continue
                if word == goal_word:
                    continue
                if weight > best_score:
                    best_score = weight
                    best_word = word
            
            if best_word:
                generated = [best_word]
    
    # Limit filler words - this helps ensure the words are coherent
    if len(generated) > 1:
        content_words = [w for w in generated if w not in FILLER_WORDS]
        filler_words_list = [w for w in generated if w in FILLER_WORDS]
        
        # Keep all content, limit fillers to 1 - this helps ensure the words are coherent
        if len(content_words) >= 1 and len(filler_words_list) > 1:
            filtered = []
            filler_count = 0
            for w in generated:
                if w not in FILLER_WORDS:
                    filtered.append(w)
                elif filler_count < 1:
                    filtered.append(w)
                    filler_count += 1
            generated = filtered
    
    # Remove goal_word if it's in generated (it's already in prefix)
    generated = [w for w in generated if w != goal_word]
    
    # prefix + generated + after_words built
    result = prefix_words + generated + after_words_original
    
    # Remove consecutive duplicates only - output otherwise duplicated words because of limited vocabulary 
    final_result = []
    prev_word = None
    for word in result:
        if word != prev_word:
            final_result.append(word)
            prev_word = word
    
    return final_result