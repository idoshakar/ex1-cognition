import os
from typing import List, Dict, Set, Tuple

import nltk
from nltk.corpus import wordnet as wn, wordnet_ic
import pandas as pd
import math
import itertools

import pandas as pd  # Optional, but useful for large files


def read_frequency_list(file_path):
    """Reads a .freq_list file and returns a dictionary of {word: frequency}."""
    word_frequencies = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 1. Clean up the line (remove leading/trailing whitespace)
                line = line.strip()
                if not line:
                    continue

                # 2. Split the line into word and frequency.
                # Assuming the format is "word count" (separated by space or tab)
                parts = line.split()

                if len(parts) == 2:
                    word = parts[0].lower()  # Normalize to lowercase for consistency
                    try:
                        frequency = int(parts[1])
                        word_frequencies[word] = frequency
                    except ValueError:
                        # Handle cases where the frequency part is not a number
                        print(f"Skipping line with invalid frequency: {line}")
                else:
                    # Handle cases where the line doesn't split into two parts
                    print(f"Skipping malformed line: {line}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    return word_frequencies



# print(frequency_data)
def validate_wordnet_subtree(root_synset):
    """
    Checks if a WordNet subtree starting at root_synset meets two criteria:
    1. Has at least 20 nodes.
    2. Has at least 3 levels (Root -> Child -> Grandchild).

    Args:
        root_synset (Synset): The NLTK WordNet Synset to start from.

    Returns:
        bool: True if both conditions are met, False otherwise.
    """
    # Use a set to track visited nodes (handles potential DAG overlaps)
    visited = set()

    # Stack for DFS traversal: (current_node, current_level)
    # Level 1 = Root, Level 2 = Child, Level 3 = Grandchild
    stack = [(root_synset, 1)]

    max_depth_found = 0
    node_count = 0

    while stack:
        current_node, current_level = stack.pop()

        # Skip if we've already counted this node
        if current_node in visited:
            continue

        visited.add(current_node)
        node_count += 1

        # Update max depth found so far
        if current_level > max_depth_found:
            max_depth_found = current_level

        # --- CHECK CONDITIONS ---
        # We need:
        # 1. Count >= 20
        # 2. Max Depth >= 3 (meaning we found a grandchild)
        if node_count >= 20 and max_depth_found >= 3:
            return True

        # Add children (hyponyms) to the stack
        # We use hyponyms() to traverse 'down' the specific/is-a hierarchy
        for child in current_node.hyponyms():
            stack.append((child, current_level + 1))

    # If the loop finishes without returning True, the conditions were not met
    print(f"Failed: Found {node_count} nodes and {max_depth_found} levels.")
    return False


def get_words_in_subtree(root_synset):
    """
    Recursively finds all unique words (lemmas) associated with the
    given root synset and all of its hyponyms (subordinates).

    Args:
        root_synset (nltk.corpus.wordnet.Synset): The top-level synset
            (the root of the desired subtree).

    Returns:
        set: A set of unique string words found in the subtree.
    """

    # 1. Initialize a set to store unique words and synsets
    # Using a set automatically handles duplicates.
    all_synsets = {root_synset}
    all_words = set()

    # 2. Use a queue for breadth-first traversal (or recursion for depth-first)
    queue = [root_synset]

    while queue:
        current_synset = queue.pop(0)

        # Add the current synset's direct words
        for lemma_name in current_synset.lemma_names():
            all_words.add(lemma_name.replace('_', ' '))  # Replace underscores for clean output

        # Get the immediate children (hyponyms)
        hyponyms = current_synset.hyponyms()

        for hyponym in hyponyms:
            if hyponym not in all_synsets:
                all_synsets.add(hyponym)
                queue.append(hyponym)

    return all_words


def get_all_nodes(root_synset: 'wn') -> List['wn']:
    """
    Helper: Returns a list of all unique Synsets (nodes) in the custom sub-tree
    defined by traversing hyponyms (is-a hierarchy) from the root.
    """
    nodes = {root_synset}
    queue = [root_synset]

    while queue:
        current = queue.pop(0)
        # Traverse down the hierarchy using hyponyms
        for child in current.hyponyms():
            if child not in nodes:
                nodes.add(child)
                queue.append(child)
    return list(nodes)


def compute_synset_probabilities(root_synset: 'wn', corpus_freqs: Dict[str, int]) -> Dict['wn', float]:
    """
    Computes P(w) for every Synset w in the custom sub-tree, based ONLY on
    frequencies of words contained within that sub-tree, as required by the exercise.

    Args:
        root_synset (wn): The root node of the custom 20+ node sub-tree.
        corpus_freqs (Dict[str, int]): Dictionary of {lemma: frequency} from the .freq_list.

    Returns:
        Dict[wn, float]: Dictionary of {Synset: P(w)} probabilities.
    """

    # 1. Identify all nodes in the sub-tree to define the scope (N)
    all_nodes: List['wn'] = get_all_nodes(root_synset)
    all_nodes_set = set(all_nodes)  # Convert to set for O(1) membership check

    # 2. Calculate the Total Corpus Count (N) for the denominator
    # Sum frequencies of all lemmas linked to *all* nodes in the sub-tree.
    all_subtree_lemmas: Set[str] = set()
    for node in all_nodes:
        for lemma in node.lemma_names():
            # Normalize lemma string to match keys in corpus_freqs
            all_subtree_lemmas.add(lemma.lower().replace('_', ' '))

    total_corpus_count: int = 0
    for lemma in all_subtree_lemmas:
        total_corpus_count += corpus_freqs.get(lemma, 0)

    if total_corpus_count == 0:
        print("Warning: Total corpus count for the sub-tree is 0. Returning empty dict.")
        return {}

    # 3. Define helper to calculate freq(w) for a specific synset
    def get_synset_frequency(synset: 'wn') -> int:
        """Calculates freq(synset) by summing all descendant lemma frequencies in the sub-tree."""

        # Use DFS/BFS starting from the current synset to find descendants in the sub-tree
        descendant_nodes: Set['wn'] = set()
        stack = [synset]

        while stack:
            current = stack.pop()

            # CRUCIAL: Only proceed if the current node is part of our custom sub-tree
            if current in all_nodes_set:
                descendant_nodes.add(current)

                # Check children (hyponyms) for the next level
                for child in current.hyponyms():
                    # Only check children that are also in our sub-tree
                    if child in all_nodes_set and child not in descendant_nodes:
                        stack.append(child)

        # Sum the corpus frequencies of all lemmas belonging to these descendant nodes
        current_freq: int = 0
        for node in descendant_nodes:
            for lemma in node.lemma_names():
                current_freq += corpus_freqs.get(lemma.lower().replace('_', ' '), 0)

        return current_freq

    # 4. Calculate freq(w) and P(w) for every node
    synset_probabilities: Dict['wn', float] = {}

    for node in all_nodes:
        freq_w = get_synset_frequency(node)

        # P(w) = freq(w) / N
        p_w = freq_w / total_corpus_count

        synset_probabilities[node] = p_w

    return synset_probabilities

def generate_distance_matrix(root_synset, filename="dist_matrix.csv"):
    """
    Calculates the shortest path distance matrix for all nodes in the subtree
    and saves it to CSV. Returns a dictionary of pairwise distances.
    """
    # 1. Gather all nodes in the subtree
    print(f"Collecting nodes for subtree: {root_synset.name()}...")
    nodes = get_all_nodes(root_synset)
    node_names = [n.name() for n in nodes]
    n_count = len(nodes)

    print(f"Found {n_count} nodes. Calculating {n_count}x{n_count} matrix...")

    # Safety check for performance
    if n_count > 500:
        print("Warning: Large matrix generation might be slow.")

    # 2. Create the Matrix and Dictionary
    matrix_data = []
    # Initialize the dictionary to return
    pairwise_distances = {}

    for row_node in nodes:
        row_data = []
        name1 = row_node.name()
        for col_node in nodes:
            name2 = col_node.name()

            # Calculate shortest path distance (edges between nodes)
            dist = row_node.shortest_path_distance(col_node)

            # Handle cases where no path exists (shouldn't happen in a valid subtree)
            if dist is None:
                dist = -1

            # NEW: Add the pair and distance to the dictionary
            pairwise_distances[(name1, name2)] = dist

            row_data.append(dist)
        matrix_data.append(row_data)

    # 3. Convert to DataFrame for display and saving
    df = pd.DataFrame(matrix_data, index=node_names, columns=node_names)

    # Print to console
    print("\n--- Distance Matrix (Preview) ---")
    print(df.head().to_string())

    # Save to CSV
    df.to_csv(filename)
    print(f"\nSuccess! Matrix saved to '{filename}'")

    # 4. Return the new dictionary
    return pairwise_distances

def compute_information_content(probabilities: Dict['wn', float]) -> Dict['wn', float]:
    """
    Calculates the Information Content (IC) for each synset: IC(w) = -log(P(w)).

    Args:
        probabilities (Dict[wn.Synset, float]): Dictionary of {Synset: P(w)}.

    Returns:
        Dict[wn.Synset, float]: Dictionary of {Synset: IC(w)}.
    """
    ic_scores = {}
    for synset, p_w in probabilities.items():
        if p_w > 0:
            ic_scores[synset] = -math.log(p_w)
        else:
            # Assigning infinity if P(w) is 0 (word was never observed)
            ic_scores[synset] = float('inf')
    return ic_scores


def compute_resnik_similarity(
        synset1: 'nltk.corpus.wordnet.Synset',
        synset2: 'nltk.corpus.wordnet.Synset',
        ic_scores: Dict['nltk.corpus.wordnet.Synset', float],
        wn_api  # Pass in the WordNet API (e.g., nltk.corpus.wordnet as wn)
) -> float:
    # ... (omitted docstring)
    try:
        # FIX: Replaced non-existent lowest_common_subsumer with lowest_common_hypernyms
        # The correct method returns a list, so we take the first element.
        lcs_list = synset1.lowest_common_hypernyms(synset2)

        if not lcs_list:
            # If no common hypernyms are found
            return 0.0

        # The LCS is the first element of the list of lowest common hypernyms
        lcs = lcs_list[0]

        # Resnik similarity is the IC of the LCS
        # CRUCIAL: Check if the LCS is in our *custom* IC scores
        if lcs in ic_scores:
            similarity = ic_scores[lcs]
            return similarity
        else:
            # If LCS is outside the custom sub-tree, similarity based on custom IC is 0.
            return 0.0

    except Exception:
        # Catch exceptions (e.g., if synsets are incompatible or error in traversal)
        return 0.0


# --- 5. Function to Compute Pairwise Resnik Similarity for the Entire Sub-Tree ---

def compute_pairwise_resnik_for_subtree(
        root_synset: 'nltk.corpus.wordnet.Synset',
        corpus_freqs: Dict[str, int],
        wn_api
) -> Dict[Tuple[str, str], float]:
    """
    Computes the maximum Resnik similarity between every unique pair of words
    associated with the nodes in the custom WordNet sub-tree.

    Args:
        root_synset (wn.Synset): The root node of the custom sub-tree.
        corpus_freqs (Dict[str, int]): Dictionary of {lemma: frequency} from the .freq_list.
        wn_api: The WordNet API object (e.g., wn).

    Returns:
        Dict[Tuple[str, str], float]: Dictionary of the form {('word1', 'word2'): max_resnik_sim}.
    """

    # Step 1: Compute custom Information Content (IC) for all nodes
    probabilities = compute_synset_probabilities(root_synset, corpus_freqs)
    ic_scores = compute_information_content(probabilities)

    # Step 2: Gather all unique words (lemmas) in the sub-tree
    all_nodes = get_all_nodes(root_synset)
    all_words: Set[str] = set()
    for node in all_nodes:
        for lemma in node.lemma_names():
            all_words.add(lemma.lower().replace('_', ' '))

    word_list = sorted(list(all_words))
    word_pairs = list(itertools.combinations(word_list, 2))

    print(f"Starting pairwise Resnik calculation for {len(word_list)} words and {len(word_pairs)} pairs.")

    similarity_matrix: Dict[Tuple[str, str], float] = {}

    # Step 3: Compute max similarity for every word pair
    for word1, word2 in word_pairs:
        # Find all noun synsets for each word
        synsets1 = wn_api.synsets(word1, wn_api.NOUN)
        synsets2 = wn_api.synsets(word2, wn_api.NOUN)

        if not synsets1 or not synsets2:
            similarity_matrix[(word1, word2)] = 0.0
            continue

        max_similarity = 0.0

        # Resnik similarity between *words* is the maximum similarity between any pair of their *senses*
        for s1 in synsets1:
            for s2 in synsets2:
                # Use the custom synset-level similarity function
                similarity = compute_resnik_similarity(s1, s2, ic_scores, wn_api)

                if similarity > max_similarity:
                    max_similarity = similarity

        similarity_matrix[(word1, word2)] = max_similarity

    return similarity_matrix
#
#
if __name__ == '__main__':
    # validate root selected is good
    root = wn.synset('road.n.01')
    result = validate_wordnet_subtree(root)
    print(f"Does '{root.name()}' pass? {result}")
#
#     # create distance matrix
    pair_distance = generate_distance_matrix(root, filename="game_distance.csv")
#
#     # Get all words in the subtree
    # my_words = list(get_words_in_subtree(root))
    my_nodes = get_all_nodes(root)
    # Example usage (assuming the file is named 'corpus_ex1.freq_list')
    frequency_data = read_frequency_list('corpus_ex1.freq_list')
#
    probabilities = compute_synset_probabilities(root, frequency_data)

    ic_scores = compute_information_content(probabilities)

    resnik_pairs = compute_pairwise_resnik_for_subtree(root, frequency_data, wn)

