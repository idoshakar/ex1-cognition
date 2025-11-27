from typing import List, Dict, Set, Tuple
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import math
import itertools
import os


def read_frequency_list(file_path: str) -> Dict[str, int]:
    """Reads a .freq_list file and returns a dictionary of {word: frequency}."""
    word_frequencies = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                if len(parts) == 2:
                    word = parts[0].lower()
                    try:
                        frequency = int(parts[1])
                        word_frequencies[word] = frequency
                    except ValueError:
                        print(f"Skipping line with invalid frequency: {line}")
                else:
                    print(f"Skipping malformed line: {line}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {}

    return word_frequencies


def validate_wordnet_subtree(root_synset: 'wn.Synset') -> bool:
    """
    Checks if a WordNet subtree starting at root_synset meets the required criteria:
    1. Has at least 20 nodes.
    2. Has at least 3 levels (Root -> Child -> Grandchild).
    """
    visited: Set['wn.Synset'] = set()
    stack: List[Tuple['wn.Synset', int]] = [(root_synset, 1)]  # (node, current_level)

    max_depth_found = 0
    node_count = 0

    while stack:
        current_node, current_level = stack.pop()

        if current_node in visited:
            continue

        visited.add(current_node)
        node_count += 1

        max_depth_found = max(max_depth_found, current_level)

        # Check for early exit
        if node_count >= 20 and max_depth_found >= 3:
            return True

        # Traverse down the hierarchy using hyponyms
        for child in current_node.hyponyms():
            stack.append((child, current_level + 1))

    print(f"Failed: Found {node_count} nodes and {max_depth_found} levels.")
    return False


def get_words_in_subtree(root_synset: 'wn.Synset') -> Set[str]:
    """
    Recursively finds all unique words (lemmas) associated with the
    given root synset and all of its hyponyms (subordinates).
    """
    all_synsets: Set['wn.Synset'] = {root_synset}
    all_words: Set[str] = set()

    queue = [root_synset]

    while queue:
        current_synset = queue.pop(0)

        for lemma_name in current_synset.lemma_names():
            # Normalize lemma: replace underscores and make lowercase
            all_words.add(lemma_name.replace('_', ' ').lower())

        for hyponym in current_synset.hyponyms():
            if hyponym not in all_synsets:
                all_synsets.add(hyponym)
                queue.append(hyponym)

    return all_words


def get_all_nodes(root_synset: 'wn.Synset') -> List['wn.Synset']:
    """
    Helper: Returns a list of all unique Synsets (nodes) in the custom sub-tree
    defined by traversing hyponyms (is-a hierarchy) from the root.
    """
    nodes: Set['wn.Synset'] = {root_synset}
    queue = [root_synset]

    while queue:
        current = queue.pop(0)
        for child in current.hyponyms():
            if child not in nodes:
                nodes.add(child)
                queue.append(child)
    return list(nodes)


def compute_synset_probabilities(root_synset: 'wn.Synset', corpus_freqs: Dict[str, int]) -> Dict['wn.Synset', float]:
    """
    Computes P(w) for every Synset w in the custom sub-tree, based ONLY on
    frequencies of words contained within that sub-tree.
    """
    all_nodes: List['wn.Synset'] = get_all_nodes(root_synset)
    all_nodes_set = set(all_nodes)

    # 1. Calculate Total Corpus Count (N) for the denominator, scoped to the sub-tree
    all_subtree_lemmas: Set[str] = set()
    for node in all_nodes:
        for lemma in node.lemma_names():
            all_subtree_lemmas.add(lemma.lower().replace('_', ' '))

    total_corpus_count: int = sum(corpus_freqs.get(lemma, 0) for lemma in all_subtree_lemmas)

    if total_corpus_count == 0:
        print("Warning: Total corpus count for the sub-tree is 0.")
        return {}

    # 2. Helper to calculate freq(w) for a specific synset (sum of all descendant lemma freqs)
    def get_synset_frequency(synset: 'wn.Synset') -> int:
        descendant_nodes: Set['wn.Synset'] = set()
        stack = [synset]
        current_freq: int = 0

        while stack:
            current = stack.pop()

            # Ensure node is part of the custom sub-tree and not processed yet
            if current in all_nodes_set and current not in descendant_nodes:
                descendant_nodes.add(current)

                # Sum frequencies for the lemmas in the current node
                for lemma in current.lemma_names():
                    current_freq += corpus_freqs.get(lemma.lower().replace('_', ' '), 0)

                # Check children (hyponyms)
                for child in current.hyponyms():
                    # Only check children that are also in our sub-tree
                    if child in all_nodes_set:
                        stack.append(child)
        return current_freq

    # 3. Calculate P(w) for every node
    synset_probabilities: Dict['wn.Synset', float] = {}

    for node in all_nodes:
        freq_w = get_synset_frequency(node)
        p_w = freq_w / total_corpus_count
        synset_probabilities[node] = p_w

    return synset_probabilities


def generate_distance_matrix(root_synset: 'wn.Synset', filename: str = "dist_matrix.csv") -> Dict[Tuple[str, str], int]:
    """
    Calculates the shortest path distance matrix for all nodes in the subtree
    and saves it to CSV. Returns a dictionary of pairwise distances.
    """
    nodes = get_all_nodes(root_synset)
    node_names = [n.name() for n in nodes]
    n_count = len(nodes)

    print(f"Found {n_count} nodes. Calculating {n_count}x{n_count} matrix...")

    matrix_data: List[List[int]] = []
    pairwise_distances: Dict[Tuple[str, str], int] = {}

    for row_node in nodes:
        row_data: List[int] = []
        name1 = row_node.name()
        for col_node in nodes:
            name2 = col_node.name()

            # Calculate shortest path distance
            dist = row_node.shortest_path_distance(col_node)

            if dist is None:
                dist = -1

            pairwise_distances[(name1, name2)] = dist
            row_data.append(dist)
        matrix_data.append(row_data)

    df = pd.DataFrame(matrix_data, index=node_names, columns=node_names)
    df.to_csv(filename)
    print(f"\nSuccess! Distance Matrix saved to '{filename}'")

    return pairwise_distances


def compute_pairwise_path_similarity(
        root_synset: 'wn.Synset',
        node_distances: Dict[Tuple[str, str], int],
        wn_api
) -> Dict[Tuple[str, str], float]:
    """
    Computes the path-based similarity between every unique pair of words
    (lemmas) in the sub-tree.

    Similarity is calculated as: Sim(w1, w2) = 1 / (min_distance(s1, s2) + 1).
    """
    all_words: Set[str] = get_words_in_subtree(root_synset)
    word_list = sorted(list(all_words))
    word_pairs = list(itertools.combinations(word_list, 2))

    print(f"Starting pairwise Path Similarity calculation for {len(word_list)} words.")

    similarity_matrix: Dict[Tuple[str, str], float] = {}
    node_names_in_matrix = set(k[0] for k in node_distances.keys())

    for word1, word2 in word_pairs:
        # Only consider NOUN senses
        synsets1 = wn_api.synsets(word1, wn_api.NOUN)
        synsets2 = wn_api.synsets(word2, wn_api.NOUN)

        if not synsets1 or not synsets2:
            similarity_matrix[(word1, word2)] = 0.0
            continue

        min_distance = float('inf')

        for s1 in synsets1:
            s1_name = s1.name()
            # Skip sense if it's not a node in the custom sub-tree
            if s1_name not in node_names_in_matrix:
                continue

            for s2 in synsets2:
                s2_name = s2.name()
                if s2_name not in node_names_in_matrix:
                    continue

                # Retrieve distance from the pre-calculated custom matrix
                distance_key = (s1_name, s2_name)
                current_dist = node_distances.get(distance_key)

                if current_dist is None:
                    # Check reverse key, matrix is symmetric but keys might be one-directional
                    current_dist = node_distances.get((s2_name, s1_name))

                if current_dist is not None and current_dist != -1:
                    min_distance = min(min_distance, current_dist)

        if min_distance != float('inf'):
            # Sim = 1 / (Distance + 1)
            path_similarity = 1.0 / (min_distance + 1.0)
        else:
            path_similarity = 0.0  # No common path found in the subtree

        similarity_matrix[(word1, word2)] = path_similarity

    return similarity_matrix


def compute_information_content(probabilities: Dict['wn.Synset', float]) -> Dict['wn.Synset', float]:
    """
    Calculates the Information Content (IC) for each synset: IC(w) = -log(P(w)).
    """
    ic_scores = {}
    for synset, p_w in probabilities.items():
        if p_w > 0:
            ic_scores[synset] = -math.log(p_w)
        else:
            ic_scores[synset] = float('inf')
    return ic_scores


def compute_resnik_similarity(
        synset1: 'wn.Synset',
        synset2: 'wn.Synset',
        ic_scores: Dict['wn.Synset', float],
        wn_api
) -> float:
    """
    Computes Resnik similarity between two Synsets: Sim_resnik(s1, s2) = IC(LCS(s1, s2)).
    Uses custom IC scores constrained to the sub-tree.
    """
    try:
        # Use lowest_common_hypernyms to find the LCS
        lcs_list = synset1.lowest_common_hypernyms(synset2)

        if not lcs_list:
            return 0.0

        lcs = lcs_list[0]

        # Resnik similarity is the IC of the LCS
        # CRUCIAL: Check if the LCS is in our *custom* IC scores (i.e., in the sub-tree)
        if lcs in ic_scores:
            return ic_scores[lcs]
        else:
            # If LCS is outside the custom sub-tree, similarity is 0.
            return 0.0

    except Exception:
        return 0.0


def compute_pairwise_resnik_for_subtree(
        root_synset: 'wn.Synset',
        corpus_freqs: Dict[str, int],
        wn_api
) -> Dict[Tuple[str, str], float]:
    """
    Computes the maximum Resnik similarity between every unique pair of words
    associated with the nodes in the custom WordNet sub-tree.
    """
    # 1. Compute custom Information Content (IC) for all nodes
    probabilities = compute_synset_probabilities(root_synset, corpus_freqs)
    ic_scores = compute_information_content(probabilities)

    # 2. Gather all unique words (lemmas) in the sub-tree
    all_words: Set[str] = get_words_in_subtree(root_synset)
    word_list = sorted(list(all_words))
    word_pairs = list(itertools.combinations(word_list, 2))

    print(f"Starting pairwise Resnik calculation for {len(word_list)} words and {len(word_pairs)} pairs.")

    similarity_matrix: Dict[Tuple[str, str], float] = {}

    # 3. Compute max similarity for every word pair
    for word1, word2 in word_pairs:
        # Find all noun synsets for each word
        synsets1 = wn_api.synsets(word1, wn_api.NOUN)
        synsets2 = wn_api.synsets(word2, wn_api.NOUN)

        if not synsets1 or not synsets2:
            similarity_matrix[(word1, word2)] = 0.0
            continue

        max_similarity = 0.0

        # Resnik similarity is the maximum IC(LCS) over all sense pairs
        for s1 in synsets1:
            for s2 in synsets2:
                similarity = compute_resnik_similarity(s1, s2, ic_scores, wn_api)
                max_similarity = max(max_similarity, similarity)

        similarity_matrix[(word1, word2)] = max_similarity

    return similarity_matrix


if __name__ == '__main__':
    # Define the root of the custom WordNet sub-tree
    ROOT_SYNSET = wn.synset('road.n.01')
    FREQ_FILE = 'corpus_ex1.freq_list'

    # 1. Validate the selected sub-tree
    result = validate_wordnet_subtree(ROOT_SYNSET)
    print(f"Does '{ROOT_SYNSET.name()}' pass criteria (>=20 nodes, >=3 levels)? {result}")

    # 2. Compute Distance Matrix and Path Similarity
    node_pair_distance = generate_distance_matrix(ROOT_SYNSET, filename="distance_matrix.csv")
    word_pair_path_sim = compute_pairwise_path_similarity(ROOT_SYNSET, node_pair_distance, wn)

    # 3. Compute Resnik Similarity
    frequency_data = read_frequency_list(FREQ_FILE)
    word_pair_resnik_sim = compute_pairwise_resnik_for_subtree(ROOT_SYNSET, frequency_data, wn)

    # 4. Display Results (Preview Distance Matrix)
    if os.path.exists('distance_matrix.csv'):
        df = pd.read_csv('distance_matrix.csv')
        print("\n--- Full Distance Matrix (First 5 Rows) ---")
        print(df.head().to_string())

    print("\n--- Example Similarity Results ---")

    # Example pairs from the solution document
    example_pairs = [
        ('bypath', 'road'),
        ('byway', 'road'),
        ('cartroad', 'road'),
        ('alley', 'road'),
        ('roadway', 'route')
    ]

    for pair in example_pairs:
        w1, w2 = pair
        # Find the similarity regardless of key order
        path_sim = word_pair_path_sim.get(pair) or word_pair_path_sim.get((w2, w1), 0.0)
        resnik_sim = word_pair_resnik_sim.get(pair) or word_pair_resnik_sim.get((w2, w1), 0.0)

        print(f"Pair: {pair} | Path Sim: {path_sim:.4f} | Resnik Sim: {resnik_sim:.4f}")