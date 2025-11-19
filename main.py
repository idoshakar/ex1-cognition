import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import math
import itertools

# Ensure WordNet is downloaded
try:
    wn.synset('dog.n.01')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


# --- 1. Build the Sub-Tree ---

def get_subtree_nodes(root_synset, max_nodes=30):
    """
    Extracts a list of Synsets forming a sub-tree.
    """
    nodes = [root_synset]
    queue = [root_synset]

    while queue and len(nodes) < max_nodes:
        current = queue.pop(0)
        children = current.hyponyms()

        for child in children:
            if len(nodes) < max_nodes:
                if child not in nodes:
                    nodes.append(child)
                    queue.append(child)
            else:
                break
    return nodes


# --- 2. Load Frequency Data ---

def load_frequency_map(filepath):
    freq_map = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                word = parts[0]
                try:
                    count = int(parts[1])
                    freq_map[word] = count
                except ValueError:
                    continue
    return freq_map


# --- 3. Calculate Frequencies & Probabilities ---

def get_concept_freq(node, sub_tree_nodes, raw_counts_dict):
    """
    Recursive function to get freq(c) = raw_freq(c) + sum(freq(children)).
    ONLY counts children that are in the sub_tree_nodes list.
    Memoization could optimize this, but for <30 nodes, recursion is fine.
    """
    lemma = node.lemmas()[0].name().lower()
    # Get raw count, default to 0.
    # We smooth by adding 1 to total frequency later to avoid log(0)
    node_raw_count = raw_counts_dict.get(lemma, 0)

    children_sum = 0
    for child in node.hyponyms():
        if child in sub_tree_nodes:
            children_sum += get_concept_freq(child, sub_tree_nodes, raw_counts_dict)

    total_freq = node_raw_count + children_sum
    return total_freq


def get_probability(node, root_freq, sub_tree_nodes, raw_counts_dict):
    """
    Calculates P(c) = freq(c) / N
    """
    node_freq = get_concept_freq(node, sub_tree_nodes, raw_counts_dict)
    # Laplace smoothing: ensure probability is never truly 0
    if node_freq == 0:
        node_freq = 1e-10  # Small epsilon

    return node_freq / root_freq


def get_information_content(node, root_freq, sub_tree_nodes, raw_counts_dict):
    """
    IC(c) = -log(P(c))
    """
    prob = get_probability(node, root_freq, sub_tree_nodes, raw_counts_dict)
    return -math.log(prob)


# --- 4. Find LCS and Resnik Similarity ---

def get_lcs(node1, node2):
    """
    Finds the Least Common Subsumer using NLTK.
    """
    lcs_list = node1.lowest_common_hypernyms(node2)
    if lcs_list:
        return lcs_list[0]
    return None


def resnik_similarity(node1, node2, root_freq, sub_tree_nodes, raw_counts_dict):
    """
    Sim_resnik(c1, c2) = IC(LCS(c1, c2))
    """
    lcs = get_lcs(node1, node2)

    # If LCS exists, calculate its IC.
    # Note: Even if LCS is outside our sub-tree list, we calculate its freq
    # based on the sub-tree logic (which might just be 0 + children sum).
    # Ideally, for this assignment, LCS is usually the root or inside the tree.
    if lcs:
        return get_information_content(lcs, root_freq, sub_tree_nodes, raw_counts_dict)
    return 0


# --- 5. Create Matrices ---

def create_resnik_matrix(nodes, root_freq, raw_counts_dict):
    n = len(nodes)
    matrix = [[0.0] * n for _ in range(n)]
    labels = [node.lemmas()[0].name() for node in nodes]

    for i in range(n):
        for j in range(n):
            sim = resnik_similarity(nodes[i], nodes[j], root_freq, nodes, raw_counts_dict)
            matrix[i][j] = round(sim, 4)

    return pd.DataFrame(matrix, index=labels, columns=labels)


def create_distance_matrix(nodes):
    n = len(nodes)
    matrix = [[0] * n for _ in range(n)]
    labels = [node.lemmas()[0].name() for node in nodes]

    for i in range(n):
        for j in range(n):
            dist = nodes[i].shortest_path_distance(nodes[j])
            if dist is None: dist = -1
            matrix[i][j] = dist

    return pd.DataFrame(matrix, index=labels, columns=labels)


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    # 1. Setup
    root_word = 'book.n.01'  # Change this if you want a different tree
    root = wn.synset(root_word)
    sub_tree_nodes = get_subtree_nodes(root, max_nodes=20)
    freq_map = load_frequency_map('corpus_ex1.freq_list')

    print(f"Root: {root.name()}")
    print(f"Nodes in tree: {len(sub_tree_nodes)}")

    # 2. Calculate Root Frequency (N)
    root_freq = get_concept_freq(root, sub_tree_nodes, freq_map)
    if root_freq == 0: root_freq = 1  # Avoid division by zero
    print(f"Total Root Frequency (N): {root_freq}\n")

    # 3. Generate Distance Matrix (Question 1a)
    dist_df = create_distance_matrix(sub_tree_nodes)
    print("--- Distance Matrix (Edges) ---")
    print(dist_df.to_string())  # to_string() prints the whole table
    print("\n")

    # 4. Generate Resnik Similarity Matrix (Question 1b)
    resnik_df = create_resnik_matrix(sub_tree_nodes, root_freq, freq_map)
    print("--- Resnik Similarity Matrix ---")
    print(resnik_df.to_string())
    print("\n")

    # 5. Find Interesting Pairs (Helper for Question 1b)
    # This prints pairs with high Resnik but high Distance (or vice versa)
    # to help you find the "5 pairs" for your report.
    print("--- Pair Analysis (Helper for 1b) ---")
    print(f"{'Word 1':<15} {'Word 2':<15} {'Dist':<6} {'Resnik':<8}")
    print("-" * 46)

    count = 0
    for i in range(len(sub_tree_nodes)):
        for j in range(i + 1, len(sub_tree_nodes)):
            w1 = sub_tree_nodes[i].lemmas()[0].name()
            w2 = sub_tree_nodes[j].lemmas()[0].name()
            d = dist_df.iloc[i, j]
            r = resnik_df.iloc[i, j]

            # Just printing first 10 pairs as an example
            if count < 10:
                print(f"{w1:<15} {w2:<15} {d:<6} {r:<8.4f}")
                count += 1


