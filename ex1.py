import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import math
import itertools


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


def get_all_nodes(root_synset):
    """Helper: Returns a list of all unique Synsets in the subtree."""
    nodes = set([root_synset])
    queue = [root_synset]

    while queue:
        current = queue.pop(0)
        for child in current.hyponyms():
            if child not in nodes:
                nodes.add(child)
                queue.append(child)
    return list(nodes)


def generate_distance_matrix(root_synset, filename="dist_matrix.csv"):
    # 1. Gather all nodes in the subtree
    print(f"Collecting nodes for subtree: {root_synset.name()}...")
    nodes = get_all_nodes(root_synset)
    node_names = [n.name() for n in nodes]
    n_count = len(nodes)

    print(f"Found {n_count} nodes. Calculating {n_count}x{n_count} matrix...")

    # Safety check for performance
    if n_count > 500:
        print("Warning: Large matrix generation might be slow.")

    # 2. Create the Matrix
    # We initialize a list of lists to store the distances
    matrix_data = []

    for row_node in nodes:
        row_data = []
        for col_node in nodes:
            # Calculate shortest path distance (edges between nodes)
            # Returns an integer, or None if they are not connected
            dist = row_node.shortest_path_distance(col_node)

            # Handle cases where no path exists (shouldn't happen in a valid subtree)
            if dist is None:
                dist = -1

            row_data.append(dist)
        matrix_data.append(row_data)

    # 3. Convert to DataFrame for display and saving
    df = pd.DataFrame(matrix_data, index=node_names, columns=node_names)

    # Print to console
    print("\n--- Distance Matrix (Preview) ---")
    print(df.to_string())

    # Save to CSV
    df.to_csv(filename)
    print(f"\nSuccess! Matrix saved to '{filename}'")

if __name__ == '__main__':
    # validate root selected is good
    root = wn.synset('game.n.01')
    result = validate_wordnet_subtree(root)
    print(f"Does '{root.name()}' pass? {result}")

    # create distance matrix
    generate_distance_matrix(root, filename="game_distance.csv")