from pathlib import Path
import shutil
import pprint


def find_connected_nodes(node, graph, visited=None):
    if visited is None:
        visited = set()

    # Check if the node is in the graph keys or values
    if node not in visited:
        visited.add(node)
        if node in graph:
            for connected_node in graph[node]:
                find_connected_nodes(connected_node, graph, visited)
        for key, value in graph.items():
            if node in value:
                find_connected_nodes(key, graph, visited)
    return list(visited)


def remove_elements(list1, list2):
    return [item for item in list1 if item not in list2]


def get_groups_alt(my_nodes_dict, verbose = False):

    candidates_nodes = list(my_nodes_dict.keys())
    my_groups = []

    for current_node in my_nodes_dict.keys():

        if verbose:
            print(f'len: {len(candidates_nodes)}')

        if current_node not in candidates_nodes:
            if verbose:
                print(f'\tNode {current_node} already processed')
            continue

        connected_nodes = find_connected_nodes(current_node, my_nodes_dict)

        candidates_nodes = remove_elements(candidates_nodes, connected_nodes)   
        candidates_nodes.insert(0, current_node)

        my_groups.append((current_node, len(connected_nodes)))

    return my_groups


def copy_arrays_to_folder(arrays, indices, probs_dict, folder_path, verbose=False):
    """
    Copy a list of arrays to a folder, renaming them with the corresponding probability.
    """

    # Create the subfolder using pathlib
    folder_path.mkdir(parents=True, exist_ok=True)

    # Loop over the indices and copy each WAV file
    for idx in indices:
        if idx < len(arrays):

            current_prob = probs_dict[idx]

            file_path = arrays[idx]
            new_filename = Path(file_path).stem + f'_{current_prob:.2f}' + Path(file_path).suffix

            destination_path = folder_path / new_filename

            shutil.copy(file_path, destination_path)
            if verbose:
                print(f"{folder_path.name}: Copied {new_filename}")
        else:
            print(f"!!!!! \t{folder_path.name}: Index {idx} is out of range.")


def merge_and_validate_dicts(index_dict, prob_dict):
    """
    Merge two dictionaries where values are lists of indices and probabilities.
    Validates that repeated indices have consistent probability values.
    
    Args:
        index_dict (dict): Dictionary with elements as keys and lists of indices as values
        prob_dict (dict): Dictionary with elements as keys and lists of probabilities as values
        
    Returns:
        dict: Merged dictionary with indices as keys and probabilities as values
    """
    # Validate input dictionaries have the same keys
    if set(index_dict.keys()) != set(prob_dict.keys()):
        raise ValueError("Input dictionaries must have the same keys")
    
    # Validate that for each key, the lists have the same length
    for key in index_dict:
        if len(index_dict[key]) != len(prob_dict[key]):
            raise ValueError(
                f"Lists for key '{key}' have different lengths: "
                f"indices: {len(index_dict[key])}, probabilities: {len(prob_dict[key])}"
            )
    
    
    # Keep track of encountered indices for validation
    index_prob_map = {}
    prob_avg_dict = {}
    
    for element in index_dict:
        indices = index_dict[element]
        probs = prob_dict[element]
        
        # Process each index-probability pair
        for idx, prob in zip(indices, probs):
            # Check if we've seen this index before
            if idx in index_prob_map:
                # print(f"index {idx}: Found {prob} -> prev {index_prob_map[idx]}")
                index_prob_map[idx].append(prob)
            else:
                index_prob_map[idx] = [prob]
    
    # # Pretty print the index_prob_map
    # pprint.pprint(index_prob_map)

    # Calculate the average probability for each index
    for idx, probs in index_prob_map.items():
        prob_avg = sum(probs) / len(probs)
        prob_avg_dict[idx] = prob_avg
        # print(f"Index {idx}: {probs} -> avg: {prob_avg}")

                
    
    return prob_avg_dict
