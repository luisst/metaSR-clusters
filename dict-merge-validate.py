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
    
    # Initialize result dictionary
    result = {}
    
    # Keep track of encountered indices for validation
    index_prob_map = {}
    
    for element in index_dict:
        indices = index_dict[element]
        probs = prob_dict[element]
        
        # Process each index-probability pair
        for idx, prob in zip(indices, probs):
            # Check if we've seen this index before
            if idx in index_prob_map:
                if not abs(index_prob_map[idx] - prob) < 1e-10:  # Using small epsilon for float comparison
                    raise ValueError(
                        f"Inconsistent probability values for index {idx}: "
                        f"Found {prob} but previously saw {index_prob_map[idx]}"
                    )
            else:
                index_prob_map[idx] = prob
                
            result[idx] = prob
    
    return result

# Example usage
if __name__ == "__main__":
    # Example input dictionaries with lists
    index_dict = {
        'A': [1, 2, 3],
        'B': [4, 5, 1],  # Note: 1 is repeated
        'C': [6, 7, 8]
    }
    
    prob_dict = {
        'A': [0.25, 0.30, 0.15],
        'B': [0.10, 0.05, 0.25],  # 0.25 matches index 1 from 'A'
        'C': [0.05, 0.05, 0.05]
    }
    
    print("\nTest Case 1: Valid input")
    print("Index dictionary:", index_dict)
    print("Probability dictionary:", prob_dict)
    
    try:
        result = merge_and_validate_dicts(index_dict, prob_dict)
        print("Merged dictionary:", result)
        
    except ValueError as e:
        print("Error:", str(e))

    # Example with inconsistent probabilities
    prob_dict_inconsistent = {
        'A': [0.25, 0.30, 0.15],
        'B': [0.10, 0.05, 0.35],  # 0.35 doesn't match 0.25 for index 1
        'C': [0.05, 0.05, 0.05]
    }
    
    print("\nTest Case 2: Inconsistent probabilities")
    try:
        result = merge_and_validate_dicts(index_dict, prob_dict_inconsistent)
        print("Merged dictionary:", result)
        
    except ValueError as e:
        print("Error:", str(e))

    # Example with different list lengths
    index_dict_invalid = {
        'A': [1, 2, 3],
        'B': [4, 5],  # Different length
        'C': [6, 7, 8]
    }
    
    print("\nTest Case 3: Different list lengths")
    try:
        result = merge_and_validate_dicts(index_dict_invalid, prob_dict)
        print("Merged dictionary:", result)
        
    except ValueError as e:
        print("Error:", str(e))
