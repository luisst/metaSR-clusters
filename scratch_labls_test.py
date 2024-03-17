def filter_test_values(input_list):
    input_list_train = [x for x in input_list if x < 100]
    input_list_test = [x for x in input_list if x >= 100 ]
    
    for value in input_list_test:
        if not(value - 100 in input_list_train):
            input_list_train.append(value)
    
    return sorted(input_list_train)

# Example usage:
input_list = [2, 103, 5, 106, 101, 1, 4, 3, 100]
filtered_list = filter_values(input_list)
print(filtered_list)