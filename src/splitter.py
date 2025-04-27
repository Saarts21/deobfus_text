import re

MAX_SUBSTRING_LENGTH = 3


def split_text_to_strings(text):
    # Split the text by spaces, keeping the spaces in the list
    words = re.split(r'(\s+)', text)
    
    # Further split words that end with .,:!?
    result = []
    for word in words:
        # Use regex to split the word if it ends with punctuations
        parts = re.split(r"([.,:!?]+)$", word)
        # Filter out empty strings from the split result
        parts = [part for part in parts if part]
        # Add the parts to the result list
        result.extend(parts)
    
    return result

def split_string_to_substrings(str):
    """
    Generates all possible splits of str with substrings of up to length max_substring_length.
    Returns a list of lists, where each inner list represents a possible split of str.
    """
    def backtrack(index, path):
        if index == len(str):
            result.append(path[:])
            return
        
        for length in range(1, MAX_SUBSTRING_LENGTH + 1):
            if index + length <= len(str):
                backtrack(index + length, path + [str[index:index + length]])
    
    result = []
    backtrack(0, [])
    return result