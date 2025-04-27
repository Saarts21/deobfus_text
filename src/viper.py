from letters import leet_speak_alphabet
import random

def viper(text, p):
    """
    This function takes a text string and a probability p, and replaces characters in the text with 
    their leet speak equivalents based on the given probability.

    Parameters:
    text (str): The input text string to be transformed.
    p (float): The probability with which each character in the text will be replaced with its leet speak equivalent.

    Returns:
    str: The transformed text string with some characters replaced by their leet speak equivalents.
    """
    # transform text to list of single characters
    chars_list = list(text)
    result_chars = []

    for orig_c in chars_list:
        rand = random.random()
        
        if orig_c.upper() in leet_speak_alphabet and rand < p:
            replacement_c = random.choice(leet_speak_alphabet[orig_c.upper()])
            result_chars.append(replacement_c)
        else:
            result_chars.append(orig_c)
    
    return ''.join(result_chars)
