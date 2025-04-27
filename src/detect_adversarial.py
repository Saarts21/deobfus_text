import re
import nltk
from nltk.corpus import words, names
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download("words")
nltk.download("names")
nltk.download("wordnet")
nltk.download("omw-1.4")
from spellchecker import SpellChecker

# initialize
english_words = set(words.words())
name_list = set(names.words())
lemmatizer = WordNetLemmatizer()
spell = SpellChecker(language='en')

adversarial_pattern = re.compile(r"^(?=.*[a-zA-Z])(?=.*[^a-zA-Z']).+$")
punctuation_marks_pattern = re.compile(r"^[!,\.?]+$")
#numbers_pattern = pattern = re.compile(r"^[0-9]+$")

def is_valid_word(word):
    """
    The input to this function is raw words from real world text.
    Returns False if it's suspicious to be adversarial leetspeak example.
    """
    # ignore one letter string (caused by the text splitting, e.g. ,.?!)
    if len(word) == 1:
        return True

    # if word ends with 's, ignore it
    if word.endswith("'s") or word.endswith("s'"):
        word = word[:-2]
    
    # ignore string with punctuation marks only
    if punctuation_marks_pattern.search(word):
        return True

    if adversarial_pattern.search(word):
        return False  # Adversarial, needs deobfuscation

    # Check original word against the English dictionary
    if word.lower() in english_words or word in name_list:
        return True
    
    # Try lemmatizing as different parts of speech
    for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
        lemma = lemmatizer.lemmatize(word.lower(), pos=pos)
        if lemma in english_words:
            return True

    # Check if it's a misspelling and can be corrected
    correction = spell.correction(word.lower())
    if correction and correction in english_words:
        return True

    # If none match, consider it adversarial
    return False

def is_valid_word_light(word):
    # ignore one letter string (caused by the text splitting, e.g. ,.?!)
    if len(word) == 1:
        return True

    # if word ends with 's, ignore it
    if word.endswith("'s") or word.endswith("s'"):
        word = word[:-2]
    
    # ignore string with punctuation marks only
    if punctuation_marks_pattern.search(word):
        return True

    if adversarial_pattern.search(word):
        return False  # Adversarial, needs deobfuscation
    
    return True

def is_word_exists(word):
    """
    The input to this function is a word comprises of only English letters,
    as part of the deobfuscation engine.
    Returns True if the word appears in the dictionary.
    """
     # Check original word against the dictionaries
    if word.lower() in english_words or word in name_list:
        return True
    
    # Try lemmatizing as different parts of speech
    for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
        lemma = lemmatizer.lemmatize(word.lower(), pos=pos)
        if lemma in english_words:
            return True
    
    return False


# some test cases
assert is_valid_word("eats") == True, "Test case 'eats' failed"
assert is_valid_word("walked") == True, "Test case 'walked' failed"
assert is_valid_word("Jane") == True, "Test case 'Jane' failed"
assert is_valid_word("running") == True, "Test case 'running' failed"
assert is_valid_word("jumped") == True, "Test case 'jumped' failed"
assert is_valid_word("quickly") == True, "Test case 'quickly' failed"
assert is_valid_word("bicycle") == True, "Test case 'bicycle' failed"
assert is_valid_word("giraffe") == True, "Test case 'giraffe' failed"
assert is_valid_word("elephant") == True, "Test case 'elephant' failed"
assert is_valid_word("happiness") == True, "Test case 'happiness' failed"
assert is_valid_word("sincerely") == True, "Test case 'sincerely' failed"
assert is_valid_word("John's") == True, "Test case 'John's' failed"

# Misspelled words (should be corrected)
assert is_valid_word("Bagle") == True, "Test case 'Bagle' failed"
assert is_valid_word("valdiation") == True, "Test case 'valdiation' failed"
assert is_valid_word("recieve") == True, "Test case 'recieve' failed"
assert is_valid_word("thorough") == True, "Test case 'thorough' failed"
assert is_valid_word("accomodate") == True, "Test case 'accomodate' failed"
assert is_valid_word("definately") == True, "Test case 'definately' failed"
assert is_valid_word("seprate") == True, "Test case 'seprate' failed"
assert is_valid_word("occured") == True, "Test case 'occured' failed"
assert is_valid_word("untill") == True, "Test case 'untill' failed"

# Adversarial inputs (should return False)
assert is_valid_word("!3a9le") == False, "Test case '!3a9le' failed"
assert is_valid_word("h3ll0") == False, "Test case 'h3ll0' failed"
assert is_valid_word("c@t") == False, "Test case 'c@t' failed"
assert is_valid_word("123abc") == False, "Test case '123abc' failed"
assert is_valid_word("!@#$%^&*") == False, "Test case '!@#$%^&*' failed"
assert is_valid_word("abcdefg") == False, "Test case 'abcdefg' failed"
assert is_valid_word("12345") == False, "Test case '12345' failed"