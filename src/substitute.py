import torch
import torch.nn.functional as F
from resnet18 import ResNet18
from char2img import *
from letters import *
import math
import re

class SubstModel:
    def __init__(self, device, model, mean, std):
        self.device = device
        self.model = model
        self.mean = mean
        self.std = std

    def substitute(self, text):
        """
        Returns most probable predictions as list of tuples: (lowercase letter, probability).
        """
        predictions = []
        array = generate_chars_image(text)
        tensor = torch.tensor(array).unsqueeze(0).unsqueeze(0).float().to(self.device)
        tensor = (tensor - self.mean) / self.std
        
        self.model.eval()
        with torch.no_grad():
            # Inference
            logits = self.model(tensor)

            # Extract only the logits corresponding to letters
            letter_logits = logits[:, :ABC_SIZE]
            
            # Calculate softmax probabilities only on letter logits
            probas = F.softmax(letter_logits, dim=1)

            # predict the maximum likelihood substitution
            predicted_label = torch.argmax(probas).item()
            predicted_prob = probas[0][predicted_label]

            predictions.append((chr(predicted_label + ord('a')), predicted_prob.item()))

            # other labels with relatively high probability
            for i in range(len(probas[0])):
                if probas[0][i] > (predicted_prob * 0.5) and i != predicted_label:
                    label = chr(i + ord('a'))
                    prob = probas[0][i]
                    predictions.append((label, prob.item()))

        return predictions

def load_substitution_model(experiment_name):
    # set device
    if torch.cuda.is_available():
        print("using Cuda")
        device = torch.device("cuda")
    elif torch.backends.mps.is_built():
        print("using MPS")
        device = torch.device("mps")
    else:
        print("using CPU")
        device = torch.device("cpu")

    # load pre-trained model
    model = ResNet18()
    model.load_state_dict(torch.load(f'models/resnet18_{experiment_name}.pth'))
    model = model.to(device)

    # compute the mean and std of the training set
    dataset = LettersDataset(f"data/letters_{experiment_name}.csv", device)
    mean = torch.mean(torch.stack([array.mean() for array, _ in dataset]))
    std = torch.std(torch.stack([array.std() for array, _ in dataset]))

    # init SubstModel
    subst_model = SubstModel(device, model, mean, std)
    return subst_model


def substitute_split(split, subst_model):
    """
    This function takes a list of character splits and a substitution model, and returns all possible
    substituted words along with their log probabilities sum as a likelihood score.
    High score = high confidence in the substitution.
    Low score = model is unsure (e.g., when substitutions are weird or unexpected).
    
    Args:
        split (list): List of character splits.
        subst_model (SubstModel): The substitution model to use.
    
    Returns:
        list: A list of tuples, each containing a possible substituted word and its score.
    """
    if len(split) == 0:
        return [("", float('inf'))]  # No splits, max uncertainty
    
    results = [("", 0)]  # Start with empty word and 0 log probability
    
    for chars in split:
        new_results = []
        
        # substitute only strings that contains any character that is not an English letter
        if bool(re.search(r"[^a-zA-Z]", chars)):
            substitutions = subst_model.substitute(chars)  # Now returns list of (letter, prob) tuples
            
            for result_word, result_log_prob in results:
                for letter, prob in substitutions:
                    # Handle zero probability gracefully
                    if prob == 0:
                        prob = 1e-10  # Smoothing to avoid log(0)
                    
                    new_word = result_word + letter
                    new_log_prob = result_log_prob + math.log(prob)
                    new_results.append((new_word, new_log_prob))
        else:
            # Just append the character as is with probability 1.0
            for result_word, result_log_prob in results:
                new_word = result_word + chars
                new_results.append((new_word, result_log_prob))
        
        results = new_results

    return results