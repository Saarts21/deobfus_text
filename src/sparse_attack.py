import torch
import torch.nn as nn

def l0_pgd_attack(x, y, model, k, alpha, steps):
    """
    Performs an L0 Projected Gradient Descent (PGD) attack on the input data.
    This attack modifies the input `x` to maximize the loss of the model while
    restricting the number of modified elements (L0 norm constraint). The attack
    iteratively updates a perturbation `delta` by selecting the top-k elements
    with the highest gradient magnitudes and applying gradient steps only at
    those locations.
    Args:
        x (torch.Tensor): The input tensor to be perturbed. Shape: (batch_size, ...).
        y (torch.Tensor): The true labels corresponding to the input `x`. Shape: (batch_size,).
        model (torch.nn.Module): The model to be attacked. Should output logits.
        k (int): The number of elements to modify in each input (L0 constraint).
        alpha (float): The step size for the gradient update.
        steps (int): The number of iterations to perform the attack.
    Returns:
        torch.Tensor: The final perturbation `delta` after the attack. Shape matches `x`.
    Note:
        - The perturbation `delta` is clipped to the range [-1, 1] after each update.
        - The function assumes that the input `x` and the model's output are compatible
          with the CrossEntropyLoss.
    """
    delta = torch.zeros_like(x).requires_grad_(True)
    for t in range(steps):
        logits = model(x + delta)

        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        grad = delta.grad.data

        # Flatten and get top-k absolute gradient locations
        abs_grad = grad.abs().view(grad.shape[0], -1)
        topk_indices = abs_grad.topk(k, dim=1).indices

        # Zero out everything not in top-k
        mask = torch.zeros_like(abs_grad)
        mask.scatter_(1, topk_indices, 1)
        mask = mask.view_as(delta)

        # Gradient step only at selected locations
        delta = delta + alpha * grad * mask
        delta = delta.detach().clamp(-1, 1).requires_grad_(True)  # clip if needed
    return delta.detach()
