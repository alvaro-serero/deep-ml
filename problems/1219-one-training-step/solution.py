import torch

def train_step(model, x, y, optimizer, loss_fn):
    """Run one training step and return the pre-update loss as a float.

    Args:
        model: torch.nn.Module to train.
        x: Input batch tensor.
        y: Target batch tensor.
        optimizer: torch.optim optimizer bound to model parameters.
        loss_fn: Callable (pred, y) -> scalar loss tensor.

    Returns:
        float: Loss value computed before optimizer.step().
    """
    # TODO: zero_grad -> forward -> loss -> backward -> step; return float loss
    # Zero out optimizer's gradients
    optimizer.zero_grad()

    # Forward pass
    pred = model(x)

    # Loss
    loss = loss_fn(pred, y)

    # Backward pass
    loss.backward()

    # Step
    optimizer.step()

    # Return float loss
    return float(loss.item())
