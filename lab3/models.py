import torch


def loss_g(A, X, labels, eps=1e-12):
    """
    Computes the NCA loss for a given transformation matrix A, input data X, and labels.

    Args:
        A (torch.Tensor): Transformation matrix of shape (r, d).
        X (torch.Tensor): Data matrix of shape (d, n) where each column is a data point.
        labels (torch.Tensor): Labels tensor of shape (n,), with integer class labels.
        eps (float): A small value to prevent division by zero.

    Returns:
        torch.Tensor: The computed loss (scalar).
    """
    # Transform data: Y = A * X, resulting in shape: (r, n)
    X_prime = torch.matmul(A, X)
    X_prime_t = X_prime.t()  # shape: (n, r)

    # Compute squared Euclidean distances between all pairs of transformed points:
    # ||X_prime_i - X_prime_j||^2 = ||X_prime_i||^2 + ||X_prime_j||^2 - 2 * X_prime_i^T X_prime_j
    norm_sq = (X_prime_t ** 2).sum(dim=1, keepdim=True)  # shape: (n, 1)
    D = norm_sq + norm_sq.t() - 2 * torch.matmul(X_prime_t, X_prime_t.t())

    # Exclude self-comparison by setting the diagonal to a large value:
    n = X.shape[1]
    D = D + torch.eye(n, device=D.device) * 1e12

    # Stabilize the softmax: subtract the minimum value in each row from D.
    D_min, _ = D.min(dim=1, keepdim=True)
    D_stable = D - D_min

    # Compute the affinity using the Gaussian kernel: f(i,j) = exp(-D_stable(i,j))
    f = torch.exp(-D_stable)

    # Normalize to obtain probabilities p_{ij}
    sum_f = f.sum(dim=1, keepdim=True) + eps
    p = f / sum_f

    # Create a mask: mask[i, j] = 1 if labels[i] == labels[j], otherwise 0
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

    # Compute the objective: g(A) = sum_{i,j with same label} p_{ij}
    g_val = (p * mask).sum()

    # Since we maximize g(A), the loss is defined as the negative of g(A)
    loss = -g_val
    return loss


class NCA:

    def __init__(self, lr=0.05, n_epochs=1000, optimizer_name="Adam", device=None):

        self.lr = lr
        self.n_epochs = n_epochs
        self.optimizer_name = optimizer_name
        self.device = device if device is not None else torch.device("cpu")
        self.A = None
        self.loss_history = []
        self.A_history = []

    def fit(self, x_train, y_train, r, verbose=False):

        x_train_tensor = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        self.A = torch.nn.Parameter(torch.randn(r, x_train_tensor.shape[0], dtype=torch.float32, device=self.device))

        if self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD([self.A], lr=self.lr)
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam([self.A], lr=self.lr)
        else:
            raise ValueError("Unsupported optimizer. Choose 'SGD' or 'Adam'.")

        # Reset histories
        self.loss_history = []
        self.A_history = []

        # Main training loop
        for epoch in range(self.n_epochs):
            loss = loss_g(self.A, x_train_tensor, y_train_tensor)

            # Backward pass: compute gradients.
            loss.backward()

            # Update parameters.
            optimizer.step()
            optimizer.zero_grad()

            # Record training history every 10 epochs.
            if epoch % 10 == 0:
                self.loss_history.append(loss.item())
                if verbose:
                    print(f"Epoch {epoch}: loss = {loss.item():.4f}")

        return self

    def transform(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        return torch.matmul(self.A, x_tensor)
