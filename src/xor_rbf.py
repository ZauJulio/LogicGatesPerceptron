from typing import Callable
import numpy as np


class RBFNetwork:
    """
    Radial Basis Function (RBF) Neural Network for solving binary classification problems such as XOR.

    args:
        centers (np.ndarray): The centers (centroids) for the RBF neurons.
        sigma (float): The spread (standard deviation) of the Gaussian functions.
        weights (np.ndarray): The weights of the output layer.
        rbf_function (Callable[[np.ndarray, np.ndarray, float], float]): The RBF function, typically a Gaussian.
    """

    def __init__(
        self,
        centers: np.ndarray,
        sigma: float,
        weights: np.ndarray = None,
        rbf_function: Callable[[np.ndarray, np.ndarray, float], float] = None,
    ):
        """
        Initialize the RBF network with specified centers and sigma.

        Args:
            centers (np.ndarray): The centers (centroids) for the RBF neurons.
            sigma (float): The spread (standard deviation) of the Gaussian functions.
        """
        self.centers = centers
        self.sigma = sigma
        self.weights = weights
        self.rbf_function = rbf_function or self.gauss_rbf

    def gauss_rbf(self, x: np.ndarray, center: np.ndarray, sigma: float) -> float:
        """
        Gaussian Radial Basis Function (RBF).

        Args:
            x (np.ndarray): Input vector.
            center (np.ndarray): Center vector.
            sigma (float): Spread of the Gaussian function.

        Returns:
            float: The computed RBF value.
        """
        # Formula: exp(-||x - center||^2 / (2 * sigma^2))
        return np.exp(-(np.linalg.norm(x - center) ** 2) / (2 * sigma**2))

    def _construct_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Construct the design matrix using the RBF function.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: The design matrix.
        """
        # Initialize the design matrix
        G = np.zeros((X.shape[0], len(self.centers)))

        for i, x in enumerate(X):
            # Compute the RBF values for each center
            for j, center in enumerate(self.centers):
                G[i, j] = self.rbf_function(x, center, self.sigma)

        return G

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the RBF network using the provided training data.

        Args:
            X (np.ndarray): Training input data.
            y (np.ndarray): Training output data.
        """
        # Construct the design matrix
        G = self._construct_design_matrix(X)

        # Solve for the weights using least-squares solution
        self.weights, _, _, _ = np.linalg.lstsq(G, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input data.

        Args:
            X (np.ndarray): Input data to predict.

        Returns:
            np.ndarray: Predicted output data.
        """
        G = self._construct_design_matrix(X)

        return G.dot(self.weights)


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    # Initialize and train the RBF network
    rbf_network = RBFNetwork(centers=np.array([[0, 1], [1, 0]]), sigma=0.5)
    rbf_network.fit(X, y)

    # Predict the outputs
    predictions = rbf_network.predict(X)
    print("Predicted XOR outputs:", np.round(predictions))
