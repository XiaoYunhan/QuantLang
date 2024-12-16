import numpy as np

class ChebyshevInterpolator:
    """
    This class handles the computation of Chebyshev interpolation nodes
    and collocation times for American option pricing.

    Attributes:
    - n (int): Number of interpolation nodes.
    - tau_max (float): Maximum time for the collocation grid.
    - x_nodes (numpy array): Computed Chebyshev nodes.
    - tau_nodes (numpy array): Computed collocation times.
    """

    def __init__(self, n, tau_max):
        self.n = n
        self.tau_max = tau_max
        self.z_nodes = None
        self.x_nodes = None
        self.tau_nodes = None

    def compute_nodes(self):
        """
        Compute the Chebyshev interpolation nodes and collocation times.
        """
        # Compute Chebyshev extrema points
        z = np.cos(np.pi * np.arange(self.n + 1) / self.n)
        self.z_nodes = z
        
        # Compute interpolation nodes
        self.x_nodes = np.sqrt(self.tau_max)/2 * (1 + z)
        
        # Compute collocation times
        self.tau_nodes = self.x_nodes ** 2

    def get_nodes(self):
        """
        Retrieve the computed Chebyshev nodes and collocation times.
        """
        if self.x_nodes is None or self.tau_nodes is None:
            raise ValueError("Nodes have not been computed yet. Call compute_nodes() first.")
        
        return self.x_nodes, self.tau_nodes

