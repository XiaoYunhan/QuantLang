import numpy as np
from scipy.linalg import solve_banded

class CrankNicolsonSolver:
    def __init__(self, riskfree, dividend, volatility, strike, maturity, option_type="put", N=250, M=250):
        self.r = riskfree
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.option_type = option_type.lower()  # "put" or "call"
        self.Smax = 3 * strike
        self.x_min = -5  # S -> 0
        self.x_max = 5   # S -> inf
        self.N = N  # Number of grid points
        self.max_dt = maturity/M
        self.USE_PSOR = False
        self.tol = 1e-5
        self.max_iter = 200
        self.omega = 1.2
        self.cached_dt = 0
        self.err = 0
        self.iter = 0

        self.x = np.linspace(self.x_min, self.x_max, self.N)  # log price grid
        self.S = np.exp(self.x)  # Convert x to S
        self.A = np.zeros((self.N, self.N))
        self.b = np.zeros((self.N, 1))
        self.X = np.zeros(self.N)

    def solve(self, S0):
        self.setInitialCondition()
        self.solvePDE()
        prices = np.exp(self.x)  # Convert back from log prices to actual prices
        return np.interp(S0, prices, self.X)

    def solvePDE(self):
        t = self.T
        while t > 0:
            dt = min(t, self.max_dt)
            self.setCoeff(dt ,t)
            if self.USE_PSOR:
                self.solvePSOR()
            else:
                self.solveLinearSystem()
            t -= dt
            print(f"t = {t:.5f}, err = {self.err:.5e}, iters = {self.iter}")

    def setInitialCondition(self):
        if self.option_type == "put":
            self.X = np.maximum(self.K - self.S, 0)  # Put Option Payoff
        elif self.option_type == "call":
            self.X = np.maximum(self.S - self.K, 0)  # Call Option Payoff

    def setCoeff(self, dt, t):
        N = self.N
        dx = self.x[1] - self.x[0]
        alpha = 0.25 * dt * (self.sigma**2 / dx**2)
        beta = 0.25 * dt * ((self.r - self.q - 0.5 * self.sigma**2) / dx)
        gamma = 0.5 * dt * self.r

        for i in range(1, N - 1):  # Corrected to avoid out-of-bounds indexing
            self.A[i, i - 1] = -alpha + beta  # Left neighbor
            self.A[i, i] = 1 + 2 * alpha + gamma  # Center
            self.A[i, i + 1] = -alpha - beta  # Right neighbor

            self.b[i] = alpha * self.X[i - 1] + (1 - 2 * alpha - gamma) * self.X[i] + alpha * self.X[i + 1]

        # Boundary conditions
        if self.option_type == "put":
            self.A[0, 0] = 1
            self.A[0, 1] = 0
            self.b[0] = self.K * np.exp(-self.r * t) - self.S[0]  # Left boundary (S -> 0)

            self.A[-1, -1] = 1
            self.A[-1, -2] = 0
            self.b[-1] = 0  # Right boundary (S -> ∞)

        elif self.option_type == "call":
            self.A[0, 0] = 1
            self.A[0, 1] = 0
            self.b[0] = 0  # Left boundary (S -> 0)

            self.A[-1, -1] = 1
            self.A[-1, -2] = 0
            self.b[-1] = self.S[-1] - self.K * np.exp(-self.r * t)  # Right boundary (S -> ∞)
        else:
            raise ValueError("Invalid option type")

    def solveLinearSystem(self):
        # Extract diagonals for banded solver
        lower = np.diag(self.A, k=-1)
        main = np.diag(self.A, k=0)
        upper = np.diag(self.A, k=1)
        ab = np.zeros((3, self.N))
        ab[0, 1:] = upper
        ab[1, :] = main
        ab[2, :-1] = lower
        # Solve using banded solver
        self.X = solve_banded((1, 1), ab, self.b.flatten())
        # self.X = np.linalg.solve(self.A, self.b).flatten()

    def solvePSOR(self):
        N = self.N
        iter = 0
        omega = self.omega
        self.err = 1e10
        while self.err > self.tol and iter < self.max_iter:
            iter += 1
            x_old = self.X.copy()
            for i in range(1, N - 1):
                self.X[i] = (1 - omega) * self.X[i] + omega / self.A[i, i] * (
                    self.b[i] - self.A[i, i - 1] * self.X[i - 1] - self.A[i, i + 1] * self.X[i + 1]
                )
            # Boundary conditions
            self.X[0] = self.b[0]
            self.X[-1] = self.b[-1]
            self.applyConstraint()
            self.err = np.linalg.norm(self.X - x_old, ord=np.inf)
            self.iter = iter

    def applyConstraint(self):
        if self.option_type == "put":
            self.X = np.maximum(self.X, self.K - self.S)  # Early exercise constraint for Put
        elif self.option_type == "call":
            self.X = np.maximum(self.X, self.S - self.K)  # Early exercise constraint for Call