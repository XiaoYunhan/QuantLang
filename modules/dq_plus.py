import numpy as np
from scipy.stats import norm
from scipy.interpolate import BarycentricInterpolator
from scipy.integrate import quad
from chebyshev_interpolator import ChebyshevInterpolator
from Option import OptionType, EuropeanOption
from quadrature_nodes import QuadratureNodes
from utils import QDplus
from enum import Enum

class QuadratureType(Enum):
    Gauss_Legendre = 'GL'
    tanh_sinh = 'TS'

class AmericanOptionPricing:
    """
    The core class for American option pricing using DQ+ method.
    This class handles initialization of exercise boundary, fixed-point iteration, and interpolation.
    """

    def __init__(self, K, r, q, vol, tau_max, l, m, n, p,
                 option_type=OptionType.Put,
                 quadrature_type=QuadratureType.tanh_sinh,
                 eta=0.5
    ):
        """
        Initialize the DQPlus engine with option parameters and collocation nodes.

        Parameters:
        - K (float): Strike price.
        - r (float): Risk-free interest rate.
        - q (float): Dividend yield.
        - vol (float): Volatility of the underlying asset.
        - tau_nodes (numpy array): Collocation time points (tau_i).
        """
        self.K = K
        self.r = r
        self.q = q
        self.vol = vol
        self.n = n
        self.m = m
        self.l = l
        self.p = p
        self.eta = eta
        self.option_type = option_type
        self.X = self.K * min(1, self.r/self.q)
        self.tau_max = tau_max
        self.iteration_no = 0

        self.initial_boundary = np.zeros(self.n+1)
        self.updateded_boundary = np.zeros(self.n+1)
        self.updateded_boundary_q_values = None
        self.chebyshev_interpolator = ChebyshevInterpolator(n, tau_max)
        self.chebyshev_interpolator.compute_nodes()
        self.tau_nodes = self.chebyshev_interpolator.get_nodes()[1]  # Get tau_nodes
        self.chebyshev_coefficients = None
        self.B_yk = None
        self.k1 = np.zeros(self.n)
        self.k2 = np.zeros(self.n)
        self.k3 = np.zeros(self.n)
        self.N_values = np.zeros(self.n)
        self.D_values = np.zeros(self.n)
        self.f_values = np.zeros(self.n)
        self.N_prime = np.zeros(self.n)
        self.D_prime = np.zeros(self.n)
        self.f_prime = np.zeros(self.n)

        # Initialize quadrature nodes
        self.quadrature = QuadratureNodes(l)
        self.quadrature.compute_legendre_nodes()
        self.y_nodes, self.w_weights = self.quadrature.get_nodes_and_weights()

        # Initialize quadrature nodes for calculation of American Option Price
        self.pricing_quadrature = QuadratureNodes(self.p)
        if quadrature_type == QuadratureType.Gauss_Legendre:
            self.pricing_quadrature.compute_legendre_nodes()
        else:
            self.pricing_quadrature.compute_tanh_sinh_nodes_weights()
        self.yp_nodes, self.wp_weights = self.pricing_quadrature.get_nodes_and_weights()
        self.u = 0.5*self.tau_max*(1+self.yp_nodes)
        self.Bu_pricing = None

    ## For American Option pricing

    ### Transform (4) to be interval of [-1,1]
    def compute_pricing_points(self):
        H_values = self.compute_H()
        self.initialize_chebyshev_interpolation(H_values)
        qp_interpolated = np.zeros(len(self.yp_nodes))
        self.z = 2 * np.sqrt(self.u / self.tau_max) - 1 # From the tau obtain the z value
        z_pricing = np.zeros(len(self.yp_nodes))
        for i in range(len(self.yp_nodes)):
            z_pricing[i] = max(self.clenshaw_algorithm(self.z[i], self.chebyshev_coefficients),0)
        self.Bu_pricing = self.q_to_B(z_pricing)        

    def compute_pricing_integral_1(self,S):
       return np.exp(-self.r*(self.tau_max - self.u))*norm.cdf(-self.d2(self.tau_max -self.u, S/self.Bu_pricing))

    def compute_pricing_integral_2(self,S):        
       return np.exp(-self.q*(self.tau_max - self.u))*norm.cdf(-self.d1(self.tau_max -self.u, S/self.Bu_pricing))
    
    def compute_option_pricing(self,S):
        # Step 1: Compute European option price
        tau = self.tau_max

        european_price = EuropeanOption.european_put_value(self.tau_max, S, self.r, self.q, self.vol, self.K)

        integral1 = self.compute_pricing_integral_1(S)
        integral2 = self.compute_pricing_integral_2(S)

        american_premium = self.r*self.K*tau*0.5*sum(self.wp_weights*integral1) - self.q*S*tau*0.5*sum(self.wp_weights*integral2)
        return european_price, american_premium


    def initialize_boundary(self):
        """
        Initialize the early exercise boundary using QD+ approximation and compute the Chebyshev_coefficients (a_k)
        """
        # Initialize B^(0)(tau_0)
        # B_tau_0 = self.K * min(1, self.r / self.q) if self.q > 0 else self.K
        # ---Adjust for B_tau_0 to improve precision---
        if self.option_type == OptionType.Call:
            # For Call options: Early exercise happens typically when r > q
            if self.r > self.q:
                B_tau_0 = self.K * (self.r / self.q)
            else:
                B_tau_0 = self.K
        elif self.option_type == OptionType.Put:
            # For Put options: Early exercise happens typically when r < q
            if self.r < self.q:
                B_tau_0 = self.K * (self.r / self.q)
            else:
                B_tau_0 = self.K
        else:
            raise ValueError("Invalid option type")
        
        a=QDplus(self.r, self.q, self.vol, self.K, self.option_type)

        for i in range(len(self.tau_nodes)):
            self.initial_boundary[i] = a.compute_exercise_boundary(self.tau_nodes[i])
        
        # make copy of the intial boundary for checking later
        self.updateded_boundary = self.initial_boundary

    def compute_H(self):
        """
        Compute H(sqrt(tau)) for each collocation point based on the previous boundary values.

        Returns:
        - H_values (numpy array): The computed H(sqrt(tau)) values.
        """
        H_values = np.square((np.log(self.updateded_boundary / self.X)))
        self.updateded_boundary_q_values = H_values
        return H_values
    
    ## compute b(q_values) to obtain the reverse 
    def q_to_B(self, q):
        """
        Given that H_values = q
        From q/H_values calculate B

        Returns:
        - B_values (numpy array): The computed B(tau) values.
        """
        B_values = np.zeros(len(q))
        for i in range(len(q)):
            if q[i] < 1: ### When q_value is less than 1 it means that log(B(tau)/X) was negative value
                B_values[i] = self.X * np.exp(-np.sqrt(q[i]))
            else:
                B_values[i] = self.X * np.exp(np.sqrt(q[i]))

        return B_values

    
    def get_boundary_values(self):
        """
        Retrieve the computed exercise boundary values.

        Returns:
        - initial_boundary (numpy array): Refined exercise boundary values.
        """
        return self.updateded_boundary
    
    def get_boundary_values_q(self):
        """
        Retrieve the computed exercise boundary values in q(z).

        Returns:
        - initial_boundary (numpy array): Refined exercise boundary values.
        """
        return self.updateded_boundary_q_values
    
    def initialize_chebyshev_interpolation(self, q):
        n = self.n
        a = np.zeros(n + 1)
        for k in range(n + 1):
            ans = 0
            for i in range(n + 1):
                term = q[i] * np.cos(i * k * np.pi / n)
                if i == 0 or i == n:
                    term *= 0.5
                ans += term
            ans *= (2.0 / n)
            a[k] = ans
        self.chebyshev_coefficients = a

    def clenshaw_algorithm(self, z, a_coefficients):
        """
        Evaluate the Chebyshev polynomial using Clenshaw's algorithm.

        Parameters:
        - z (float): The input value to evaluate the polynomial.

        Returns:
        - (float): The evaluated value of the Chebyshev polynomial.
        """
        b0 = self.chebyshev_coefficients[self.n] * 0.5
        b1 = 0
        b2 = 0

        for k in range(self.n - 1, -1, -1):
            b1, b2 = b0, b1
            b0 = a_coefficients[k] + 2 * z * b1 - b2
        return 0.5 * (b0 - b2)    
    
    def evaluate_boundary(self):
        self.B_yk = np.zeros((len(self.tau_nodes), len(self.y_nodes))) 
        for i,tau in enumerate(self.tau_nodes):
            ## For each tau we hope to obtain a series of nodes
            qz_interpolated = np.zeros(len(self.y_nodes))
            
            # Use the quadrature nodes obtained in the quadacture earlier
            for k, y_k in enumerate(self.y_nodes):
                adjusted_tau = tau - tau * (1 + y_k)**2 / 4              # Obtain the adjusted tau for each y_k, but the adjusted tau cannot be too small
                z = 2 * np.sqrt(adjusted_tau / self.tau_max) - 1             # From the tau obtain the z value
                qz_interpolated[k] = max(self.clenshaw_algorithm(z, self.chebyshev_coefficients),0) # Interpolate qz using the clenshaw algorithm

            self.B_yk[i] = self.q_to_B(qz_interpolated)                                    # q(z) = H(xi) to back out B_value for the z

    # Before Calculating the Integrand ensure that d1 and d2 can be calculated / Also in the option class
    # riskfree, dividend, strike, volatility should be defined externally, so the funciton depends only on tau and s0
    def d1(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau + 0.5 * self.vol * self.vol * tau)/(self.vol * np.sqrt(tau))

    def d2(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau - 0.5 * self.vol * self.vol * tau)/(self.vol * np.sqrt(tau))
    
    # Obtain K1 for each tau(i)

    def K1_integrad (self, tau, B, yk_nodes, B_y):
        k = len(self.y_nodes)
        K1integrads = np.zeros(k)

        for i,yk in enumerate(yk_nodes):
            term0 = tau * (1 + yk)**2 / 4
            term1 = np.exp(-self.q * term0)
            term2 = (1 + yk)
            term3 = norm.cdf(self.d1(term0, B / B_y[i]))
            # Debugging prints
            K1integrads[i] = term1 * term2 * term3
        return K1integrads

    def K1(self):
        for i in range(self.n):
            tau = self.tau_nodes[i]
            integrad = self.K1_integrad(tau, self.updateded_boundary[i], self.y_nodes, self.B_yk[i])
            self.k1[i] = tau*np.exp(self.q*tau)*0.5*sum(integrad*self.w_weights)

    def K2_integrad (self, tau, B_tau, yk_nodes, B_y):
        k = len(yk_nodes)
        K2integrads = np.zeros(k)
    
        for i in range(k):
            term0 = tau / 4 * (1 + yk_nodes[i])**2
            term1 = (np.exp(-self.q * term0))/self.vol
            term2 = norm.pdf(self.d1(term0, B_tau/B_y[i]))
            K2integrads[i] = term1*term2
        return K2integrads

    def K2(self):
        for i in range(self.n):
            tau = self.tau_nodes[i]
            integrad = self.K2_integrad(tau, self.updateded_boundary[i], self.y_nodes, self.B_yk[i])
            self.k2[i] = np.sqrt(tau)*np.exp(self.q*tau)*sum(integrad*self.w_weights)

    def K3_integrand(self, tau, B_tau, yk_nodes, B_y):
        k = len(yk_nodes)
        K3integrads = np.zeros(k)

        for i in range(k):
            term0 = 0.25 * tau *(1 + yk_nodes[i])**2
            term1 = (np.exp(-self.r * term0))/self.vol
            term2 = norm.pdf(self.d2(term0, B_tau/B_y[i]))
            K3integrads[i] = term1*term2
        return K3integrads

    def K3(self):
        for i in range(self.n):
            tau = self.tau_nodes[i]
            integrad = self.K3_integrand(tau, self.updateded_boundary[i], self.y_nodes, self.B_yk[i])
            self.k3[i] = np.sqrt(tau)*np.exp(self.r*tau)*sum(integrad*self.w_weights)

    def compute_ND_values(self):
        """
        Compute N(tau, B) and D(tau, B) from computed k1,k2,k3
        """
        N_values = np.zeros(self.n )
        D_values = np.zeros(self.n )

        for i in range(self.n):
            tau = self.tau_nodes[i]
            B_tau = self.updateded_boundary[i]

            N_values[i] = norm.pdf(self.d2(tau, B_tau/self.K))/(self.vol*np.sqrt(tau)) + self.r*self.k3[i]
            D_values[i] = norm.cdf(self.d1(tau, B_tau/self.K)) + norm.pdf(self.d1(tau, B_tau/self.K))/(self.vol*np.sqrt(tau)) + self.q*(self.k1[i] + self.k2[i])

        self.N_values = N_values
        self.D_values = D_values
    
    def compute_f_values(self):
        """
        Compute f(tau, B) values based on N and D.
        """
        f_values =  self.K*np.exp(-(self.r-self.q)*self.tau_nodes[:-1])*self.N_values/self.D_values
        self.f_values =  f_values

    ## Approximation of NPrime
    def Nprime(self):
        self.N_prime = -self.d2(self.tau_nodes[:-1], self.updateded_boundary[:-1]/self.K) * norm.pdf(self.d2(self.tau_nodes[:-1], self.updateded_boundary[:-1]/self.K)) / (self.updateded_boundary[:-1] * self.vol * self.vol * self.tau_nodes[:-1])

    ## Approximation of DPrime
    def DPrime(self):
        self.D_prime = -self.d2(self.tau_nodes[:-1], self.updateded_boundary[:-1]/self.K) * norm.pdf(self.d1(self.tau_nodes[:-1], self.updateded_boundary[:-1]/self.K)) / (self.updateded_boundary[:-1] * self.vol * self.vol * self.tau_nodes[:-1])

    def fprime(self):
        if self.iteration_no == 0:
            self.f_prime = self.K*np.exp(-(self.r-self.q)*self.tau_nodes[:-1])*(self.N_prime/self.D_values - self.D_prime*self.N_values/np.square(self.D_values) )
        else:
            self.f_prime = np.zeros(len(self.tau_nodes[:-1]))
    
    def update_boundary(self):
        """
        Update the boundary values using the Jacobi-Newton scheme.

        Parameters:
        - B_values (numpy array): Current boundary values B^{(j)}(\tau_i).

        Returns:
        - B_next (numpy array): Updated boundary values B^{(j+1)}(\tau_i).
        """
        B_next = np.zeros(self.n+1)
        
        for i in range(self.n):
            tau = self.tau_nodes[i]
            B_current = self.updateded_boundary[i]

            # Calculate f(tau, B) and f'(tau, B)
            f_value = self.f_values[i]
            f_derivative = self.f_prime[i]

            # Avoid division by zero
            denominator = f_derivative - 1.0
            if denominator == 0:
                denominator = 1e-20

            # Jacobi-Newton update formula
            numerator = B_current - f_value
            delta_B = self.eta * (numerator / denominator)

            # Update boundary value 
            B_current = B_current + delta_B

            # Ensure the boundary value is non-negative
            B_next[i] = max(B_current, 1e-10)
        B_next[self.n] = self.updateded_boundary[self.n]
        return B_next
    
    def run_full_algorithm(self):
        """
        Run the complete Jacobi-Newton iterative scheme for pricing American options.

        Parameters:
        - m (int): Number of iterations for the Jacobi-Newton scheme.

        Returns:
        - B_values (numpy array): The final boundary values after m iterations.
        """
        self.initialize_boundary()
        for j in range(1, self.m + 1):
            print(f"Starting iteration {j}/{self.m}")

            # Step 5: Compute H(sqrt(tau)) and initialize Chebyshev interpolation
            H_values = self.compute_H()
            self.initialize_chebyshev_interpolation(H_values)

            # Step 6: Evaluate boundary using Clenshaw algorithm at adjusted points
            self.evaluate_boundary()

            # Step 7: Compute N(tau_i, B) and D(tau_i, B), then compute f(tau_i, B)
            self.K1()
            self.K2()
            self.K3()
            self.compute_ND_values()
            self.Nprime()
            self.DPrime()
            self.compute_f_values()
            self.fprime()

            # Step 9: Update B_values using the Jacobi-Newton scheme
            self.updateded_boundary = self.update_boundary()

            print(f"Iteration {j}/{self.m} completed.")

        print("Jacobi-Newton iterations completed.")

    ## create one iteration for testin
    def run_once(self):
        self.initialize_boundary()
        H_values = self.compute_H()
        self.initialize_chebyshev_interpolation(H_values)

        # Step 6: Evaluate boundary using Clenshaw algorithm at adjusted points
        self.evaluate_boundary()

        # Step 7: Compute N(tau_i, B) and D(tau_i, B), then compute f(tau_i, B)
        self.K1()
        self.K2()
        self.K3()
        self.compute_ND_values()
        self.Nprime()
        self.DPrime()
        self.compute_f_values()
        self.fprime()

        # Step 9: Update B_values using the Jacobi-Newton scheme
        self.initial_boundary = self.updateded_boundary
        self.updateded_boundary = self.update_boundary()
        print("Jacobi-Newton iterations completed.")
        print(self.updateded_boundary)
        self.iteration_no = 1

    
    ## Manually run once more for testing
    def manual_update(self):
        H_values = self.compute_H()
        self.initialize_chebyshev_interpolation(H_values)

        # Step 6: Evaluate boundary using Clenshaw algorithm at adjusted points
        self.evaluate_boundary()

        # Step 7: Compute N(tau_i, B) and D(tau_i, B), then compute f(tau_i, B)
        self.K1()
        self.K2()
        self.K3()
        self.compute_ND_values()
        self.Nprime()
        self.DPrime()
        self.compute_f_values()
        self.fprime()

        # Step 9: Update B_values using the Jacobi-Newton scheme
        self.initial_boundary = self.updateded_boundary
        self.updateded_boundary = self.update_boundary()
        print("Jacobi-Newton iterations completed.")
        print(self.updateded_boundary)
        self.iteration_no += 1




        
