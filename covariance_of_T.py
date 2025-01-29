import numpy as np

seed = 42
np.random.seed(seed)

n = 3   # Number of observations
d = 2   # Number of features
mu_Y = 0  # Mean of Y
sigma_Y = 1  # Standard deviation of Y
mu_X = 0  # Mean of X
sigma_X = 1  # Standard deviation of X
rho_X = 0 # Correlation between X_1 and X_2
rho_Y = 0 # Correlation between Y_1 and Y_2

# Precompute a_n, b_n based on n
a_n = (2 * np.log(np.log(n)))**(-0.5)
b_n = (1 / a_n
       + 0.5 * a_n * np.log(np.log(np.log(n)))
       + a_n * np.log(2 * np.pi**(-0.5)))

# Generate time ordered X_1 and X_2 as N(mu_X, sigma_X^2) with correlation rho_X
X = np.random.multivariate_normal([mu_X, mu_X], [[sigma_X**2, rho_X], [rho_X, sigma_X**2]], n)
X_1 = X[:, 0]
X_2 = X[:, 1]

# Generate response variables Y_1 and Y_2 as N(mu_Y, sigma_Y^2) without correlation
Y = np.random.multivariate_normal([mu_Y, mu_Y], [[sigma_Y**2, rho_Y], [rho_Y, sigma_Y**2]], n)
Y_1 = Y[:, 0]
Y_2 = Y[:, 1]

# Find ordering of X_1 and X_2 by setting pi_k as the index list of the size ordered elements of X_m
pi_1 = np.argsort(X_1)
pi_2 = np.argsort(X_2)
Xpi_1 = X_1[pi_1]
Xpi_2 = X_2[pi_2]

# Order Y_1 and Y_2 according to the pi ordering of X_1 and X_2
Ypi_1 = Y_1[pi_1]
Ypi_2 = Y_2[pi_2]

# Print results of data generation and ordering
print(f"\n X (unordered):\n {X}")
print(f"\n Y (unordered):\n {Y}")
print(f"\n X_1 (unordered): {X_1}")
print(f"\n X_2 (unordered): {X_2}")
print(f"\n pi_1: {pi_1}")
print(f"\n pi_2: {pi_2}")
print(f"\n X_1 (ordered): {Xpi_1}")
print(f"\n X_2 (ordered): {Xpi_2}")
print(f"\n Y_1 (ordered): {Ypi_1}")
print(f"\n Y_2 (ordered): {Ypi_2}")

# --------------------------------------------------------
# 1) Full sum of squares for each array
# --------------------------------------------------------
# sum of (y - mean(y))^2 for the *entire* array
S1 = np.sum((Ypi_1 - np.mean(Ypi_1))**2)
S2 = np.sum((Ypi_2 - np.mean(Ypi_2))**2)

print(f"\nFull sum of squares (Ypi_1): S1 = {S1:.3f}")
print(f"Full sum of squares (Ypi_2): S2 = {S2:.3f}")
print("")

# --------------------------------------------------------
# 2) Loop over a single change point r (1..n-1)
#    splitting BOTH Ypi_1 and Ypi_2 at the same r
# --------------------------------------------------------
#   Left block = Y[:r]
#   Right block = Y[r:]
#
#   Sums of squares about each block's own mean:
#   Spi_1L = sum((Ypi_1[:r] - mean(Ypi_1[:r]))^2)
#   Spi_1R = sum((Ypi_1[r:] - mean(Ypi_1[r:]))^2)
#   (and similarly for Ypi_2)
# --------------------------------------------------------
for r in range(1, n):
    # --- Ypi_1 ---
    Y1_left = Ypi_1[:r]
    Y1_right = Ypi_1[r:]

    Spi_1L = np.sum((Y1_left - np.mean(Y1_left))**2)
    Spi_1R = np.sum((Y1_right - np.mean(Y1_right))**2)

    # --- Ypi_2 ---
    Y2_left = Ypi_2[:r]
    Y2_right = Ypi_2[r:]

    Spi_2L = np.sum((Y2_left - np.mean(Y2_left))**2)
    Spi_2R = np.sum((Y2_right - np.mean(Y2_right))**2)

    # Print or store the results
    print(
        f"r={r:2d} | "
        f"Spi_1L={Spi_1L:.3f}, Spi_1R={Spi_1R:.3f}, "
        f"Spi_2L={Spi_2L:.3f}, Spi_2R={Spi_2R:.3f}"
    )

# --------------------------------------------------------
# 3) Find T_n and best r
# --------------------------------------------------------
def Tn_find(Y):
    """
    Given a 1D NumPy array Y,
    computes T_n^2 = max_{1 <= r < n} [ S - Spi_L - Spi_R ],
    where:
      S      = sum((Y - mean(Y))^2)         (entire array's sum of squares),
      Spi_L  = sum((Y[:r]   - mean(Y[:r]))^2),
      Spi_R  = sum((Y[r:]   - mean(Y[r:]))^2).
    Returns the scalar T_n^2 and the r that achieves it.
    """
    n = len(Y)
    if n < 2:
        raise ValueError("Array must have at least 2 elements.")

    # 1) Full sum of squares over entire array
    S = np.sum((Y - np.mean(Y))**2)

    best_val = float('-inf')
    best_r = None

    # 2) Try all splits 1..(n-1)
    for r in range(1, n):
        Y_left  = Y[:r]
        Y_right = Y[r:]

        # (Y[:r] - mean(Y[:r]))^2
        Spi_L = np.sum((Y_left - np.mean(Y_left))**2)
        # (Y[r:] - mean(Y[r:]))^2
        Spi_R = np.sum((Y_right - np.mean(Y_right))**2)

        val = S - Spi_L - Spi_R  # quantity of interest

        # Keep track of the maximum
        if val > best_val:
            best_val = val
            best_r = r
    
    Tn = np.sqrt(best_val)

    # best_val is T_n^2, best_r is the split that achieves it
    return Tn, best_r

# Compute T_n for Ypi_1
Tn_1, split_1 = Tn_find(Ypi_1)
print(f"\nFor Ypi_1: T_n^2 = {Tn_1:.4f}, best split r={split_1}")

# Compute T_n for Ypi_2
Tn_2, split_2 = Tn_find(Ypi_2)
print(f"For Ypi_2: T_n^2 = {Tn_2:.4f}, best split r={split_2}")

# Standardize T_n by a_n and b_n
Tn_1_tilde = (Tn_1 - b_n)/a_n
Tn_2_tilde = (Tn_2 - b_n)/a_n

print(f"\nStandardized T_n for Ypi_1: T_n_tilde = {Tn_1_tilde:.4f}")
print(f"Standardized T_n for Ypi_2: T_n_tilde = {Tn_2_tilde:.4f}")

# --------------------------------------------------------
# 3) Compute covariance of T_n for Ypi_1 and Ypi_2
# --------------------------------------------------------

cov_Tn = np.cov([Tn_1_tilde, Tn_2_tilde], bias=True)
print(f"\nCovariance of T_n for Ypi_1 and Ypi_2:\n{cov_Tn}")