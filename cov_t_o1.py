import numpy as np

def compute_Tn_1D(data_1d, a_n, b_n):
    """
    Compute the test statistic T_n for 1D data (n observations).
    The formula matches your approach:
      1) For r in {1,...,n-1}:
         T_r = |S_r/sqrt(n) - (r/n)*S_n/sqrt(n)| / sqrt((r/n)*(1-r/n))
      2) T_n(raw) = max_r T_r
      3) T_n = (T_n(raw) - b_n) / a_n
    """
    n = len(data_1d)
    S_n = np.sum(data_1d)  # total sum

    # We will scan over r=1,...,n-1 and pick the maximum
    T_n_raw = -np.inf

    for r in range(1, n):
        S_r = np.sum(data_1d[:r])  # partial sum up to r
        numerator = abs((S_r / np.sqrt(n)) - (r / n) * (S_n / np.sqrt(n)))
        denominator = np.sqrt((r / n) * (1 - r / n))

        # Avoid division by zero if n=3 is extremely small, but here it won't be zero.
        if denominator == 0:
            continue

        T_r = numerator / denominator
        if T_r > T_n_raw:
            T_n_raw = T_r

    # Apply shift and scale
    return (T_n_raw - b_n) / a_n

def main_simulation(num_sims=1000, n=3, mu=0, sigma=1):
    np.random.seed(42)  # For reproducibility
    
    # 2D data (since you have T_n1 for dimension 1, T_n2 for dimension 2)
    d = 2

    # Precompute a_n, b_n based on n
    a_n = (2 * np.log(np.log(n)))**(-0.5)
    b_n = (1 / a_n
           + 0.5 * a_n * np.log(np.log(np.log(n)))
           + a_n * np.log(2 * np.pi**(-0.5)))
    
    print(f"n = {n}")
    print(f"a_n = {a_n:.4f}")
    print(f"b_n = {b_n:.4f}\n")

    Tn1_list = []
    Tn2_list = []

    # Repeat the experiment many times
    for _ in range(num_sims+1):
        # Generate n x d data from N(mu, sigma^2)
        X = np.random.normal(mu, sigma, (n, d))

        # Dimension 1 data
        X_dim1 = X[:, 0]
        # Dimension 2 data
        X_dim2 = X[:, 1]

        # Compute T_n1 from dimension 1
        T_n1 = compute_Tn_1D(X_dim1, a_n, b_n)

        # Compute T_n2 from dimension 2
        T_n2 = compute_Tn_1D(X_dim2, a_n, b_n)

        Tn1_list.append(T_n1)
        Tn2_list.append(T_n2)

    # Convert to arrays
    Tn1_array = np.array(Tn1_list)
    Tn2_array = np.array(Tn2_list)

    # Estimate covariance matrix
    cov_matrix = np.cov(Tn1_array, Tn2_array)
    print("Covariance matrix (2x2):")
    print(cov_matrix)

    # Off-diagonal is the covariance
    cov_Tn1_Tn2 = cov_matrix[0, 1]
    print(f"\nSample Cov(T_n1, T_n2) over {num_sims} runs: {cov_Tn1_Tn2:.4f}")

if __name__ == "__main__":
    main_simulation(num_sims=1000, n=3, mu=0, sigma=1)
