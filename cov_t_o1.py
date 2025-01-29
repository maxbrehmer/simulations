import numpy as np
import matplotlib.pyplot as plt

def Tn_find(Y):
    """
    Computes T_n^2 = max_{1 <= r < n} [S - Spi_L - Spi_R],
    returns T_n = sqrt(T_n^2).
    """
    n = len(Y)
    if n < 2:
        raise ValueError("Array must have at least 2 elements.")
    # Full sum of squares
    S = np.sum((Y - np.mean(Y))**2)

    best_val = float('-inf')
    for r in range(1, n):
        Y_left  = Y[:r]
        Y_right = Y[r:]
        Spi_L = np.sum((Y_left - np.mean(Y_left))**2)
        Spi_R = np.sum((Y_right - np.mean(Y_right))**2)
        val = S - Spi_L - Spi_R  # T_n^2 if it's the max
        if val > best_val:
            best_val = val
    return np.sqrt(best_val)


def single_simulation(n, mu_X=0, sigma_X=1, rho_X=0.5,
                      mu_Y=0, sigma_Y=1, rho_Y=0,
                      seed=None):
    """
    1) Generates X_1, X_2 ~ MVN(rho_X), 
       Y_1, Y_2 ~ MVN(rho_Y).
    2) Sorts Y_1, Y_2 by X_1, X_2.
    3) Computes T_n(Y_1), T_n(Y_2) and returns (Tn_1, Tn_2).
    """
    if seed is not None:
        np.random.seed(seed)

    cov_X = [[sigma_X**2, rho_X*sigma_X*sigma_X],
             [rho_X*sigma_X*sigma_X, sigma_X**2]]
    X = np.random.multivariate_normal([mu_X, mu_X], cov_X, n)
    X_1, X_2 = X[:, 0], X[:, 1]

    cov_Y = [[sigma_Y**2, rho_Y*sigma_Y*sigma_Y],
             [rho_Y*sigma_Y*sigma_Y, sigma_Y**2]]
    Y = np.random.multivariate_normal([mu_Y, mu_Y], cov_Y, n)
    Y_1, Y_2 = Y[:, 0], Y[:, 1]

    # Sort indices
    pi_1 = np.argsort(X_1)
    pi_2 = np.argsort(X_2)
    # Ordered data
    Ypi_1 = Y_1[pi_1]
    Ypi_2 = Y_2[pi_2]

    Tn_1 = Tn_find(Ypi_1)
    Tn_2 = Tn_find(Ypi_2)

    return Tn_1, Tn_2


def simulate_covariance_across_n(n_values, m=500, seed=42):
    """
    For each n in n_values:
      - Run m simulations,
      - Compute Tn^* for Y_1 and Y_2,
      - Estimate Cov(Tn^*_1, Tn^*_2).
    Returns a dict with the results, keyed by n.

    We define Tn^* = (Tn - b_n)/a_n, with user-chosen a_n, b_n.
    """
    # For reproducibility: fix master seed here
    np.random.seed(seed)

    results = {}
    for n in n_values:
        # Compute a_n, b_n for this n
        a_n = (2 * np.log(np.log(n)))**(-0.5)
        b_n = (1 / a_n
               + 0.5 * a_n * np.log(np.log(np.log(n)))
               + a_n * np.log(2 * np.pi**(-0.5)))

        # Collect Tn^*(Y_1) and Tn^*(Y_2)
        Tn_1_vals = []
        Tn_2_vals = []

        for _ in range(m):
            # We can pass None for a random seed each iteration.
            # The main np.random.seed(...) call outside controls reproducibility.
            Tn_1, Tn_2 = single_simulation(n=n, rho_X=rho_X, rho_Y=rho_Y, seed=None)

            # Standardize
            Tn_1_tilde = (Tn_1 - b_n)/a_n
            Tn_2_tilde = (Tn_2 - b_n)/a_n
            Tn_1_vals.append(Tn_1_tilde)
            Tn_2_vals.append(Tn_2_tilde)

        Tn_1_vals = np.array(Tn_1_vals)
        Tn_2_vals = np.array(Tn_2_vals)

        # Covariance (single scalar). ddof=1 => sample covariance
        cov_matrix = np.cov(Tn_1_vals, Tn_2_vals, ddof=1)
        cov_scalar = cov_matrix[0, 1]

        # Store in dict
        mean_T1 = np.mean(Tn_1_vals)
        mean_T2 = np.mean(Tn_2_vals)
        results[n] = {
            "cov": cov_scalar,
            "mean_Tn1": mean_T1,
            "mean_Tn2": mean_T2,
            "std_Tn1": np.std(Tn_1_vals, ddof=1),
            "std_Tn2": np.std(Tn_2_vals, ddof=1),
        }

    return results


if __name__ == "__main__":
    # Example n values
    n_values = [3, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250, 350, 500]
    m = 500   # simulations per n
    rho_X = 0.1
    rho_Y = 0.1
    master_seed = 42

    # Run the simulations
    results = simulate_covariance_across_n(n_values, m=m, seed=master_seed)

    # Print table-like output
    print("n\tCov(Tn1*,Tn2*)\tMean(Tn1*)\tMean(Tn2*)\tStd(Tn1*)\tStd(Tn2*)")
    for n in n_values:
        row = results[n]
        print(f"{n}\t{row['cov']:.4f}\t\t{row['mean_Tn1']:.4f}\t\t{row['mean_Tn2']:.4f}"
              f"\t\t{row['std_Tn1']:.4f}\t\t{row['std_Tn2']:.4f}")

    # For plotting: extract covariances in order
    covariances = [results[n]["cov"] for n in n_values]

    # Plot covariance vs n
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, covariances, marker='o', linestyle='-')
    plt.title(f"Estimated Covariance of (Tn^*_1, Tn^*_2) vs n (m={m}, rho_X={rho_X}, rho_Y={rho_Y})")
    plt.xlabel("n")
    plt.ylabel("Covariance")
    plt.grid(True)
    plt.show()
