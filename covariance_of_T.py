import numpy as np

seed = 42
np.random.seed(seed)

n = 3   # Number of observations
d = 2   # Number of features
mu = 0  # Mean of the distribution
sigma = 1  # Standard deviation of the distribution

print('\n <<< THIS IS A SIMULATION!!! >>> \n')

# Compute standardization factors
a_n = (2 * np.log(np.log(n)))**(-0.5)
b_n = 1/a_n + 1/2*a_n*np.log(np.log(np.log(n))) + a_n*np.log(2*np.pi**(-0.5))

print(f'a_n = {a_n}')
print(f'b_n = {b_n} \n')

# Generate data
X = np.random.normal(mu, sigma, (n, d))
print(f'X = {X}')

for j in range(d):
    X_j = X[:, j]

# Perform partition for a range of r values between 1 and n-1
partitions = {}
for r in range(1, n):
    # Partition the data by the change point
    X_r1 = X[:r]
    X_r2 = X[r:]
    # Save each partition with its own index
    partitions[f'X1_r{r}'] = X_r1
    partitions[f'X2_r{r}'] = X_r2
    print(f'\n For r = {r}:')
    print(f'X_1 = {X_r1}')
    print(f'X_2 = {X_r2}')

# Show elements in the partitions dict
#for key, value in partitions.items():
#    print(f'\n {key}: {value}')

# Set up test statistic for each r in range 1 to n-1
sums = {}
for r in range(1, n):
    # Sum of X_1:r for each dimension j
    S_r = np.sum(X[:r], axis=0)
    # Sum of X_1:n for each dimension j
    S_n = np.sum(X, axis=0)

    # Save each sum with its own index
    sums[f'Sr_r{r}'] = S_r
    sums[f'Sn_r{r}'] = S_n
    print(f'\n For r = {r}:')
    print(f'S_r = {S_r}')
    print(f'S_n = {S_n}')

for r in range(1, n):
    # Array of sums for each dimension j
    S_r = []
    S_n = []
    for j in range(d):
        S_r.append(sums[f'Sr_r{r}'][j])
        S_n.append(sums[f'Sn_r{r}'][j])

    # Initialize T_n to a very small value
    T_n = -np.inf

    # Convert S_r and S_n to numpy arrays
    S_r = np.array(S_r)
    S_n = np.array(S_n)

    print(f'\n (r={r}) T_n before scaling: {np.abs(
            S_r/np.sqrt(n) - r/n * S_n/np.sqrt(n)
        ) / (r/n * (1-r/n))**0.5}')

    # Compute the test statistic for each r and keep the maximum
    T_r = np.max(
        np.abs(
            S_r/np.sqrt(n) - r/n * S_n/np.sqrt(n)
        ) / (r/n * (1-r/n))**0.5
    )
    if T_r > T_n:
        T_n = T_r

    # Adjust T_n with b_n and a_n
    T_n = (T_n - b_n) / a_n

    print(f'\n For r = {r}:')
    print(f'S_r = {S_r}')
    print(f'max S_r = {np.max(S_r)}')
    print(f'S_n = {S_n}')
    print(f'max S_n = {np.max(S_n)}')
    print(f'T_n = {T_n}')
