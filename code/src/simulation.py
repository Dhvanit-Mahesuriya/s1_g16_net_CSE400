import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parameters
# -----------------------------
K = 1000          # number of samples
L = 10            # number of sensors
SNR_dB = 0
SNR = 10**(SNR_dB / 10)
N = 10000         # Monte Carlo runs

# -----------------------------
# Store test statistic values
# -----------------------------
T_H0 = np.zeros(N)
T_H1 = np.zeros(N)

# -----------------------------
# Monte Carlo Simulation
# -----------------------------
for i in range(N):

    # -------- H0: Noise Only --------
    W0 = np.random.normal(0, 1, (L, K))
    Ry0 = (1 / K) * (W0 @ W0.T)
    eigvals0 = np.linalg.eigvalsh(Ry0)
    T_H0[i] = np.max(eigvals0) / np.min(eigvals0)

    # -------- H1: Signal + Noise --------
    X = np.random.normal(0, np.sqrt(SNR), (L, K))
    W1 = np.random.normal(0, 1, (L, K))
    Y1 = X + W1
    Ry1 = (1 / K) * (Y1 @ Y1.T)
    eigvals1 = np.linalg.eigvalsh(Ry1)
    T_H1[i] = np.max(eigvals1) / np.min(eigvals1)

# -----------------------------
# PDF Plot
# -----------------------------
plt.figure()
plt.hist(T_H0, bins=80, density=True, alpha=0.6, label="H0")
plt.hist(T_H1, bins=80, density=True, alpha=0.6, label="H1")
plt.title("Empirical PDF of T")
plt.xlabel("T = lambda_max / lambda_min")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# CDF Plot
# -----------------------------
plt.figure()

T0_sorted = np.sort(T_H0)
T1_sorted = np.sort(T_H1)

cdf0 = np.arange(1, N + 1) / N
cdf1 = np.arange(1, N + 1) / N

plt.plot(T0_sorted, cdf0, label="CDF under H0")
plt.plot(T1_sorted, cdf1, label="CDF under H1")
plt.title("Empirical CDF of T")
plt.xlabel("T = lambda_max / lambda_min")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.grid(True)
plt.show()
