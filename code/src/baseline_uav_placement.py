import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm

# Configuration
AREA_SIZE = 5000
DEPLOYABLE_X_MAX = 1800
MIN_UAV_SEPARATION = 500
MIN_TARGET_SEPARATION = 400

# Fixed Targets (spread out to test collision avoidance)
targets = np.array([(2600, 3500), (4000, 3800), (3000, 2000)])

class EBDModel:
    def __init__(self):
        self.H = 100.0             # UAV Altitude in meters
        self.P_n = 0.1             # Transmit power of targets in Watts (20 dBm)
        self.beta0 = 1e-4          # Path loss at reference distance of 1m
        self.Nt = 1                # Number of transmit antennas 
        self.noise_var = 1e-11     # Noise variance (linear scale) -> ~ -80 dBm
        self.K = 200               # Number of samples (processing gain)
        self.P_fa_target = 0.05    # Target local false alarm probability
        self.beamwidth_deg = 60.0  # Default 3dB beamwidth in degrees

    def compute_distance(self, horizontal_dist):
        return np.sqrt(horizontal_dist**2 + self.H**2)

    def compute_gain(self, horizontal_dist):
        bw_rad = np.radians(self.beamwidth_deg)
        solid_angle = 2 * np.pi * (1 - np.cos(bw_rad / 2))
        G0 = 4 * np.pi / solid_angle if solid_angle > 0 else 1.0
        theta_rad = np.arctan2(horizontal_dist, self.H)
        return G0 if theta_rad <= (bw_rad / 2) else 1e-3

    def compute_sinr(self, d_3d, G_mn):
        signal_power = (self.P_n * self.beta0 * self.Nt * G_mn) / (d_3d**2)
        return signal_power / self.noise_var

    def compute_threshold(self):
        q_inv_pfa = norm.isf(self.P_fa_target)
        return 1.0 + np.sqrt(2.0 / self.K) * q_inv_pfa

    def compute_local_pd(self, sinr_val, lambda_ebd):
        mean_h1 = 1.0 + sinr_val
        std_h1 = np.sqrt(2.0 * (1.0 + sinr_val)**2 / self.K)
        z = (lambda_ebd - mean_h1) / std_h1
        return norm.sf(z)

    def compute_or_fusion(self, pd_matrix):
        pd_matrix = np.clip(pd_matrix, 0.0, 1.0)
        return 1.0 - np.prod(1.0 - pd_matrix, axis=0)

def baseline_placement(M):
    """
    Deterministic unoptimized placement:
    Distribute UAVs evenly along the Y-axis within the deployable X-region.
    """
    uavs = np.zeros((M, 2))
    # Center them in the X direction (between 0 and DEPLOYABLE_X_MAX)
    fixed_x = DEPLOYABLE_X_MAX / 2
    
    # Space them evenly in the Y direction (between 0 and AREA_SIZE)
    # E.g. if M=3, space them at 1/4, 2/4, 3/4 of AREA_SIZE
    y_spacing = AREA_SIZE / (M + 1)
    
    for i in range(M):
        uavs[i, 0] = fixed_x
        uavs[i, 1] = y_spacing * (i + 1)
        
    return uavs

# --- Main Simulation ---
print("Running Baseline Placement for M=3...")
best_uavs_3 = baseline_placement(3)
print("Running Baseline Placement for M=4...")
best_uavs_4 = baseline_placement(4)
print("Running Baseline Placement for M=5...")
best_uavs_5 = baseline_placement(5)

scenarios = [
    {"label": "(a)", "title": "M = 3 UAVs (Baseline)", "uavs": best_uavs_3, "M": 3},
    {"label": "(b)", "title": "M = 4 UAVs (Baseline)", "uavs": best_uavs_4, "M": 4},
    {"label": "(c)", "title": "M = 5 UAVs (Baseline)", "uavs": best_uavs_5, "M": 5},
]

# --- Monte Carlo Sum Detection Probability Calculation ---
np.random.seed(42)
N_RANDOM_TARGETS = 20
N_MONTE_CARLO_TRIALS = 100

ebd = EBDModel()
lambda_ebd = ebd.compute_threshold()

print(f"\nEvaluating Average Sum Detection Probability over {N_MONTE_CARLO_TRIALS} Monte Carlo Trials ({N_RANDOM_TARGETS} targets each):")
for scenario in scenarios:
    uavs = scenario['uavs']
    monte_carlo_psum = []
    
    for trial in range(N_MONTE_CARLO_TRIALS):
        random_targets = np.zeros((N_RANDOM_TARGETS, 2))
        random_targets[:, 0] = np.random.uniform(0, AREA_SIZE, N_RANDOM_TARGETS)
        random_targets[:, 1] = np.random.uniform(0, AREA_SIZE, N_RANDOM_TARGETS)
        
        pd_matrix = np.zeros((len(uavs), N_RANDOM_TARGETS))
        
        for i, uav in enumerate(uavs):
            for j, tgt in enumerate(random_targets):
                r = np.linalg.norm(uav - tgt)
                d_3d = ebd.compute_distance(r)
                G = ebd.compute_gain(r)
                sinr = ebd.compute_sinr(d_3d, G)
                pd_matrix[i, j] = ebd.compute_local_pd(sinr, lambda_ebd)
                
        p_fused = ebd.compute_or_fusion(pd_matrix)
        p_sum = np.sum(p_fused)
        monte_carlo_psum.append(p_sum)
        
    avg_p_sum = np.mean(monte_carlo_psum)
    print(f"M = {scenario['M']} UAVs: Average P_sum = {avg_p_sum:.4f} / {N_RANDOM_TARGETS}")
print()

# Formatting parameters for scientific style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'font.family': 'serif'
})

# Set up figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, scenario in zip(axes, scenarios):
    ax.set_xlim(0, AREA_SIZE)
    ax.set_ylim(0, AREA_SIZE)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{scenario['label']} {scenario['title']}")
    ax.set_aspect('equal')
    ax.grid(True, linestyle=(0, (5, 5)), alpha=0.5)
    
    # Draw deployable region
    deploy_patch = patches.Rectangle((0, 0), DEPLOYABLE_X_MAX, AREA_SIZE, 
                                     linewidth=0, facecolor='lightgreen', alpha=0.3, label='Deployable Area')
    ax.add_patch(deploy_patch)
    
    # Plot Targets and their Minimum Distance Circles
    for idx, t in enumerate(targets):
        tx, ty = t[0], t[1]
        ax.plot(tx, ty, 'rX', markersize=8, label='Target' if idx == 0 else "")
        
        # Specific labels to match the image style roughly
        labels_list = ["{205, 210, ...,\n340, 345}", "{305, 310, ...,\n395, 400}", "{105, 110, ...,\n240, 245}"]
        ax.annotate(labels_list[idx % len(labels_list)], (tx, ty), textcoords="offset points", xytext=(5, 15), fontsize=8, color='black')
        
        # Minimum Target Distance Circle
        min_dist_circle = patches.Circle((tx, ty), MIN_TARGET_SEPARATION, linewidth=1.5, edgecolor='orange', 
                                         facecolor='none', linestyle='-.', label='Minimum UAV-Target Distance' if idx == 0 else "")
        ax.add_patch(min_dist_circle)
        
    # Plot UAVs and orient them exactly to their nearest target
    for idx, u in enumerate(scenario['uavs']):
        ux, uy = u[0], u[1]
        
        # Find nearest target for orientation
        target_dists = [np.linalg.norm(u - t) for t in targets]
        nearest_t = targets[np.argmin(target_dists)]
        dx = nearest_t[0] - ux
        dy = nearest_t[1] - uy
        norm = np.hypot(dx, dy)
        if norm == 0: norm = 1
        
        # Antenna orientation dashed line extending towards/past target
        ax.plot([ux, ux + dx * 1.5], [uy, uy + dy * 1.5], linestyle='--', color='grey', linewidth=1, 
                label='Antenna orientation' if idx == 0 else "", zorder=1)
        
        # UAV marker with label e.g., [200, 300]
        ax.plot(ux, uy, 'k*', markersize=10, label='UAV' if idx == 0 else "", zorder=3)
        ax.annotate(f"[{int(ux)}, {int(uy)}]", (ux, uy), textcoords="offset points", xytext=(-45, 5), fontsize=9, fontweight='bold')

# Adding unified legend
handles, labels = axes[0].get_legend_handles_labels()

# To preserve specific order
order_of_items = ['Deployable Area', 'UAV', 'Target', 'Antenna orientation', 'Minimum UAV-Target Distance']
by_label = dict(zip(labels, handles))
ordered_handles = [by_label[k] for k in order_of_items if k in by_label]
ordered_labels = [k for k in order_of_items if k in by_label]

fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), frameon=True, edgecolor='black')

plt.tight_layout()
fig.subplots_adjust(bottom=0.2)
plt.savefig("baseline_uav_deployment_simulation.png", dpi=300, bbox_inches='tight')
print("Saved baseline_uav_deployment_simulation.png")
plt.close()
