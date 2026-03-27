import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as st
from scipy.stats import norm
import sys

# ==========================================
# 1. SYSTEM PARAMETERS & TOGGLES
# ==========================================
ENABLE_REPRODUCIBILITY = False

if ENABLE_REPRODUCIBILITY:
    np.random.seed(42)

AREA_X = 1750          
AREA_Y = 5000          

H = 500.0              
S_MIN = 500.0          
R_MIN = 200.0          

K = 1000               
P_FA = 0.001           
THETA_DEG = 20.0       
N_t = 7                
BETA_0 = 10**(-20.0/10)  

NOISE_DBM = -80.0
P_N_DBM = 20.0
NOISE_VAR = 10**((NOISE_DBM - 30.0) / 10.0) 
P_N = 10**((P_N_DBM - 30.0) / 10.0)         

LAMBDA_PENALTY = 1e6 
Q_INV_PFA = norm.isf(P_FA)
SQRT_K_2 = np.sqrt(K / 2.0)

class Target:
    def __init__(self, x, y, freqs):
        self.x = x
        self.y = y
        self.z = 0.0
        self.pos = np.array([x, y])
        self.freq_set = freqs

def get_uav_bands(M):
    total_min = 100
    total_max = 400
    bands = []
    step = (total_max - total_min) / M
    for i in range(M):
        bands.append({'f_min': total_min + i*step, 'f_max': total_min + (i+1)*step})
    return bands

# ==========================================
# 2. PHYSICAL MODULAR EVALUATION 
# ==========================================
def compute_sinr_vectorized(uav_pos, uav_eta, target):
    M = uav_pos.shape[0]
    dx = target.x - uav_pos[:, 0]
    dy = target.y - uav_pos[:, 1]
    
    az_angle = np.arctan2(dy, dx)
    az_diff = np.abs((az_angle - uav_eta + np.pi) % (2 * np.pi) - np.pi)
    
    d_horizontal = np.sqrt(dx**2 + dy**2)
    el_angle = np.arctan2(H, d_horizontal)
    el_diff = np.abs(el_angle)
    
    alpha = np.radians(THETA_DEG / 2.0)
    
    solid_angle = 2 * np.pi * (1 - np.cos(alpha))
    G_modifier = (4 * np.pi / solid_angle) if solid_angle > 0 else 1.0
    G = np.zeros(M)
    
    valid_mask = (az_diff <= alpha) & (el_diff <= alpha)
    G[valid_mask] = np.exp(-(az_diff[valid_mask]**2 + el_diff[valid_mask]**2) / (2 * alpha**2)) * G_modifier
    
    d_3d_sq = dx**2 + dy**2 + H**2
    signal_power = (P_N * BETA_0 * N_t * G) / d_3d_sq
    return signal_power / NOISE_VAR

def compute_psum(uavs, targets, bands):
    P_sum = 0.0
    M = len(uavs)
    target_pds = []
    
    for target in targets:
        f_mask = np.zeros((M, len(target.freq_set)), dtype=bool)
        for m in range(M):
            f_mask[m, :] = (target.freq_set >= bands[m]['f_min']) & (target.freq_set <= bands[m]['f_max'])
            
        uav_has_overlap = np.any(f_mask, axis=1)
        avg_pds = np.zeros(M)
        
        if np.any(uav_has_overlap):
            base_sinrs = compute_sinr_vectorized(uavs[:, :2], uavs[:, 2], target)
            valid_sinrs = base_sinrs[uav_has_overlap]
            
            numerator = Q_INV_PFA - valid_sinrs * SQRT_K_2
            denominator = 1.0 + valid_sinrs
            pds = np.clip(norm.sf(numerator / denominator), 0.0, 1.0)
            
            avg_pds[uav_has_overlap] = pds
            
        complement_pds = np.ones((M, len(target.freq_set)))
        for m in range(M):
            complement_pds[m, f_mask[m, :]] = 1.0 - avg_pds[m]
            
        target_sum = np.sum(1.0 - np.prod(complement_pds, axis=0))
        target_pds.append(target_sum)
        P_sum += target_sum
        
    return P_sum, target_pds

def fitness_evaluation(solution, targets, bands):
    x, y = solution[:, 0], solution[:, 1]

    penalty = np.sum(np.abs(x[x < 0])) + np.sum(x[x > AREA_X] - AREA_X) + \
              np.sum(np.abs(y[y < 0])) + np.sum(y[y > AREA_Y] - AREA_Y)

    for i in range(len(solution)):
        for t in targets:
            dist = np.linalg.norm(solution[i, :2] - t.pos)
            if dist < S_MIN:
                penalty += (S_MIN - dist)

        for j in range(i + 1, len(solution)):
            dist = np.linalg.norm(solution[i, :2] - solution[j, :2])
            if dist < R_MIN:
                penalty += (R_MIN - dist)

    p_sum, t_pds = compute_psum(solution, targets, bands)

    return p_sum - LAMBDA_PENALTY * penalty, p_sum, t_pds

# ==========================================
# 3. OPTIMIZATION ROUTINES
# ==========================================
def baseline_placement(M, targets, bands):
    uavs = np.zeros((M, 3))
    for i in range(M):
        uavs[i, 0], uavs[i, 1] = AREA_X / 2.0, (AREA_Y / (M + 1)) * (i + 1)
        t_dists = [np.linalg.norm([uavs[i,0]-t.x, uavs[i,1]-t.y]) for t in targets]
        nearest_t = targets[np.argmin(t_dists)]
        uavs[i, 2] = np.arctan2(nearest_t.y - uavs[i,1], nearest_t.x - uavs[i,0])
        
    _, p_sum, t_pds = fitness_evaluation(uavs, targets, bands)
    return uavs, p_sum, t_pds

def run_pso(M, targets, bands, num_particles=50, max_iter=200):
    particles = np.random.uniform([0, 0, -np.pi], [AREA_X, AREA_Y, np.pi], (num_particles, M, 3))
    velocities = np.random.uniform(-10, 10, (num_particles, M, 3))
    pbest, pbest_fitness = np.copy(particles), np.full(num_particles, -np.inf)
    gbest = pbest[0]
    gbest_fitness = -np.inf
    history = []
    
    for t in range(1, max_iter + 1):
        w = 0.7 - (0.3) * (t / max_iter)
        for p in range(num_particles):
            velocities[p] = (w * velocities[p] + 
                             1.5 * np.random.rand(M, 3) * (pbest[p] - particles[p]) + 
                             2.0 * np.random.rand(M, 3) * (gbest - particles[p]))
            particles[p] += velocities[p]
            particles[p, :, 0] = np.clip(particles[p, :, 0], 0, AREA_X)
            particles[p, :, 1] = np.clip(particles[p, :, 1], 0, AREA_Y)
            particles[p, :, 2] = ((particles[p, :, 2] + np.pi) % (2 * np.pi)) - np.pi
            
            fit, _, _ = fitness_evaluation(particles[p], targets, bands)
            if fit > pbest_fitness[p]:
                pbest_fitness[p], pbest[p] = fit, np.copy(particles[p])
                if fit > gbest_fitness:
                    gbest_fitness, gbest = fit, np.copy(particles[p])
        history.append(gbest_fitness)
    return gbest, fitness_evaluation(gbest, targets, bands)[1], history

def run_ga(M, targets, bands, pop_size=50, max_iter=200):
    pop = np.random.uniform([0, 0, -np.pi], [AREA_X, AREA_Y, np.pi], (pop_size, M, 3))
    pop_fitness = np.array([fitness_evaluation(p, targets, bands)[0] for p in pop])
    gbest, gbest_fit = pop[np.argmax(pop_fitness)], np.max(pop_fitness)
    history = []
    
    for t in range(1, max_iter + 1):
        new_pop = np.zeros_like(pop)
        new_pop[0] = np.copy(pop[np.argmax(pop_fitness)]) 
        for p in range(1, pop_size):
            p1 = pop[max(*np.random.choice(pop_size, 2), key=lambda x: pop_fitness[x])]
            p2 = pop[max(*np.random.choice(pop_size, 2), key=lambda x: pop_fitness[x])]
            child = np.where(np.random.rand(M, 3) < 0.5, p1, p2)
            child += (np.random.rand(M, 3) < 0.1) * np.random.normal(0, [100.0, 100.0, 0.5], (M, 3))
            
            child[:, 0] = np.clip(child[:, 0], 0, AREA_X)
            child[:, 1] = np.clip(child[:, 1], 0, AREA_Y)
            child[:, 2] = ((child[:, 2] + np.pi) % (2 * np.pi)) - np.pi
            new_pop[p] = child
            
        pop = np.copy(new_pop)
        for p in range(pop_size):
            fit, _, _ = fitness_evaluation(pop[p], targets, bands)
            pop_fitness[p] = fit
            if fit > gbest_fit: gbest_fit, gbest = fit, np.copy(pop[p])
        history.append(gbest_fit)
    return gbest, fitness_evaluation(gbest, targets, bands)[1], history

# ==========================================
# 4. STATISTICAL CONFIGURATION & OUTPUTS
# ==========================================
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    class OutputLogger:
        def __init__(self): self.terminal, self.log = sys.stdout, open("output.txt", "w")
        def write(self, m): self.terminal.write(m); self.log.write(m)
        def flush(self): self.terminal.flush(); self.log.flush()
    sys.stdout = OutputLogger()
    
    base_freq_sets = [np.arange(105, 246, 5), np.arange(205, 346, 5), np.arange(305, 396, 5)]
    ORIG_TARGETS = [Target(3000, 2500, base_freq_sets[0]), Target(2200, 3500, base_freq_sets[1]), Target(4000, 3800, base_freq_sets[2])]
    M_values = [3, 4, 5]
    N_OUTER = 20  
    
    results = {'baseline': {m: [] for m in M_values}, 'pso': {m: [] for m in M_values}, 'ga': {m: [] for m in M_values}}
    histories_pso, histories_ga = {m: [] for m in M_values}, {m: [] for m in M_values}
    
    print(f"--- LAUNCHING EVALUATION (Reproducibility={ENABLE_REPRODUCIBILITY}) ---")
    for M in M_values:
        print(f"\nEvaluating M={M} Configurations...")
        bands = get_uav_bands(M)
        
        for mc in range(N_OUTER):
            # OUTER MC: Small target displacement allowing true variance validation in confidence intervals
            mc_targets = [Target(t.pos[0] + np.random.uniform(-50, 50), t.pos[1] + np.random.uniform(-50, 50), t.freq_set) for t in ORIG_TARGETS]
            print(f"  [DEBUG MC={mc+1}] Target T1 generated at ({mc_targets[0].x:.1f}, {mc_targets[0].y:.1f})")
            
            # Baseline evaluated on perturbed targets ensures FAIR deterministic comparison
            _, b_psum, _ = baseline_placement(M, mc_targets, bands)
            _, p_psum, p_hist = run_pso(M, mc_targets, bands)
            _, g_psum, g_hist = run_ga(M, mc_targets, bands)
            
            results['baseline'][M].append(b_psum)
            results['pso'][M].append(p_psum)
            results['ga'][M].append(g_psum)
            histories_pso[M].append(p_hist); histories_ga[M].append(g_hist)
            
            print(f"  > Run {mc+1}/{N_OUTER} Complete | Base: {b_psum:.2f} | GA: {g_psum:.2f} | PSO: {p_psum:.2f}")
            
        b_data, p_data, g_data = results['baseline'][M], results['pso'][M], results['ga'][M]
        
        ci_b = st.t.interval(0.95, df=len(b_data)-1, loc=np.mean(b_data), scale=st.sem(b_data) + 1e-9)
        ci_p = st.t.interval(0.95, df=len(p_data)-1, loc=np.mean(p_data), scale=st.sem(p_data) + 1e-9)
        ci_g = st.t.interval(0.95, df=len(g_data)-1, loc=np.mean(g_data), scale=st.sem(g_data) + 1e-9)
        
        print(f"\n--- M={M} Statistical Breakdown ---")
        print(f"Baseline: {np.mean(b_data):.2f} ± {np.std(b_data):.2f}  [95% CI: {ci_b[0]:.2f}, {ci_b[1]:.2f}]")
        print(f"GA Model: {np.mean(g_data):.2f} ± {np.std(g_data):.2f}  [95% CI: {ci_g[0]:.2f}, {ci_g[1]:.2f}] | Imprv: {((np.mean(g_data)-np.mean(b_data))/(np.mean(b_data) + 1e-9))*100:.1f}%")
        print(f"PSO Mod:  {np.mean(p_data):.2f} ± {np.std(p_data):.2f}  [95% CI: {ci_p[0]:.2f}, {ci_p[1]:.2f}] | Imprv: {((np.mean(p_data)-np.mean(b_data))/(np.mean(b_data) + 1e-9))*100:.1f}%\n")

    # === PLOTS ===
    plot_m = M_values[1] if len(M_values) > 1 else M_values[0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Use tick_labels instead of labels for Matplotlib 3.11 compatibility
    axes[0].boxplot([results['baseline'][plot_m], results['ga'][plot_m], results['pso'][plot_m]], tick_labels=['Baseline', 'GA', 'PSO'])
    axes[0].set_title(f'P_sum Variance Spread across MC Runs (M={plot_m})', fontweight='bold')
    axes[0].set_ylabel('P_sum')
    axes[0].grid(True)
    
    axes[1].hist(results['pso'][plot_m], bins=6, color='coral', alpha=0.6, label='PSO Distribution')
    axes[1].hist(results['ga'][plot_m], bins=6, color='lightblue', alpha=0.6, label='GA Distribution')
    axes[1].set_title(f'Output Density Profile (M={plot_m})', fontweight='bold')
    axes[1].set_xlabel('P_sum Score')
    axes[1].legend()
    axes[1].grid(True)
    
    iters = np.arange(len(histories_pso[plot_m][0]))
    axes[2].plot(iters, np.mean(histories_pso[plot_m], axis=0), color='coral', label='PSO Mean', linewidth=2)
    axes[2].plot(iters, np.mean(histories_ga[plot_m], axis=0), color='blue', label='GA Mean', linewidth=2)
    axes[2].set_title(f'Optimizer Convergence (M={plot_m})', fontweight='bold')
    axes[2].set_xlabel('Iteration Step')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('stochastic_statistical_analysis.png', dpi=300)
    plt.close()
    
    print("Simulation analysis complete. Plots generated seamlessly.")
