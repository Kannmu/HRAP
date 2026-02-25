import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
# 1. System Parameters
# ==========================================

# Constants
PI = np.pi

# Geometry & Mass
L = 8e-3          # Length of FPC island (m)
W = 4e-3          # Width of FPC island (m)
Area = L * W      # Area (m^2)
m = 40e-6         # Total mass (kg) (40 mg)

# Air Properties
mu_air = 1.81e-5  # Dynamic viscosity of air (Pa*s)

# Driving Force (Ultrasound Radiation Force)
# User calculated F_max_static = 0.11 mN.
# For AM modulation at 250Hz, the effective dynamic driving amplitude is approx half.
F_drive_amp = 0.055e-3 # (N)
f_drive = 250.0        # (Hz)
w_drive = 2 * PI * f_drive

# System Stiffness (Tuned to resonance)
# The user wants the resonance frequency of the island to be 250 Hz.
# k = m * w_n^2
k_fpc = m * (2 * PI * 250)**2

# Skin Properties
k_skin = 1000.0   # Stiffness of skin (N/m)
# The user mentioned a "very small air gap". Let's assume 100 microns (0.1 mm) as a baseline.
# This is a critical parameter.
h0_gap = 100e-6   # Initial Air Gap (m)

# Simulation Settings
t_start = 0.0
t_end = 0.1       # Simulate for 0.1 seconds (25 cycles)
fs = 100000       # Sampling frequency for output
t_eval = np.linspace(t_start, t_end, int((t_end-t_start)*fs))

# ==========================================
# 2. Physics Models
# ==========================================

def get_squeeze_film_damping(x, v):
    """
    Calculate squeeze film damping coefficient c(x).
    
    Model: Rectangular plate approximation.
    F_damping = c(x) * v
    c(x) approx (mu * L * W^3) / h(x)^3
    
    We limit the gap to a minimum value to avoid division by zero singularity
    and to represent surface roughness / contact.
    """
    current_gap = h0_gap - x
    
    # Minimum effective gap for fluid dynamics (roughness limit)
    # If gap is smaller than this, we assume the air film is fully compressed/displaced 
    # or contact mechanics dominate.
    min_gap = 1e-6 # 1 micron
    
    if current_gap < min_gap:
        effective_gap = min_gap
    else:
        effective_gap = current_gap
        
    # Rectangular plate formula approximation (assuming W < L)
    # A common approximation factor for rectangle is roughly proportional to width^3
    c_squeeze = (mu_air * L * (W**3)) / (effective_gap**3)
    
    return c_squeeze

def get_skin_force(x):
    """
    Calculate restoring force from skin.
    Piecewise linear model.
    """
    if x >= h0_gap:
        # Penetration into skin
        penetration = x - h0_gap
        return k_skin * penetration
    else:
        return 0.0

# ==========================================
# 3. Solver
# ==========================================

def system_dynamics(t, y):
    """
    State vector y = [position x, velocity v]
    dx/dt = v
    dv/dt = (F_drive - F_damping - F_spring - F_skin) / m
    """
    x = y[0]
    v = y[1]
    
    # 1. Driving Force
    F_d = F_drive_amp * np.sin(w_drive * t)
    
    # 2. Damping Force
    # Base structural damping (small) + Squeeze film damping (large nonlinear)
    # Assume a small Q for the FPC material itself, e.g., Q=50 -> zeta ~ 0.01
    c_struct = 2 * 0.01 * np.sqrt(k_fpc * m) 
    c_aero = get_squeeze_film_damping(x, v)
    c_total = c_struct + c_aero
    F_damp = c_total * v
    
    # 3. Elastic Forces
    F_spring = k_fpc * x
    F_contact = get_skin_force(x)
    
    # Equation of Motion
    # m * a = Sum(Forces)
    a = (F_d - F_damp - F_spring - F_contact) / m
    
    return [v, a]

# Initial Conditions
y0 = [0.0, 0.0] # Start at rest

print("Starting simulation...")
print(f"Mass: {m*1e6:.2f} mg")
print(f"Drive Force: {F_drive_amp*1e3:.3f} mN")
print(f"Gap: {h0_gap*1e6:.1f} um")
print(f"Target Frequency: {f_drive} Hz")

# Solve ODE
sol = solve_ivp(
    system_dynamics, 
    [t_start, t_end], 
    y0, 
    t_eval=t_eval, 
    method='RK45', 
    rtol=1e-6, 
    atol=1e-9
)

# ... (previous imports)

# ==========================================
# 4. Analysis Loop
# ==========================================

gaps_to_test = [1e-3, 300e-6, 200e-6, 100e-6, 50e-6, 20e-6, 10e-6, 5e-6, 1e-6]
results = []

print("-" * 60)
print(f"{'Gap (um)':<10} | {'Max Disp (um)':<15} | {'Penetration (um)':<18} | {'Status'}")
print("-" * 60)

plt.figure(figsize=(10, 6))

for h_val in gaps_to_test:
    # Update global or pass as arg? 
    # Let's redefine the wrapper to use h_val
    
    def system_dynamics_sweep(t, y):
        x = y[0]
        v = y[1]
        
        # 1. Driving Force
        F_d = F_drive_amp * np.sin(w_drive * t)
        
        # 2. Damping
        # Squeeze film with CURRENT gap h_val
        current_gap = h_val - x
        min_gap = 1e-7 # 0.1 um limit
        if current_gap < min_gap:
            effective_gap = min_gap
        else:
            effective_gap = current_gap
            
        c_squeeze = (mu_air * L * (W**3)) / (effective_gap**3)
        c_struct = 2 * 0.01 * np.sqrt(k_fpc * m)
        c_total = c_struct + c_squeeze
        
        # 3. Elastic
        F_spring = k_fpc * x
        if x >= h_val:
            F_contact = k_skin * (x - h_val)
        else:
            F_contact = 0.0
            
        a = (F_d - c_total * v - F_spring - F_contact) / m
        return [v, a]

    # Solve
    sol = solve_ivp(
        system_dynamics_sweep, 
        [t_start, t_end], 
        y0, 
        t_eval=t_eval, 
        method='LSODA', # LSODA is better for stiff systems (damping can be stiff)
        rtol=1e-6, 
        atol=1e-9
    )
    
    max_d = np.max(sol.y[0])
    penetration = max(0, max_d - h_val)
    
    status = "FAIL"
    if penetration >= 1e-6:
        status = "SUCCESS"
    elif penetration > 0:
        status = "TOUCH (Weak)"
        
    print(f"{h_val*1e6:<10.1f} | {max_d*1e6:<15.3f} | {penetration*1e6:<18.3f} | {status}")
    
    results.append((h_val, max_d, penetration))
    plt.plot(sol.t*1000, sol.y[0]*1e6, label=f'Gap={h_val*1e6:.0f} um')

print("-" * 60)

# Plotting
plt.title('Displacement vs Time for Different Initial Gaps')
plt.xlabel('Time (ms)')
plt.ylabel('Displacement (um)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('harp_gap_sweep.png')
print("Sweep plot saved to 'harp_gap_sweep.png'")

# Original single run analysis code removed/replaced by this loop

