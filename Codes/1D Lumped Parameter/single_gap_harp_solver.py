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
h0_gap = 200e-6   # Initial Air Gap (m)

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

# ==========================================
# 4. Analysis
# ==========================================

x_response = sol.y[0]
v_response = sol.y[1]
time = sol.t

# Calculate forces for analysis
damping_forces = []
skin_forces = []
for i, t_i in enumerate(time):
    xi = x_response[i]
    vi = v_response[i]
    damping_forces.append( (get_squeeze_film_damping(xi, vi) + 2*0.01*np.sqrt(k_fpc*m)) * vi )
    skin_forces.append(get_skin_force(xi))

damping_forces = np.array(damping_forces)
skin_forces = np.array(skin_forces)

# Metrics
max_disp = np.max(x_response)
min_gap_reached = h0_gap - max_disp
max_skin_penetration = max(0, max_disp - h0_gap)
max_velocity = np.max(np.abs(v_response))

print("-" * 30)
print("RESULTS")
print("-" * 30)
print(f"Max Displacement: {max_disp*1e6:.2f} um")
print(f"Initial Gap:      {h0_gap*1e6:.2f} um")
print(f"Min Gap Reached:  {min_gap_reached*1e6:.2f} um")
print(f"Skin Penetration: {max_skin_penetration*1e6:.2f} um")
print(f"Max Velocity:     {max_velocity:.4f} m/s")

# Check feasibility
if max_skin_penetration >= 1.0:
    print("\n[SUCCESS] The system achieves > 1 um skin penetration.")
else:
    print("\n[FAILURE] The system FAILS to penetrate the skin by 1 um.")
    print("Reason: Squeeze film damping likely consumed the energy.")

# Plotting
plt.figure(figsize=(12, 10))

# Plot 1: Displacement vs Time
plt.subplot(3, 1, 1)
plt.plot(time*1000, x_response*1e6, label='Displacement')
plt.axhline(y=h0_gap*1e6, color='r', linestyle='--', label='Skin Surface')
plt.title('FPC Island Displacement vs Time')
plt.ylabel('Displacement (um)')
plt.xlabel('Time (ms)')
plt.legend()
plt.grid(True)

# Plot 2: Phase Portrait (Velocity vs Position)
plt.subplot(3, 1, 2)
plt.plot(x_response*1e6, v_response, label='Trajectory')
plt.axvline(x=h0_gap*1e6, color='r', linestyle='--', label='Skin Surface')
plt.title('Phase Portrait')
plt.xlabel('Displacement (um)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

# Plot 3: Forces
plt.subplot(3, 1, 3)
plt.plot(time*1000, damping_forces*1e3, label='Damping Force')
plt.plot(time*1000, skin_forces*1e3, label='Skin Contact Force', alpha=0.7)
plt.plot(time*1000, F_drive_amp * np.sin(w_drive * time) * 1e3, label='Drive Force', linestyle='--', alpha=0.5)
plt.title('Forces Analysis')
plt.ylabel('Force (mN)')
plt.xlabel('Time (ms)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('Codes/1D Lumped Parameter/harp_simulation_results.png')
print("\nPlot saved to 'Codes/1D Lumped Parameter/harp_simulation_results.png'")
