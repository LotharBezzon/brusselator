import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Parameters ---
N = 80
frames = 200        # Number of video frames
steps_per_frame = 20 # Math steps per video frame
dt = 0.01
dx = 1.0
Omega = 5000.0        # System Volume (Noise level)

# Model Coefficients
a = 1.0
b = 5.0
Dx = 0.0
Dy = 0.0

# --- 2. Initialization ---
x_ss = 1.0
y_ss = b / a

# Initialize grid, add gaussian noise around the steady state
x = np.ones((N, N)) * x_ss + np.random.normal(0, 0.4, (N, N))
y = np.ones((N, N)) * y_ss + np.random.normal(0, 0.4, (N, N))
# Clip to avoid negative concentrations
x = np.clip(x, 0, None)
y = np.clip(y, 0, None)

# Lists to store time-series data for the center pixel
center_idx = N // 2
history_x = []
history_y = []
time_points = []

# --- 3. Helper Functions ---
def laplacian(Z):
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z
    ) / (dx ** 2)

def step(x, y):
    # Deterministic Laplacian
    Lx = laplacian(x)
    Ly = laplacian(y)
    
    # Clip to avoid negative values in sqrt
    x_clip = np.maximum(x, 0)
    y_clip = np.maximum(y, 0)
    
    # Reaction Rates
    r1 = np.ones_like(x)             # Influx
    r2 = x_clip                      # Death X
    r3 = b * x_clip                  # X -> Y
    r4 = a * (x_clip**2) * y_clip    # Autocat

    # Stochastic Terms (Langevin)
    noise_scale = 1.0 / np.sqrt(Omega)
    dW1 = np.random.normal(0, 1, (N, N)) * np.sqrt(dt)
    dW2 = np.random.normal(0, 1, (N, N)) * np.sqrt(dt)
    dW3 = np.random.normal(0, 1, (N, N)) * np.sqrt(dt)
    dW4 = np.random.normal(0, 1, (N, N)) * np.sqrt(dt)
    
    # Correlated Noise
    noise_x = noise_scale * (1*np.sqrt(r1)*dW1 - 1*np.sqrt(r2)*dW2 - 1*np.sqrt(r3)*dW3 + 1*np.sqrt(r4)*dW4)
    noise_y = noise_scale * (1*np.sqrt(r3)*dW3 - 1*np.sqrt(r4)*dW4)

    # Update
    dxdt_det = Dx * Lx + (1 - (b + 1) * x + r4)
    dydt_det = Dy * Ly + (b * x - r4)
    
    x_new = x + dxdt_det * dt + noise_x
    y_new = y + dydt_det * dt + noise_y
    
    return x_new, y_new

# --- 4. Animation & Data Collection ---
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(y, cmap='inferno', interpolation='bicubic', vmin=0, vmax=y_ss*2)
ax.axis('off')
ax.set_title(f"CLE Simulation (Omega={Omega}, a={a}, b={b}, Dx={Dx}, Dy={Dy})")

current_time = 0.0

def animate(frame):
    global x, y, current_time
    
    # Run multiple physics steps for every video frame
    for _ in range(steps_per_frame):
        x, y = step(x, y)
        current_time += dt
        
        # Record data for the center pixel
        history_x.append(x[center_idx, center_idx])
        history_y.append(y[center_idx, center_idx])
        time_points.append(current_time)
    
    img.set_array(y)
    return [img]

print("1. Generating Video...")
anim = animation.FuncAnimation(fig, animate, frames=frames, blit=True)
anim.save(f'bruss_a{a}_b{b}_om{Omega}_Dx{Dx}_Dy{Dy}.gif', writer='ffmpeg', fps=30)


# --- 5. Save the Time Series Plot ---
print("2. Generating Time Series Plot...")
plt.figure(figsize=(10, 5))
plt.plot(time_points, history_x, label='Activator (x)', color='cyan', alpha=0.8, linewidth=1)
plt.plot(time_points, history_y, label='Inhibitor (y)', color='orange', alpha=0.8, linewidth=1)
plt.title(f"Concentration at Center Pixel (Omega={Omega}, a={a}, b={b}, Dx={Dx}, Dy={Dy})")
plt.xlabel("Time")
plt.ylabel("Concentration")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'pixel_trace_a{a}_b{b}_om{Omega}_Dx{Dx}_Dy{Dy}.png', dpi=150)
print("   Saved 'pixel_trace.png'")

plt.figure(figsize=(6, 6))
plt.plot(history_x, history_y, color='purple', alpha=0.6, linewidth=0.8)
# Mark the start and end points
plt.scatter(history_x[0], history_y[0], color='green', label='Start', zorder=5)
plt.scatter(history_x[-1], history_y[-1], color='red', label='End', zorder=5)
# Mark the theoretical steady state
plt.scatter([x_ss], [y_ss], color='black', marker='x', s=100, label='Steady State', zorder=5)

plt.title("Phase Space Trajectory (y vs x)")
plt.xlabel("Concentration X")
plt.ylabel("Concentration Y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'phase_portrait_a{a}_b{b}_om{Omega}_Dx{Dx}_Dy{Dy}.png', dpi=150)