#!/usr/bin/env python3
"""
Example: Gaussian noise applied to joint angles
Shows the effect of observation noise with std=0.01 and std=0.025
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Generate a realistic joint angle trajectory
# ============================================================================
# UR5e joint angles typically move in range [-π, π] or [-π/2, π/2]
# Let's simulate a typical reaching trajectory

time_steps = 200
t = np.linspace(0, 4*np.pi, time_steps)

# Typical joint positions during a reaching motion (radians)
# Multiple sine waves to create natural motion
joint_angle_clean = (
    0.5 * np.sin(t) +                    # Base oscillation
    0.3 * np.sin(2*t + np.pi/4) +        # Secondary component
    0.2 * np.cos(0.5*t) +                # Slower trend
    1.0                                   # Mean offset
)

# ============================================================================
# 2. Add Gaussian noise at two different levels
# ============================================================================
noise_std_small = 0.01    # Small noise (joint_pos observation noise)
noise_std_large = 0.025   # Larger noise (joint_pos observation noise - typical)

noise_small = np.random.normal(0, noise_std_small, time_steps)
noise_large = np.random.normal(0, noise_std_large, time_steps)

joint_angle_noisy_small = joint_angle_clean + noise_small
joint_angle_noisy_large = joint_angle_clean + noise_large

# ============================================================================
# 3. Print statistics
# ============================================================================
print("=" * 70)
print("JOINT ANGLE NOISE EXAMPLE (radians)")
print("=" * 70)
print(f"\nClean signal statistics:")
print(f"  Mean: {joint_angle_clean.mean():.4f} rad ({np.degrees(joint_angle_clean.mean()):.2f}°)")
print(f"  Std:  {joint_angle_clean.std():.4f} rad")
print(f"  Range: [{joint_angle_clean.min():.4f}, {joint_angle_clean.max():.4f}] rad")
print(f"         [{np.degrees(joint_angle_clean.min()):.2f}°, {np.degrees(joint_angle_clean.max()):.2f}°]")

print(f"\nWith noise σ={noise_std_small} rad (0.01):")
print(f"  Mean: {joint_angle_noisy_small.mean():.4f} rad")
print(f"  Std:  {joint_angle_noisy_small.std():.4f} rad")
print(f"  Max deviation from clean: {np.abs(noise_small).max():.4f} rad ({np.degrees(np.abs(noise_small).max()):.2f}°)")
print(f"  RMS error: {np.sqrt(np.mean(noise_small**2)):.4f} rad")

print(f"\nWith noise σ={noise_std_large} rad (0.025):")
print(f"  Mean: {joint_angle_noisy_large.mean():.4f} rad")
print(f"  Std:  {joint_angle_noisy_large.std():.4f} rad")
print(f"  Max deviation from clean: {np.abs(noise_large).max():.4f} rad ({np.degrees(np.abs(noise_large).max()):.2f}°)")
print(f"  RMS error: {np.sqrt(np.mean(noise_large**2)):.4f} rad")

# ============================================================================
# 4. Show specific sample points
# ============================================================================
print(f"\n{' Sample points (timestep 50, 100, 150):':~^70}")
for idx in [50, 100, 150]:
    print(f"\nTime step {idx}:")
    print(f"  Clean:         {joint_angle_clean[idx]:7.4f} rad ({np.degrees(joint_angle_clean[idx]):7.2f}°)")
    print(f"  + noise (0.01) {joint_angle_noisy_small[idx]:7.4f} rad ({np.degrees(joint_angle_noisy_small[idx]):7.2f}°)  →  error: {noise_small[idx]:+.4f} rad ({np.degrees(noise_small[idx]):+.2f}°)")
    print(f"  + noise (0.025){joint_angle_noisy_large[idx]:7.4f} rad ({np.degrees(joint_angle_noisy_large[idx]):7.2f}°)  →  error: {noise_large[idx]:+.4f} rad ({np.degrees(noise_large[idx]):+.2f}°)")

# ============================================================================
# 5. Plot
# ============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Joint Angle Observation Noise Example', fontsize=16, fontweight='bold')

# Plot 1: Clean vs noisy signals
ax = axes[0]
ax.plot(t, np.degrees(joint_angle_clean), 'k-', linewidth=2, label='Clean signal', zorder=3)
ax.plot(t, np.degrees(joint_angle_noisy_small), 'b.', alpha=0.6, label=f'+ Gaussian noise σ=0.01 rad', markersize=4)
ax.scatter(t[::20], np.degrees(joint_angle_noisy_small[::20]), c='blue', s=20, alpha=0.8, zorder=2)
ax.set_ylabel('Angle (degrees)', fontsize=11)
ax.set_title('Small Noise: σ=0.01 rad (~0.57°)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
ax.set_xlim(t[0], t[-1])

# Plot 2: Clean vs larger noise
ax = axes[1]
ax.plot(t, np.degrees(joint_angle_clean), 'k-', linewidth=2, label='Clean signal', zorder=3)
ax.plot(t, np.degrees(joint_angle_noisy_large), 'r.', alpha=0.6, label=f'+ Gaussian noise σ=0.025 rad', markersize=4)
ax.scatter(t[::20], np.degrees(joint_angle_noisy_large[::20]), c='red', s=20, alpha=0.8, zorder=2)
ax.set_ylabel('Angle (degrees)', fontsize=11)
ax.set_title('Larger Noise: σ=0.025 rad (~1.43°)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
ax.set_xlim(t[0], t[-1])

# Plot 3: Noise comparison / error envelope
ax = axes[2]
ax.fill_between(t, np.degrees(-2*noise_std_small), np.degrees(2*noise_std_small), 
                alpha=0.3, color='blue', label='±2σ envelope (σ=0.01 rad)')
ax.fill_between(t, np.degrees(-2*noise_std_large), np.degrees(2*noise_std_large), 
                alpha=0.2, color='red', label='±2σ envelope (σ=0.025 rad)')
ax.plot(t, np.degrees(noise_small), 'b-', alpha=0.7, linewidth=1, label=f'Actual noise (σ=0.01)')
ax.plot(t, np.degrees(noise_large), 'r-', alpha=0.7, linewidth=1, label=f'Actual noise (σ=0.025)')
ax.axhline(0, color='k', linestyle='--', alpha=0.5, linewidth=0.8)
ax.set_xlabel('Time (arbitrary units)', fontsize=11)
ax.set_ylabel('Noise magnitude (degrees)', fontsize=11)
ax.set_title('Noise Magnitude Comparison', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
ax.set_xlim(t[0], t[-1])

plt.tight_layout()
plt.savefig('/home/lrenevey/Woodworking_Simulation/noise_example.png', dpi=150, bbox_inches='tight')
print(f"\n{'':~^70}")
print(f"Plot saved to: noise_example.png")
print(f"{'':~^70}")
plt.show()

# ============================================================================
# 6. Context: What do these values mean?
# ============================================================================
print(f"\n{'INTERPRETATION':~^70}")
print(f"""
σ = 0.01 rad (~0.57°):
  - Very small noise, typical for precise encoders
  - Most samples within ±0.02 rad (±1.15°) of true value
  - Barely noticeable visually
  - Good for high-precision operations

σ = 0.025 rad (~1.43°):
  - Moderate noise, typical observation standard
  - Most samples within ±0.05 rad (±2.87°) of true value
  - Clearly visible but manageable for learning
  - Simulates realistic sensor imprecision + latency effects

Context (UR5e joint angles):
  - Joint range: typically [-π, π] → [-180°, 180°]
  - 0.01 rad error = 0.01/π ≈ 0.3% of full range
  - 0.025 rad error = 0.025/π ≈ 0.8% of full range
""")
