"""
Free response of a linear spring-mass-damper system.

Model:
    m*x_ddot + c*x_dot + k*x = 0

Positive-power convention:
    Positive power means power delivered TO the mass.

The script plots:
    1. Displacement and velocity
    2. Spring, damper, and net forces
    3. Instantaneous component powers
    4. Energy-storage and dissipation rates
    5. Stored and dissipated energies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp, cumulative_trapezoid

# ============================================================
# 1. Model parameters: simple illustrative SI-unit defaults
# ============================================================

m = 1.0  # Mass [kg]
c = 0.5  # Viscous damping coefficient [N*s/m]
k = 10.0  # Spring stiffness [N/m]

x0 = 0.0  # Initial displacement [m]
v0 = 0.0  # Initial velocity [m/s]

t_start = 0.0  # Initial time [s]
t_end = 20.0  # Final time [s]
n_points = 100001

t_eval = np.linspace(t_start, t_end, n_points)

# ============================================================
# 1c. Applied force function
# ============================================================
# Applied force chosen for the forced response: a sinusoid [N].
# input_force_amplitude = k * x0 + c * v0 / 2
# input_force_amplitude = 0
# input_force_frequency = 2 * np.sqrt(k / m)  # Angular frequency [rad/s]

# Small pulse force to move the block from x0 to a target displacement,
# applied only for a short duration [N].
pulse_target_displacement = 0.05  # Target displacement [m]
pulse_duration = 1.0  # Duration of the applied pulse [s]
pulse_force_amplitude = k * pulse_target_displacement  # Force to hold at target


def applied_force(time):
    """External force applied to the mass; positive is to the right."""
    # oscillatory_force = (
    #     input_force_amplitude * (np.cos(input_force_frequency * time) + 1.0) / 2.0
    # )
    pulse_force = np.where(time <= pulse_duration, pulse_force_amplitude, 0.0)
    return pulse_force


# ============================================================
# 2. Differential equation
# ============================================================


def spring_mass_damper(t, state):
    """
    State vector:
        state[0] = x = displacement
        state[1] = v = velocity

    Returns:
        dx/dt = v
        dv/dt = acceleration
    """
    x, v = state

    acceleration = (applied_force(t) - c * v - k * x) / m

    return [v, acceleration]


# ============================================================
# 3. Numerical simulation
# ============================================================

solution = solve_ivp(
    fun=spring_mass_damper,
    t_span=(t_start, t_end),
    y0=[x0, v0],
    t_eval=t_eval,
    method="RK45",
    rtol=1e-9,
    atol=1e-11,
)

if not solution.success:
    raise RuntimeError(solution.message)

t = solution.t
x = solution.y[0]
v = solution.y[1]

# Calculate acceleration directly from the equation of motion
F_input = applied_force(t)
a = (F_input - c * v - k * x) / m


# ============================================================
# 4. Forces acting on the mass
# ============================================================

# Forces are defined as forces exerted ON the mass
F_spring = -k * x
F_damper = -c * v
F_net = F_input + F_spring + F_damper

# Newton's second law check:
# F_net should equal m*a
force_balance_error = F_net - m * a


# ============================================================
# 5. Instantaneous powers
# ============================================================

# Convention:
# Positive power means that a component supplies energy to
# the mass. Negative power means that it removes energy.

# Spring power delivered to the mass
P_spring = F_spring * v  # = -k*x*v

# Damper power delivered to the mass
P_damper = F_damper * v  # = -c*v^2 <= 0

# Net power delivered to the mass
P_net = F_net * v  # = m*a*v = dK/dt

# Power supplied by the externally applied force.
P_input = F_input * v

# Positive power absorbed/dissipated by the damper
P_dissipated = -P_damper  # = c*v^2 >= 0


# ============================================================
# 6. Energy-storage rates
# ============================================================

# Rate of change of kinetic energy
dK_dt = m * a * v

# Rate of change of spring potential energy
dU_dt = k * x * v

# Rate of change of total mechanical energy
dE_dt = dK_dt + dU_dt

# Forced-response energy balance: dE/dt = P_input + P_damper.
power_balance_error = dE_dt - P_input - P_damper


# ============================================================
# 7. Energies
# ============================================================

kinetic_energy = 0.5 * m * v**2
spring_energy = 0.5 * k * x**2
mechanical_energy = kinetic_energy + spring_energy

# Energy converted by the damper from time zero until time t
dissipated_energy = cumulative_trapezoid(P_dissipated, t, initial=0.0)
input_energy = cumulative_trapezoid(P_input, t, initial=0.0)

# Ideally: mechanical energy plus dissipation equals initial plus supplied energy.
energy_balance_error = (
    mechanical_energy + dissipated_energy - mechanical_energy[0] - input_energy
)


# ============================================================
# 7b. Active and reactive power (mechanical analogy)
# ============================================================
# Active power: power dissipated by the damper (resistive analog).
# Reactive power: power exchanged with the storage elements
# (spring + mass), i.e. the non-dissipative power.

active_power_mech = P_dissipated
reactive_power_mech = dU_dt + dK_dt


# ============================================================
# 8. Useful system properties
# ============================================================

natural_frequency = np.sqrt(k / m)
critical_damping = 2.0 * np.sqrt(k * m)
damping_ratio = c / critical_damping

if damping_ratio < 1.0:
    response_type = "underdamped"
    damped_frequency = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
elif np.isclose(damping_ratio, 1.0):
    response_type = "critically damped"
    damped_frequency = 0.0
else:
    response_type = "overdamped"
    damped_frequency = np.nan


# ============================================================
# 9. Print simulation information
# ============================================================

print("Spring-mass-damper forced response")
print("--------------------------------")
print(f"Mass                 : {m:.3f} kg")
print(f"Damping coefficient  : {c:.3f} N*s/m")
print(f"Spring stiffness      : {k:.3f} N/m")
print(f"Initial displacement  : {x0:.3f} m")
print(f"Initial velocity      : {v0:.3f} m/s")
print(f"Pulse target displacement : {pulse_target_displacement:.3f} m")
print(f"Pulse duration        : {pulse_duration:.3f} s")
print(f"Pulse force amplitude : {pulse_force_amplitude:.3f} N")
print()
print(f"Natural frequency     : {natural_frequency:.4f} rad/s")
print(f"Critical damping      : {critical_damping:.4f} N*s/m")
print(f"Damping ratio         : {damping_ratio:.4f}")
print(f"Response type         : {response_type}")

if damping_ratio < 1.0:
    print(f"Damped frequency      : {damped_frequency:.4f} rad/s")

print()
print(f"Initial mechanical energy : " f"{mechanical_energy[0]:.8f} J")
print(f"Final mechanical energy   : " f"{mechanical_energy[-1]:.8f} J")
print(f"Dissipated energy         : " f"{dissipated_energy[-1]:.8f} J")
print()
print(f"Maximum force-balance error : " f"{np.max(np.abs(force_balance_error)):.3e} N")
print(f"Maximum power-balance error : " f"{np.max(np.abs(power_balance_error)):.3e} W")
print(f"Maximum energy-balance error: " f"{np.max(np.abs(energy_balance_error)):.3e} J")


# ============================================================
# 10. Plots
# ============================================================

fig, axes = plt.subplots(4, 1, sharex=True, constrained_layout=True, num=20, clear=True)

# Response inputs and state derivatives.
axes[0].plot(t, x, label=r"Displacement $x$ [m]", color="tab:blue")
axes[0].plot(t, v, label=r"Velocity $v$ [m/s]", color="tab:orange")
axes[0].plot(t, a, label=r"Acceleration $a$ [m/s$^2$]", color="tab:purple")
axes[0].plot(t, F_input, label=r"Input force $F_{in}$ [N]", color="tab:green")
axes[0].set_title("Spring–mass–damper response")
axes[0].set_ylabel("State value")

# Stored energy and energy dissipated by the damper.
axes[1].plot(t, spring_energy, label="Spring potential energy [J]", color="tab:blue")
axes[1].plot(t, kinetic_energy, label="Mass kinetic energy [J]", color="tab:orange")
axes[1].plot(t, mechanical_energy, label="Total mechanical energy [J]", color="black")
axes[1].plot(
    t, dissipated_energy, label="Cumulative dissipated energy [J]", color="tab:red"
)
axes[1].plot(t, input_energy, label="Cumulative input energy [J]", color="tab:green")
axes[1].set_title("Energy storage and dissipation")
axes[1].set_ylabel("Energy [J]")

# Instantaneous powers: rates of change of each energy quantity.
axes[2].plot(t, dU_dt, label=r"Spring power $dU/dt$ [W]", color="tab:blue")
axes[2].plot(t, dK_dt, label=r"Mass power $dK/dt$ [W]", color="tab:orange")
axes[2].plot(t, dE_dt, label=r"Mechanical power $dE/dt$ [W]", color="black")
axes[2].plot(t, P_dissipated, label=r"Dissipated power $cv^2$ [W]", color="tab:red")
axes[2].plot(t, P_input, label=r"Input power $F_{in}v$ [W]", color="tab:green")
axes[2].axhline(0.0, color="gray", linewidth=0.8)
axes[2].set_title("Instantaneous power and energy rates")
axes[2].set_ylabel("Power [W]")

# Active (damper) and reactive (spring + mass) power.
axes[3].plot(t, active_power_mech, label=r"Active power $cv^2$ [W]", color="tab:red")
axes[3].plot(
    t,
    reactive_power_mech,
    label=r"Reactive power $dU/dt + dK/dt$ [W]",
    color="tab:green",
)
axes[3].axhline(0.0, color="gray", linewidth=0.8)
axes[3].set_title("Active and reactive power")
axes[3].set_ylabel("Power [W]")
axes[3].set_xlabel("Time [s]")

for axis in axes:
    axis.legend(ncol=2)
    axis.grid(True, alpha=0.3)

fig.suptitle("Spring–Mass–Damper Simulation", fontsize=15)


# ============================================================
# 11. Animated mechanical motion, power, and energy
# ============================================================

# Limit the number of rendered frames while retaining the full simulation data.
frame_indices = np.linspace(0, len(t) - 1, min(600, len(t)), dtype=int)
motion_fig, (motion_ax, power_ax, energy_ax) = plt.subplots(
    3, 1, constrained_layout=True, num=22, clear=True
)

motion_ax.set_title("Animated spring–mass–damper motion")
motion_ax.set_xlim(-0.18, 0.28)
motion_ax.set_ylim(-0.10, 0.10)
motion_ax.set_yticks([])
motion_ax.set_xlabel("Displacement position [m]")
motion_ax.grid(True, axis="x", alpha=0.3)

wall_x = -0.15
equilibrium_x = 0.05
mass_width = 0.05
mass_height = 0.06
motion_ax.axvline(wall_x, color="black", linewidth=4)
motion_ax.axvline(equilibrium_x, color="gray", linestyle="--", label="Equilibrium")
(spring_line,) = motion_ax.plot([], [], color="tab:blue", linewidth=2, label="Spring")
(damper_line,) = motion_ax.plot([], [], color="tab:red", linewidth=4, label="Damper")
mass_patch = plt.Rectangle(
    (0, -mass_height / 2), mass_width, mass_height, color="tab:orange"
)
motion_ax.add_patch(mass_patch)
motion_text = motion_ax.text(0.02, 0.88, "", transform=motion_ax.transAxes)
input_force_arrow = motion_ax.annotate(
    "",
    xy=(0, 0.045),
    xytext=(0, 0.045),
    arrowprops=dict(arrowstyle="->", color="tab:green", lw=2),
)
motion_ax.legend(loc="lower right")

power_ax.set_title("Instantaneous power")
power_labels = [
    r"Spring $dU/dt$",
    r"Mass $dK/dt$",
    r"Dissipated $dE_d/dt$",
    r"Input $F_{in}v$",
]
power_bars = power_ax.bar(
    power_labels,
    [0.0, 0.0, 0.0, 0.0],
    color=["tab:blue", "tab:orange", "tab:red", "tab:green"],
    alpha=0.8,
)
power_ax.axhline(0, color="gray", linewidth=0.8)
power_limit = max(
    np.max(np.abs(dU_dt)),
    np.max(np.abs(dK_dt)),
    np.max(np.abs(P_dissipated)),
    np.max(np.abs(P_input)),
    1e-12,
)
power_ax.set_ylim(-1.15 * power_limit, 1.15 * power_limit)
power_ax.set_ylabel("Power [W]")
power_ax.grid(True, alpha=0.3)

energy_ax.set_title("Stored and dissipated energy")
energy_labels = ["Spring potential", "Mass kinetic", "Dissipated", "Input"]
energy_bars = energy_ax.bar(
    energy_labels,
    [0.0, 0.0, 0.0, 0.0],
    color=["tab:blue", "tab:orange", "tab:red", "tab:green"],
    alpha=0.8,
)
energy_limit = max(
    np.max(spring_energy), np.max(kinetic_energy), np.max(dissipated_energy), 1e-12
)
energy_limit = max(energy_limit, np.max(np.abs(input_energy)))
energy_ax.set_ylim(-1.15 * energy_limit, 1.15 * energy_limit)
energy_ax.set_xlabel("Time [s]")
energy_ax.set_ylabel("Energy [J]")
energy_ax.grid(True, alpha=0.3)


def animate(frame):
    """Update the mechanical drawing and instantaneous power/energy bars."""
    index = frame_indices[frame]
    mass_x = equilibrium_x + x[index]
    mass_patch.set_x(mass_x)

    # Draw a zig-zag spring from the wall to the moving mass.
    spring_end = mass_x
    coils = 10
    spring_x = np.linspace(wall_x, spring_end, 2 * coils + 3)
    spring_y = np.zeros_like(spring_x)
    spring_y[1:-1] = 0.018 * np.tile([1, -1], coils + 1)[: len(spring_y) - 2]
    spring_line.set_data(spring_x, spring_y)
    damper_line.set_data([wall_x, mass_x], [-0.055, -0.055])

    # Draw the applied-force arrow at the mass; its direction and length show force.
    arrow_length = 0.07 * F_input[index] / max(pulse_force_amplitude, 1e-12)
    arrow_start = mass_x + mass_width / 2
    input_force_arrow.set_position((arrow_start, 0.045))
    input_force_arrow.xy = (arrow_start + arrow_length, 0.045)

    # Show the rate of change of each energy quantity.
    power_bars[0].set_height(dU_dt[index])
    power_bars[1].set_height(dK_dt[index])
    power_bars[2].set_height(P_dissipated[index])
    power_bars[3].set_height(P_input[index])

    # Show spring potential, mass kinetic, and cumulative dissipated energy.
    energy_bars[0].set_height(spring_energy[index])
    energy_bars[1].set_height(kinetic_energy[index])
    energy_bars[2].set_height(dissipated_energy[index])
    energy_bars[3].set_height(input_energy[index])
    motion_text.set_text(
        f"t = {t[index]:.2f} s    x = {x[index]:.4f} m    v = {v[index]:.4f} m/s"
    )
    return (
        spring_line,
        damper_line,
        mass_patch,
        input_force_arrow,
        *power_bars,
        *energy_bars,
        motion_text,
    )


motion_animation = FuncAnimation(
    motion_fig,
    animate,
    frames=len(frame_indices),
    # Use the simulated time between rendered frames as the wall-clock
    # interval, so one second of simulation takes one second to animate.
    interval=1000.0 * (t[frame_indices[1]] - t[frame_indices[0]]),
    blit=False,
    repeat=True,
)


plt.show()
