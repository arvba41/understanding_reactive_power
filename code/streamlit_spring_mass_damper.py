"""Interactive Streamlit version of the pulse-driven spring-mass-damper model.

Run with:
    python3 -m streamlit run streamlit_spring_mass_damper_analogy.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.integrate import cumulative_trapezoid, solve_ivp

st.set_page_config(page_title="Spring-Mass-Damper Power", layout="wide")


@st.cache_data(show_spinner=False)
def simulate_system(
    mass: float,
    damping: float,
    stiffness: float,
    initial_displacement: float,
    initial_velocity: float,
    target_displacement: float,
    pulse_duration: float,
    end_time: float,
    sample_count: int,
) -> dict[str, np.ndarray | float | str]:
    """Solve the forced system and derive forces, powers, energies, and residuals."""
    time = np.linspace(0.0, end_time, sample_count)
    pulse_amplitude = stiffness * target_displacement

    def applied_force(current_time: float | np.ndarray) -> float | np.ndarray:
        return np.where(current_time <= pulse_duration, pulse_amplitude, 0.0)

    def state_derivative(current_time: float, state: np.ndarray) -> list[float]:
        displacement, velocity = state
        acceleration = (
            applied_force(current_time) - damping * velocity - stiffness * displacement
        ) / mass
        return [velocity, acceleration]

    solution = solve_ivp(
        fun=state_derivative,
        t_span=(0.0, end_time),
        y0=[initial_displacement, initial_velocity],
        t_eval=time,
        method="RK45",
        rtol=1e-9,
        atol=1e-11,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    displacement = solution.y[0]
    velocity = solution.y[1]
    input_force = applied_force(time)
    acceleration = (input_force - damping * velocity - stiffness * displacement) / mass

    spring_force = -stiffness * displacement
    damper_force = -damping * velocity
    net_force = input_force + spring_force + damper_force
    mass_force = -mass * acceleration
    force_balance_error = net_force - mass * acceleration

    spring_power = spring_force * velocity
    damper_power = damper_force * velocity
    input_power = input_force * velocity
    net_power = net_force * velocity
    dissipated_power = -damper_power

    kinetic_rate = mass * acceleration * velocity
    spring_energy_rate = stiffness * displacement * velocity
    mechanical_energy_rate = kinetic_rate + spring_energy_rate
    power_balance_error = mechanical_energy_rate - input_power - damper_power

    kinetic_energy = 0.5 * mass * velocity**2
    spring_energy = 0.5 * stiffness * displacement**2
    mechanical_energy = kinetic_energy + spring_energy
    dissipated_energy = cumulative_trapezoid(dissipated_power, time, initial=0.0)
    input_energy = cumulative_trapezoid(input_power, time, initial=0.0)
    energy_balance_error = (
        mechanical_energy + dissipated_energy - mechanical_energy[0] - input_energy
    )

    natural_frequency = np.sqrt(stiffness / mass)
    critical_damping = 2.0 * np.sqrt(stiffness * mass)
    damping_ratio = damping / critical_damping
    if damping_ratio < 1.0:
        response_type = "Underdamped"
        damped_frequency = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
    elif np.isclose(damping_ratio, 1.0):
        response_type = "Critically damped"
        damped_frequency = 0.0
    else:
        response_type = "Overdamped"
        damped_frequency = np.nan

    return {
        "time": time,
        "displacement": displacement,
        "velocity": velocity,
        "acceleration": acceleration,
        "input_force": input_force,
        "spring_force": spring_force,
        "damper_force": damper_force,
        "net_force": net_force,
        "mass_force": mass_force,
        "force_balance_error": force_balance_error,
        "spring_power": spring_power,
        "damper_power": damper_power,
        "input_power": input_power,
        "net_power": net_power,
        "dissipated_power": dissipated_power,
        "kinetic_rate": kinetic_rate,
        "spring_energy_rate": spring_energy_rate,
        "mechanical_energy_rate": mechanical_energy_rate,
        "power_balance_error": power_balance_error,
        "kinetic_energy": kinetic_energy,
        "spring_energy": spring_energy,
        "mechanical_energy": mechanical_energy,
        "dissipated_energy": dissipated_energy,
        "input_energy": input_energy,
        "energy_balance_error": energy_balance_error,
        "pulse_amplitude": pulse_amplitude,
        "natural_frequency": natural_frequency,
        "critical_damping": critical_damping,
        "damping_ratio": damping_ratio,
        "damped_frequency": damped_frequency,
        "response_type": response_type,
    }


def draw_time_series(results: dict[str, np.ndarray | float | str]) -> plt.Figure:
    """Create the five time-series panels from the original script."""
    time = results["time"]
    figure, axes = plt.subplots(
        5, 1, figsize=(12, 17), sharex=True, constrained_layout=True
    )

    axes[0].plot(
        time, results["displacement"], label="Displacement x [m]", color="tab:blue"
    )
    axes[0].plot(
        time, results["velocity"], label="Velocity v [m/s]", color="tab:orange"
    )
    axes[0].plot(
        time,
        results["acceleration"],
        label="Acceleration a [m/s^2]",
        color="tab:purple",
    )
    axes[0].plot(
        time, results["input_force"], label="Input force F_in [N]", color="tab:green"
    )
    axes[0].set_title("Spring-mass-damper response")
    axes[0].set_ylabel("State value")

    axes[1].plot(
        time, results["spring_force"], label="Spring force -kx [N]", color="tab:blue"
    )
    axes[1].plot(
        time,
        results["mass_force"],
        label="Mass inertial force -ma [N]",
        color="tab:orange",
    )
    axes[1].plot(
        time, results["damper_force"], label="Damper force -cv [N]", color="tab:red"
    )
    axes[1].plot(
        time, results["input_force"], label="Input force F_in [N]", color="tab:green"
    )
    axes[1].set_title("Instantaneous forces")
    axes[1].set_ylabel("Force [N]")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)

    axes[2].plot(
        time,
        results["spring_energy"],
        label="Spring potential energy [J]",
        color="tab:blue",
    )
    axes[2].plot(
        time,
        results["kinetic_energy"],
        label="Mass kinetic energy [J]",
        color="tab:orange",
    )
    axes[2].plot(
        time, results["mechanical_energy"], label="Mechanical energy [J]", color="black"
    )
    axes[2].plot(
        time,
        results["dissipated_energy"],
        label="Cumulative dissipated energy [J]",
        color="tab:red",
    )
    axes[2].plot(
        time,
        results["input_energy"],
        label="Cumulative input energy [J]",
        color="tab:green",
    )
    axes[2].set_title("Energy storage and dissipation")
    axes[2].set_ylabel("Energy [J]")

    axes[3].plot(
        time,
        results["spring_energy_rate"],
        label="Spring power dU/dt [W]",
        color="tab:blue",
    )
    axes[3].plot(
        time, results["kinetic_rate"], label="Mass power dK/dt [W]", color="tab:orange"
    )
    axes[3].plot(
        time,
        results["mechanical_energy_rate"],
        label="Mechanical power dE/dt [W]",
        color="black",
    )
    axes[3].plot(
        time,
        results["dissipated_power"],
        label="Dissipated power cv^2 [W]",
        color="tab:red",
    )
    axes[3].plot(
        time, results["input_power"], label="Input power F_in v [W]", color="tab:green"
    )
    axes[3].set_title("Instantaneous power and energy rates")
    axes[3].set_ylabel("Power [W]")
    axes[3].axhline(0.0, color="gray", linewidth=0.8)

    axes[4].plot(
        time,
        results["dissipated_power"],
        label="Active power cv^2 [W]",
        color="tab:red",
    )
    axes[4].plot(
        time,
        results["mechanical_energy_rate"],
        label="Reactive power dU/dt + dK/dt [W]",
        color="tab:green",
    )
    axes[4].set_title("Active and reactive power")
    axes[4].set_xlabel("Time [s]")
    axes[4].set_ylabel("Power [W]")
    axes[4].axhline(0.0, color="gray", linewidth=0.8)

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=2, fontsize=9)
    return figure


def draw_snapshot(
    results: dict[str, np.ndarray | float | str], sample_index: int
) -> plt.Figure:
    """Draw the mechanical state and corresponding force, power, and energy bars."""
    displacement = results["displacement"][sample_index]
    input_force = results["input_force"][sample_index]
    pulse_amplitude = results["pulse_amplitude"]

    figure, (motion_axis, force_axis, power_axis, energy_axis) = plt.subplots(
        1,
        4,
        figsize=(18, 4.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 1, 1, 1]},
    )

    wall_position = -0.15
    equilibrium_position = 0.05
    mass_width = 0.05
    mass_height = 0.06
    mass_position = equilibrium_position + displacement
    motion_axis.axvline(wall_position, color="black", linewidth=4)
    motion_axis.axvline(
        equilibrium_position, color="gray", linestyle="--", label="Equilibrium"
    )
    coil_count = 10
    spring_positions = np.linspace(wall_position, mass_position, 2 * coil_count + 3)
    spring_offsets = np.zeros_like(spring_positions)
    spring_offsets[1:-1] = (
        0.018 * np.tile([1, -1], coil_count + 1)[: len(spring_offsets) - 2]
    )
    motion_axis.plot(
        spring_positions, spring_offsets, color="tab:blue", linewidth=2, label="Spring"
    )
    motion_axis.plot(
        [wall_position, mass_position],
        [-0.055, -0.055],
        color="tab:red",
        linewidth=4,
        label="Damper",
    )
    motion_axis.add_patch(
        plt.Rectangle(
            (mass_position, -mass_height / 2),
            mass_width,
            mass_height,
            color="tab:orange",
        )
    )
    arrow_scale = 0.07 * input_force / max(abs(pulse_amplitude), 1e-12)
    arrow_start = mass_position + mass_width / 2
    motion_axis.annotate(
        "",
        xy=(arrow_start + arrow_scale, 0.045),
        xytext=(arrow_start, 0.045),
        arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 2},
    )
    motion_axis.set_title("Mechanical state")
    motion_axis.set_xlim(-0.18, 0.28)
    motion_axis.set_ylim(-0.1, 0.1)
    motion_axis.set_yticks([])
    motion_axis.set_xlabel("Position [m]")
    motion_axis.grid(True, axis="x", alpha=0.3)
    motion_axis.legend(loc="lower right")

    force_names = ["Spring", "Inertia", "Damper", "Input"]
    force_values = [
        results["spring_force"][sample_index],
        results["mass_force"][sample_index],
        results["damper_force"][sample_index],
        results["input_force"][sample_index],
    ]
    force_axis.bar(
        force_names,
        force_values,
        color=["tab:blue", "tab:orange", "tab:red", "tab:green"],
    )
    force_axis.axhline(0.0, color="gray", linewidth=0.8)
    force_axis.set_title("Forces on the mass")
    force_axis.set_ylabel("Force [N]")
    force_axis.grid(True, axis="y", alpha=0.3)

    power_names = ["Spring", "Mass", "Dissipated", "Input"]
    power_values = [
        results["spring_energy_rate"][sample_index],
        results["kinetic_rate"][sample_index],
        results["dissipated_power"][sample_index],
        results["input_power"][sample_index],
    ]
    power_axis.bar(
        power_names,
        power_values,
        color=["tab:blue", "tab:orange", "tab:red", "tab:green"],
    )
    power_axis.axhline(0.0, color="gray", linewidth=0.8)
    power_axis.set_title("Instantaneous power")
    power_axis.set_ylabel("Power [W]")
    power_axis.grid(True, axis="y", alpha=0.3)

    energy_names = ["Spring", "Kinetic", "Dissipated", "Input"]
    energy_values = [
        results["spring_energy"][sample_index],
        results["kinetic_energy"][sample_index],
        results["dissipated_energy"][sample_index],
        results["input_energy"][sample_index],
    ]
    energy_axis.bar(
        energy_names,
        energy_values,
        color=["tab:blue", "tab:orange", "tab:red", "tab:green"],
    )
    energy_axis.axhline(0.0, color="gray", linewidth=0.8)
    energy_axis.set_title("Stored and accumulated energy")
    energy_axis.set_ylabel("Energy [J]")
    energy_axis.grid(True, axis="y", alpha=0.3)
    return figure


st.title("Spring-Mass-Damper: Power and Energy")
st.caption(
    "Forced response: m x'' + c x' + k x = F_in(t). Positive component power is delivered to the mass."
)

with st.sidebar:
    st.header("Model")
    mass = st.number_input("Mass m [kg]", min_value=0.01, value=1.0, step=0.1)
    damping = st.number_input("Damping c [N s/m]", min_value=0.0, value=0.5, step=0.1)
    stiffness = st.number_input(
        "Stiffness k [N/m]", min_value=0.01, value=10.0, step=0.5
    )
    st.header("Initial state")
    initial_displacement = st.number_input(
        "Displacement x0 [m]", value=0.0, step=0.01, format="%.3f"
    )
    initial_velocity = st.number_input(
        "Velocity v0 [m/s]", value=0.0, step=0.01, format="%.3f"
    )
    st.header("Input pulse")
    target_displacement = st.number_input(
        "Target displacement [m]", min_value=0.0, value=0.05, step=0.01, format="%.3f"
    )
    pulse_duration = st.number_input(
        "Pulse duration [s]", min_value=0.0, value=1.0, step=0.1
    )
    st.header("Simulation")
    end_time = st.slider(
        "End time [s]", min_value=1.0, max_value=60.0, value=20.0, step=1.0
    )
    sample_count = st.select_slider(
        "Output samples", options=[1001, 5001, 10001, 20001, 50001], value=10001
    )

if pulse_duration > end_time:
    st.warning(
        "The pulse duration exceeds the simulation duration, so the input remains on throughout the displayed interval."
    )

try:
    results = simulate_system(
        mass,
        damping,
        stiffness,
        initial_displacement,
        initial_velocity,
        target_displacement,
        pulse_duration,
        end_time,
        sample_count,
    )
except RuntimeError as error:
    st.error(f"The numerical solver did not complete: {error}")
    st.stop()

property_columns = st.columns(5)
property_columns[0].metric("Pulse amplitude", f"{results['pulse_amplitude']:.4g} N")
property_columns[1].metric(
    "Natural frequency", f"{results['natural_frequency']:.4g} rad/s"
)
property_columns[2].metric(
    "Critical damping", f"{results['critical_damping']:.4g} N s/m"
)
property_columns[3].metric("Damping ratio", f"{results['damping_ratio']:.4g}")
property_columns[4].metric("Response", str(results["response_type"]))

st.subheader("Time history")
time_figure = draw_time_series(results)
st.pyplot(time_figure, use_container_width=True)
plt.close(time_figure)

st.subheader("Inspect one instant")
selected_time = st.slider(
    "Time cursor [s]",
    min_value=0.0,
    max_value=float(end_time),
    value=0.0,
    step=float(end_time / (sample_count - 1)),
)
sample_index = int(np.abs(results["time"] - selected_time).argmin())
actual_time = results["time"][sample_index]
st.caption(
    f"Displayed sample: t = {actual_time:.5f} s, x = {results['displacement'][sample_index]:.5f} m, "
    f"v = {results['velocity'][sample_index]:.5f} m/s"
)
snapshot_figure = draw_snapshot(results, sample_index)
st.pyplot(snapshot_figure, use_container_width=True)
plt.close(snapshot_figure)

st.subheader("Balance diagnostics")
diagnostic_columns = st.columns(4)
diagnostic_columns[0].metric(
    "Final mechanical energy", f"{results['mechanical_energy'][-1]:.6g} J"
)
diagnostic_columns[1].metric(
    "Final dissipated energy", f"{results['dissipated_energy'][-1]:.6g} J"
)
diagnostic_columns[2].metric(
    "Max force residual", f"{np.max(np.abs(results['force_balance_error'])):.3e} N"
)
diagnostic_columns[3].metric(
    "Max power residual", f"{np.max(np.abs(results['power_balance_error'])):.3e} W"
)
st.metric(
    "Max energy residual", f"{np.max(np.abs(results['energy_balance_error'])):.3e} J"
)

with st.expander("Equations and conventions"):
    st.latex(r"m\ddot{x} + c\dot{x} + kx = F_{in}(t)")
    st.latex(r"F_{spring}=-kx,\quad F_{damper}=-c\dot{x},\quad P=F\dot{x}")
    st.latex(r"\frac{d}{dt}(K+U)=P_{input}+P_{damper}=P_{input}-c\dot{x}^2")
    st.write(
        "The pulse amplitude is k times the target displacement. The spring and mass exchange stored energy; the damper removes energy at c v squared."
    )
