"""
This module defines the main application for simulating action potentials.
It integrates the Hodgkin-Huxley model and Nernst potential calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import reflex as rx
from .hodgkin_huxley import HodgkinHuxley


def simulate_action_potential() -> str:
    """
    Simulate a generic action potential waveform and return its plot as a base64 string.

    Returns:
        str: Base64-encoded PNG image of the action potential plot.
    """
    time = np.linspace(0, 50, 500)  # Time in ms
    voltage = -70 + 100 * np.exp(
        -((time - 25) ** 2) / (2 * 5**2)
    )  # Example AP waveform

    # Plot the action potential
    plt.figure()
    plt.plot(time, voltage)
    plt.title("Action Potential Simulation")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.grid()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # Encode the image to base64
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return image_base64


def simulate_hodgkin_huxley_with_params(params: dict) -> str:
    """
    Simulate the Hodgkin-Huxley model with user-defined parameters and return its plot as a base64 string.

    Args:
        params (dict): Dictionary containing Hodgkin-Huxley parameters.

    Returns:
        str: Base64-encoded PNG image of the Hodgkin-Huxley simulation plot.
    """
    hh_model = HodgkinHuxley(**params)
    time, voltage = hh_model.compute()

    # Plot the Hodgkin-Huxley simulation
    plt.figure()
    plt.plot(time, voltage)
    plt.title("Hodgkin-Huxley Model Simulation")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.grid()

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # Encode the image to base64
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return image_base64


def index() -> rx.Component:
    """
    Define the main page of the application, allowing users to set parameters and view simulation results.

    Returns:
        rx.Component: Reflex component containing the main page layout.
    """
    # Define state to hold user inputs and simulation results
    class AppState(rx.State):
        g_na: float = 120.0  # Sodium conductance (mS/cm^2)
        g_k: float = 36.0    # Potassium conductance (mS/cm^2)
        g_l: float = 0.3     # Leak conductance (mS/cm^2)
        v_rest: float = -65.0  # Resting potential (mV)
        hh_image_base64: str = ""

        def run_simulation(self):
            params = {
                "g_na": self.g_na,
                "g_k": self.g_k,
                "g_l": self.g_l,
                "v_rest": self.v_rest,
            }
            self.hh_image_base64 = simulate_hodgkin_huxley_with_params(params)

    return rx.box(
        rx.text("Welcome to the Action Potential Simulator!"),
        rx.text("Set Hodgkin-Huxley Model Parameters:"),
        rx.slider(label="Sodium Conductance (g_na)", min=0, max=200, step=1, value=AppState.g_na, on_change=AppState.set_g_na),
        rx.slider(label="Potassium Conductance (g_k)", min=0, max=100, step=1, value=AppState.g_k, on_change=AppState.set_g_k),
        rx.slider(label="Leak Conductance (g_l)", min=0, max=10, step=0.1, value=AppState.g_l, on_change=AppState.set_g_l),
        rx.input(label="Resting Potential (v_rest)", value=AppState.v_rest, on_change=AppState.set_v_rest),
        rx.button("Run Simulation", on_click=AppState.run_simulation),
        rx.cond(
            AppState.hh_image_base64 != "",
            rx.image(src=f"data:image/png;base64,{{AppState.hh_image_base64}}", alt="Hodgkin-Huxley Plot"),
        ),
    )


app = rx.App()
app.add_page(index)
app.compile()
