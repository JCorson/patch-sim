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


def simulate_hodgkin_huxley() -> str:
    """
    Simulate the Hodgkin-Huxley model and return its plot as a base64 string.

    Returns:
        str: Base64-encoded PNG image of the Hodgkin-Huxley simulation plot.
    """
    hh_model = HodgkinHuxley()
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
    Define the main page of the application, displaying simulation results.

    Returns:
        rx.Component: Reflex component containing the main page layout.
    """
    # Generate the action potential plot
    ap_image_base64 = simulate_action_potential()
    ap_image_src = f"data:image/png;base64,{ap_image_base64}"

    # Generate the Hodgkin-Huxley plot
    hh_image_base64 = simulate_hodgkin_huxley()
    hh_image_src = f"data:image/png;base64,{hh_image_base64}"

    return rx.box(
        rx.text("Welcome to the Action Potential Simulator!"),
        rx.image(src=ap_image_src, alt="Action Potential Plot"),
        rx.image(src=hh_image_src, alt="Hodgkin-Huxley Plot"),
    )


app = rx.App()
app.add_page(index)
app.compile()
