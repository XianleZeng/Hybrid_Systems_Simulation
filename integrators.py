import numpy as np
import scipy
from absl import logging

logging.set_verbosity(logging.DEBUG)


# Define a helper function for the event proxy
# This function returns a lambda function that wraps around any provided function (f)
# It allows us to pass the event function in a flexible way while preserving its signature
def _proxy(f):
    return lambda *args, **kwargs: f(*args, **kwargs)


# Constants used to define tolerances for the hybrid integration
GUARD_TOL = 1e-4  # Tolerance for the guard condition (to detect discrete jumps)
PROGRESS_TOL = 1e-4  # Tolerance for checking progress between event detections


# Main hybrid integrator function
# This function solves the hybrid dynamical system using numerical integration,
# accounting for both continuous evolution (ordinary differential equation) and discrete jumps
def hybrid_integrator(f, event, direction, guard, state0, t0, tf):
    """
    f: function representing the continuous dynamics (the ODE)
    event: function that defines the event triggering discrete transitions (guard function)
    direction: the direction of zero crossing (for detecting event condition)
    guard: function defining how the state changes after an event (discrete dynamics)
    state0: initial state of the system
    t0: start time of the simulation
    tf: end time of the simulation
    """

    # Helper function to make a SciPy-compatible event for integration
    def make_scipy_event(event, direction, terminal):
        """
        event: the event function (guard condition)
        direction: the direction in which zero-crossing is detected (positive, negative, or both)
        terminal: if True, stop the integration when the event occurs
        """
        fn = _proxy(event)  # Use the proxy to wrap the event function
        fn.direction = direction  # Set the direction of the zero crossing for the event
        fn.terminal = terminal  # Indicate whether the event is terminal (stops the integration)
        return fn

    # Initialization
    tcur = t0  # Current time starts at the initial time t0
    t_total = []  # List to accumulate all the time points
    y_total = []  # List to accumulate all the state values over time
    jump_times = []  # List to store the times at which discrete events (jumps) occur
    state = state0  # Initial state
    method = "RK45"  # The numerical method used for continuous integration (Runge-Kutta 4(5))

    # Main loop for hybrid integration
    while True:
        # Evaluate the event function at the current time and state (check for event)
        cur_witness = event(tcur, state)

        # Perform continuous integration using solve_ivp (integrates f over a time span)
        res = scipy.integrate.solve_ivp(f, (tcur, tf), state,
                                        dense_output=True,  # To allow interpolation of the solution
                                        events=[make_scipy_event(event, direction, terminal=True)],  # Handle events
                                        method=method)  # Use the RK45 method for solving the ODE

        # Append the results of the integration
        t_total.append(res.t)  # Add the time points from the current integration
        y_total.append(res.y)  # Add the corresponding state values

        # Check if an event was triggered during this integration step
        if res.status == 1:
            # If the event was triggered, update the state using the jump_map (guard function)
            jump_times.append(res.t_events)  # Record the event (jump) time
            state = guard(res.y[:, -1])  # Update the state using the guard function after the event

            # Check for infinite jumping condition (if jumps occur too frequently)
            if len(jump_times) > 2:
                # If two consecutive jumps are too close in time (less than PROGRESS_TOL), raise an error
                if jump_times[-1][0] - jump_times[-2][0] < PROGRESS_TOL:
                    logging.debug("cur_witness %f, tcur %f", cur_witness, tcur)
                    raise ValueError(
                        "Infinite Jumps! Please reduce the time span so that it's smaller than {}.".format(tcur))

        # If the integration finished without events (status 0), return the results
        elif res.status == 0:
            state = res.sol(tf)  # Get the final state at the final time
            t_total = np.concatenate(t_total)  # Concatenate the accumulated time points
            y_total = np.hstack(y_total)  # Concatenate the state values
            return t_total, jump_times, y_total, state  # Return the complete time, jump times, state trajectory, and final state

        # Update the current time to the end of the last integration step
        tcur = res.t[-1]