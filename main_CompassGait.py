import numpy as np
from dynamics import BouncingBall, CompassGait
from integrators import hybrid_integrator
import matplotlib.pyplot as plt

# Main execution block
if __name__ == '__main__':
    # Initialize the Compass Gait robot system with a slope gamma of 0.05.
    # The CompassGait represents a simplified bipedal walking model.
    robot = CompassGait(gamma=0.05)

    # Define the event function to detect when the robot's foot strikes the ground.
    # The event occurs when the state's relevant component crosses zero.
    event = lambda t, state: robot.collision_witness(state)
    direction = -1  # Detect the event when crossing from positive to negative (downward)

    # Initial state of the system: [stance leg angle, swing leg angle, stance leg velocity, swing leg velocity]
    state = np.array([-0.323389, 0.218669, -0.377377, -1.092386], dtype=np.float64)

    # Simulation parameters: number of time steps and time increment (dt).
    num_time_steps = 4000
    tick = 0
    dt = 0.005
    T = []  # Store total time points
    Y = []  # Store total state results
    Jump_times = []  # Store times when jumps (events) occur

    # Simulation loop
    for i in range(num_time_steps):
        t0 = tick * dt  # Start time for this step
        tf = (tick + 1) * dt  # End time for this step
        u = 0  # Control input (not used here for the compass gait)

        # Integrate the continuous dynamics and handle discrete jumps
        t_total, jump_times, y_total, state = hybrid_integrator(lambda t, state: robot.flow_map(state, u),
                                                                event,
                                                                direction,
                                                                robot.jump_map,
                                                                state,
                                                                t0,
                                                                tf)
        tick += 1  # Increment the time step

        # Store the time and state results from this step
        T.append(t_total)
        Y.append(y_total)

        # If a jump (event) occurred, store the jump time
        if jump_times:
            Jump_times.append(jump_times)

    # Convert the lists of time and state results to numpy arrays for easier manipulation
    T = np.concatenate(T)
    Y = np.hstack(Y)

    #### Enhanced Visualization ####

    # Create a new figure for the plot
    plt.figure(figsize=(10, 6))

    # Plot the state of the robot over time.
    # Y[0] and Y[1] represent the angles of the stance and swing legs, respectively.
    # Y[2] and Y[3] represent the angular velocities of the stance and swing legs.

    # Plot the angular positions (stance leg vs. swing leg)
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
    plt.plot(T, Y[0], label="Stance Leg Angle")
    plt.plot(T, Y[1], label="Swing Leg Angle")
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Angular Positions of Stance and Swing Legs')
    plt.legend()

    # Plot the angular velocities (stance leg vs. swing leg)
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
    plt.plot(T, Y[2], label="Stance Leg Velocity")
    plt.plot(T, Y[3], label="Swing Leg Velocity")
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocities of Stance and Swing Legs')
    plt.legend()

    # Automatically adjust the subplot layout for better visibility
    plt.tight_layout()

    # Display the plot
    plt.show()