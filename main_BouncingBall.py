import numpy as np
from dynamics import BouncingBall, CompassGait
from integrators import hybrid_integrator
import matplotlib.pyplot as plt

# Main execution block
if __name__ == '__main__':
    # Initialize a bouncing ball system with a coefficient of restitution (lambda_) of 0.9
    Ball = BouncingBall(lambda_=0.9)

    # Define the event function to detect collisions (when the ball hits the ground)
    event = lambda t, state: Ball.collision_witness(state)
    direction = -1  # Detect the event when the state crosses from positive to negative (falling)

    # Initial state: [position, velocity]
    state = np.array([3, 0], dtype=np.float64)

    # Simulation parameters: number of time steps, time increment, etc.
    num_time_steps = 2500
    tick = 0
    dt = 0.005
    T = []  # Store total time points
    Y = []  # Store total state results
    Jump_times = []  # Store times when jumps (collisions) happen

    # Loop over the defined number of time steps
    for i in range(num_time_steps):
        t0 = tick * dt  # Start time for this step
        tf = (tick + 1) * dt  # End time for this step
        u = 0  # Control input (not used for the bouncing ball)

        # Call the hybrid integrator to simulate the dynamics between t0 and tf
        t_total, jump_times, y_total, state = hybrid_integrator(lambda t, state: Ball.flow_map(state, u),
                                                                event,
                                                                direction,
                                                                Ball.jump_map,
                                                                state,
                                                                t0,
                                                                tf)
        # Advance time step
        tick += 1

        # Store time and state results
        T.append(t_total)
        Y.append(y_total)

        # Store jump (event) times if they occurred
        if jump_times:
            Jump_times.append(jump_times)

    # Convert lists to numpy arrays for easier processing
    T = np.concatenate(T)
    Y = np.hstack(Y)

    #### 3D Visualization of the Bouncing Ball

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract time, position, and initialize jump counts
    t_vals = T
    pos_vals = Y[0]  # Position of the ball
    jump_counts = np.zeros_like(t_vals)  # Initialize jump counts to zero

    # For each jump event, increment the jump count for subsequent time points
    for i in range(len(Jump_times)):
        jump_counts[T > Jump_times[i][0][0]] += 1

    # Plot 3D graph of position over time with jump count
    ax.plot(t_vals, jump_counts, pos_vals, label="Position over Jumps")

    # Add labels and a title to the plot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Jump Count (j)')
    ax.set_zlabel('Position (x1)')
    ax.set_title('Bouncing Ball 3D Simulation')

    # Display the legend and grid for the plot
    ax.legend()
    plt.grid(True)

    # Show the 3D plot
    plt.show()