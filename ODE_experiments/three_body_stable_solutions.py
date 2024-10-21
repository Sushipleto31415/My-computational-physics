import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Modules.ode import runge_kutta_4th
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')

# Constants
G = 1.0  # Gravitational constant
m1 = 10.0  # Mass of body 1
m2 = 20.0  # Mass of body 2
m3 = 30.0  # Mass of body 3
dt = 0.01  # Time step
N=20000 #Número de iteraciones
t = np.linspace(0, N, int(N*dt))  # Simulation time

# Initial conditions for the three bodies (positions and velocities)
initial_conditions = np.array([-10, 10, -3, 0, 0, 0,  # Body 1
                               0, 0, 0, 0, 0, 0,     # Body 2
                               10, 10, 3, 0, 0, 0])  # Body 3

# Function for the three-body problem
def three_body(u, m1, m2, m3, G=G):
    x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2, x3, y3, z3, vx3, vy3, vz3 = u

    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2 + (z3 - z1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2 + (z3 - z2)**2)

    # Gravitational accelerations
    ax1 = -G * m2 * (x1 - x2) / r12**3 - G * m3 * (x1 - x3) / r13**3
    ay1 = -G * m2 * (y1 - y2) / r12**3 - G * m3 * (y1 - y3) / r13**3
    az1 = -G * m2 * (z1 - z2) / r12**3 - G * m3 * (z1 - z3) / r13**3

    ax2 = -G * m1 * (x2 - x1) / r12**3 - G * m3 * (x2 - x3) / r23**3
    ay2 = -G * m1 * (y2 - y1) / r12**3 - G * m3 * (y2 - y3) / r23**3
    az2 = -G * m1 * (z2 - z1) / r12**3 - G * m3 * (z2 - z3) / r23**3

    ax3 = -G * m1 * (x3 - x1) / r13**3 - G * m2 * (x3 - x2) / r23**3
    ay3 = -G * m1 * (y3 - y1) / r13**3 - G * m2 * (y3 - y2) / r23**3
    az3 = -G * m1 * (z3 - z1) / r13**3 - G * m2 * (z3 - z2) / r23**3

    return np.array([vx1, vy1, vz1, ax1, ay1, az1, vx2, vy2, vz2, ax2, ay2, az2, vx3, vy3, vz3, ax3, ay3, az3])


# Get the result for the initial conditions
result = runge_kutta_4th(lambda u, t: three_body(u, m1, m2, m3, G), initial_conditions, t, dt)

# 3D Plot setup
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

"""
# Plot the initial points for the bodies
body1, = ax.plot([], [], [], 'o', label="Body 1", color='r')
body2, = ax.plot([], [], [], 'o', label="Body 2", color='g')
body3, = ax.plot([], [], [], 'o', label="Body 3", color='b')

# Trajectories
traj1, = ax.plot([], [], [], color='r', lw=0.5)
traj2, = ax.plot([], [], [], color='g', lw=0.5)
traj3, = ax.plot([], [], [], color='b', lw=0.5)

# Set the axis limits

ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-20, 20])

# Personalizar el color de los paneles de los ejes
ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Initialization function for animation
def init():
    body1.set_data([], [])
    body1.set_3d_properties([])
    body2.set_data([], [])
    body2.set_3d_properties([])
    body3.set_data([], [])
    body3.set_3d_properties([])

    traj1.set_data([], [])
    traj1.set_3d_properties([])
    traj2.set_data([], [])
    traj2.set_3d_properties([])
    traj3.set_data([], [])
    traj3.set_3d_properties([])

    return body1, body2, body3, traj1, traj2, traj3

# Update function for animation
def update(frame):
    # Update body positions
    body1.set_data(result[frame, 0], result[frame, 1])
    body1.set_3d_properties(result[frame, 2])

    body2.set_data(result[frame, 6], result[frame, 7])
    body2.set_3d_properties(result[frame, 8])

    body3.set_data(result[frame, 12], result[frame, 13])
    body3.set_3d_properties(result[frame, 14])

    # Update trajectories
    traj1.set_data(result[:frame, 0], result[:frame, 1])
    traj1.set_3d_properties(result[:frame, 2])

    traj2.set_data(result[:frame, 6], result[:frame, 7])
    traj2.set_3d_properties(result[:frame, 8])

    traj3.set_data(result[:frame, 12], result[:frame, 13])
    traj3.set_3d_properties(result[:frame, 14])

    return body1, body2, body3, traj1, traj2, traj3

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=1000/60, blit=True)

# Show the animation
"""
# Plot the trajectories
ax.plot(result[:, 0], result[:, 1], result[:, 2], label="Body 1")
ax.plot(result[:, 6], result[:, 7], result[:, 8], label="Body 2")
ax.plot(result[:, 12], result[:, 13], result[:, 14], label="Body 3")

# Desactivar las marcas de los ejes, pero mantener los ejes visibles
plt.axis('on')


# Personalizar el color de los paneles de los ejes
ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

# Labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
