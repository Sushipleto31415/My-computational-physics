import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Modules.ode import runge_kutta_4th  # Asegúrate de que esta función esté correcta

# Constants
G = 1  # Gravitational constant
m1 = 10.0  # Mass of body 1
m2 = 20.0  # Mass of body 2
m3 = 30.0  # Mass of body 3
dt = 0.05  # Time step
t = np.linspace(0, 100, int(100/dt))  # Simulation time
d = 2  # Distance between bodies

# Initial conditions for the three bodies (positions and velocities)
# Posiciones en un triángulo equilátero
initial_conditions = np.array([-d, 0, 0, -1,   # Cuerpo 1
                                d, 0, 0, 1,    # Cuerpo 2
                                0, d * np.sqrt(3)/2, 0, 0])  # Cuerpo 3

# Function for the three-body problem
def three_body(u, m1, m2, m3, G=G):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = u

    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    # Aceleraciones debido a la gravedad
    ax1 = -G * m2 * (x1 - x2) / r12**3 - G * m3 * (x1 - x3) / r13**3
    ay1 = -G * m2 * (y1 - y2) / r12**3 - G * m3 * (y1 - y3) / r13**3

    ax2 = -G * m1 * (x2 - x1) / r12**3 - G * m3 * (x2 - x3) / r23**3
    ay2 = -G * m1 * (y2 - y1) / r12**3 - G * m3 * (y2 - y3) / r23**3

    ax3 = -G * m1 * (x3 - x1) / r13**3 - G * m2 * (x3 - x2) / r23**3
    ay3 = -G * m1 * (y3 - y1) / r13**3 - G * m2 * (y3 - y2) / r23**3

    # Devuelve las derivadas
    return np.array([vx1, vy1, ax1, ay1, vx2, vy2, ax2, ay2, vx3, vy3, ax3, ay3])

# Use the Runge-Kutta method to get the results for the initial conditions
result = runge_kutta_4th(lambda u, t: three_body(u, m1, m2, m3, G), initial_conditions, t, dt)

# Set up the figure for the animation
plt.style.use("dark_background")
fig, ax = plt.subplots()
ax.set_xlim(-2*d, 2*d)  # Set x-limits for the plot
ax.set_ylim(-2*d, 2*d)  # Set y-limits for the plot

# Create lines for each set of masses
lines = [ax.plot([], [], 'o', lw=2)[0] for _ in range(3)]  # Tres cuerpos

# Initialize the background for the animation
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Update function for the animation
def update(frame):
    x1, y1 = result[frame, 0], result[frame, 1]  # Cuerpo 1
    x2, y2 = result[frame, 4], result[frame, 5]  # Cuerpo 2
    x3, y3 = result[frame, 8], result[frame, 9]  # Cuerpo 3

    # Set the line data (position of the three bodies)
    lines[0].set_data(x1, y1)
    lines[1].set_data(x2, y2)
    lines[2].set_data(x3, y3)
    return lines

# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=dt * 1000)
ani.save("three_body.gif",fps=60)
# Show the plot with the animation
plt.show()
