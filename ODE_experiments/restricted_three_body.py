import numpy as np
import matplotlib.pyplot as plt
from Modules.ode import runge_kutta_4th  # Asegúrate de que esta función esté correcta

# Constants
G = 1.0  # Gravitational constant
m1 = 1.0  # Mass of body 1
m2 = 1.0  # Mass of body 2
m3 = 10.0  # Mass of body 3
dt = 0.05  # Time step
t = np.linspace(0, 30, int(30/dt))  # Simulation time

# Function for the three-body problem
def three_body(u, m1, m2, m3, G):
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

# Parámetros para el mapa de calor
x_range = np.linspace(-10, 10, 50)  # Rango de posición inicial para x
y_range = np.linspace(-10, 10, 50)  # Rango de posición inicial para y
stability_map = np.zeros((len(x_range), len(y_range)))  # Mapa de estabilidad

# Iterar sobre las condiciones iniciales
for i, x1 in enumerate(x_range):
    for j, y1 in enumerate(y_range):
        # Definir las condiciones iniciales
        initial_conditions = np.array([x1, y1, 0, 1,   # Cuerpo 1
                                       -x1, -y1, 0, -1,  # Cuerpo 2
                                       0, 0, 0, 0])  # Cuerpo 3

        # Simular el sistema
        result = runge_kutta_4th(lambda u, t: three_body(u, m1, m2, m3, G), initial_conditions, t, dt)

        # Calcular un criterio de estabilidad (ejemplo: distancia entre el cuerpo 1 y 2)
        distance = np.sqrt((result[:, 0] - result[:, 4])**2 + (result[:, 1] - result[:, 5])**2)
        if np.all(distance < 0.1):  # Si la distancia permanece bajo un umbral, es estable
            stability_map[i, j] = 1  # Asignar un valor a la estabilidad

# Visualizar el mapa de calor
plt.imshow(stability_map.T, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Estabilidad (1=Estable, 0=Inestable)')
plt.title('Mapa de Estabilidad del Problema de Tres Cuerpos')
plt.xlabel('Posición Inicial x del Cuerpo 1')
plt.ylabel('Posición Inicial y del Cuerpo 1')
plt.show()
