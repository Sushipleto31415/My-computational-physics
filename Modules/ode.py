import numpy as np

#This module is for solving ODEs numerically under call

#4th grade Runge-Kutta method

def runge_kutta_4th(f, u0, t, dt, *args, **kwargs):
    """
    Resuelve una ecuación diferencial ordinaria (EDO) utilizando el método de Runge-Kutta de cuarto orden.

    Args:
        f: La función que define la EDO, de la forma f(u, t, *args, **kwargs).
        u0: El valor inicial de la variable dependiente (u) en el tiempo t[0].
        t: Un array de NumPy que representa los puntos de tiempo en los que se desea calcular la solución.
        dt: El paso de tiempo.
        *args: Argumentos adicionales para pasar a la función f.
        **kwargs: Argumentos de palabra clave adicionales para pasar a la función f.

    Returns:
        Un array de NumPy que contiene la solución aproximada de la EDO en los puntos de tiempo especificados en t.
    """

    N = len(t)

    u = np.zeros([N, len(u0)]) # Initialize u as a 2D array

    u[0] = u0 # Set the initial condition

    for n in range(N - 1):
        k1 = f(u[n], t[n], *args, **kwargs)
        k2 = f(u[n] + 0.5 * dt * k1, t[n] + 0.5 * dt, *args, **kwargs)
        k3 = f(u[n] + 0.5 * dt * k2, t[n] + 0.5 * dt, *args, **kwargs)
        k4 = f(u[n] + dt * k3, t[n] + dt, *args, **kwargs)

        u[n + 1] = u[n] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return u
