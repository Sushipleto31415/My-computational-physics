import numpy as np
import matplotlib.pyplot as plt
from Modules.findroot import newton_raphson

#Se definen los parámetros constantes de la función
p0 = 101325    # Presión atmosférica [Pa]
R = 287         # Constante del gas [J/(kg·K)]
T0 = 300        # Temperatura [K]
g = 9.81        # Gravedad [m/s²]
a = 1.5         # Altura de la puerta [m]
b = 1.0         # Ancho de la puerta [m]
h = 2.0         # Altura del líquido sobre la puerta [m]
Fp = 1000       # Fuerza de la persona [N]
k= g / (R*T0)


#Definimos la función
def f(H):
    term1 = p0 * (a-H)/2 -Fp
    term2 = (p0 * R * T0)/ (2*g) * np.exp(k*h) * (np .exp(k * (a - H))-1)
    return term1 + term2

def df(H):
    return -p0/2 - (p0)/2 * np.exp(k * (h+a - H))

#Se aplica el método de Newton-Raphson
H_0 = 0.5 #Altura inicial, un valor entre 0 y a
H_solucion = newton_raphson(f,df,0.5,10, tol=1e-6  ,max_iter=100)

print(f"Altura necesaria H:{H_solucion} m")
