import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Modules.ode import runge_kutta_4th
from matplotlib.animation import FuncAnimation
plt.style.use('dark_background')

#Define time and time step
dt = 0.05  # Time step
iteraciones=int(1e5)
t = np.linspace(0, iteraciones, iteraciones)  # Defining the time array


#Define the aisawa atractor ODE system
def aisawa_ode(u,a,b,c,d,e,f):
    """
    a, b, c, d, f: Are parameters in the aisawa atractor
    u: Is the state vector of the system
    """
    x, y, z= u
    dx = (z-b)*x-d*y
    dy=d*x + (z - b)*y
    dz= c + a*z - (z**3)/3 - (x**2+y**2)*(1+e*z) + f*z*x**3

    return np.array([dx,dy,dz])

u0=np.array([0.1,0,0])

results=runge_kutta_4th(lambda u,
                         t: aisawa_ode(u,0.95,0.7,0.65,3.5,0.25,0.1),
                         u0,
                             t,
                                dt)
ax=plt.axes(projection="3d")
ax.plot3D(results[:,0],results[:,1],results[:,2],color="purple",lw=0.8, )
ax.set_xlabel('$x$ axis', labelpad=10)
ax.set_ylabel('$y$ axis', labelpad=10)
ax.set_zlabel('$z$ axis', labelpad=10)
ax.set_title('Atractor de Aisawa')

plt.savefig("aisawa_atractor.png")
plt.show()
