import numpy as np
#Método de la bisección
def biseccion(f,a,b,max_iter=100,*args,**kargs):
    """
    Encuentra la raíz de una función usando el método de La Bisección.

    Args:
        f: La función para la cual encontrar la raíz.
        a: El límite inferior del intervalo de búsqueda.
        b: El límite superior del intervalo de búsqueda.
        tol: La tolerancia para la convergencia.
        max_iter: El número máximo de iteraciones.
        *args: Argumentos adicionales para pasar a la función f.
        **kwargs: Argumentos de palabra clave adicionales para pasar a la función f.

    Returns:
        Le intervalo de la raíz aproximada de la función, o None si el método no converge.
    """
    x=np.linspace(a,b,N)

    F=f(x,*args,**kargs)

    dF=F[1:]*F[:-1] #Producto elemento por elemento F[:-1] del 0 al penultimo
    i=np.where(dF<0)

    return x[i]


#Método de la secante
def secante(f, a, b, tol=1e-3, *args, **kwargs):
    """
    Encuentra la raíz de una función usando el método de la secante.

    Args:
        f: La función para la cual encontrar la raíz.
        a: El límite inferior del intervalo de búsqueda.
        b: El límite superior del intervalo de búsqueda.
        tol: La tolerancia para la convergencia.
        *args: Argumentos adicionales para pasar a la función f.
        **kwargs: Argumentos de palabra clave adicionales para pasar a la función f.

    Returns:
        La raíz aproximada de la función, o None si el método no converge.
    """
    g = lambda x: f(x, *args, **kwargs)

    x0, x1 = a, b

    while np.abs(x1 - x0) > tol:
        if g(x1) - g(x0) == 0:
            print("The values of g(x1) and g(x0) are equal, this would cause ZeroDivisionError")
            return None

        x2 = x0 - g(x0) * (x1 - x0) / (g(x1) - g(x0))
        x0, x1 = x1, x2

    print("The method of the secand method converged to the root :{x2}")
    return x2


#Método de Newton-Raphson
def newton_raphson(f, f_prime, a, b, tol=1e-6, max_iter=1000, *args, **kwargs):
    """
    Encuentra la raíz de una función usando el método de Newton-Raphson.

    Args:
        f: La función para la cual encontrar la raíz.
        f_prime: La derivada de la función f.
        a: El límite inferior del intervalo de búsqueda.
        b: El límite superior del intervalo de búsqueda.
        tol: La tolerancia para la convergencia.
        max_iter: El número máximo de iteraciones.
        *args: Argumentos adicionales para pasar a la función f.
        **kwargs: Argumentos de palabra clave adicionales para pasar a la función f.

    Returns:
        La raíz aproximada de la función, o None si el método no converge.
    """

    r = (a + b) / 2  # Se toma como punto de inicio el punto medio entre a y b

    g = lambda x: f(x, *args, **kwargs)
    g_prime = lambda x: f_prime(x, *args, **kwargs)

    for i in range(max_iter):
        f_r = g(r)
        f_prime_r = g_prime(r)

        if f_prime_r == 0:
            print("Derivada cero. El método de Newton-Raphson falla.")
            return None

        r_next = r - f_r / f_prime_r

        if abs(r_next - r) < tol:
            print(f"Newton-Raphson method converged to root: {r_next:.6f} in {i + 1} iterations")  # Print result
            return r_next

        r = r_next

    print("El método de Newton-Raphson no convergió después de", max_iter, "iteraciones.")
    return None