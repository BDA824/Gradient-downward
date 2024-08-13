import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calcular_y(x1, x2):
    term1 = np.sin(x1 + x2)
    term2 = (x1 - x2) ** 2
    term3 = -1.5 * x1
    term4 = 2.5 * x2
    y = term1 + term2 + term3 + term4 + 1
    return y

# Se define el rango de valores para las variables x1 y x2
x1 = np.linspace(-5, 5, 400)
x2 = np.linspace(-5, 5, 400)
x1, x2 = np.meshgrid(x1, x2)

# Cálculo de y para cada par (x1, x2)
y = calcular_y(x1, x2)

# Graficación en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, y, cmap='viridis')

# Etiquetas y título
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('McCormick Function')

plt.show()


#McCormick Function

def calculo_gradiente(x1, x2):
  dy_dx1 = np.cos(x1 + x2) * (x2 - 2) - 1.5
  dy_dx2 = np.cos(x1 + x2) * (x1 - x2) + 2.5
  return dy_dx1, dy_dx2

def gradiente_descendente(x1_inicial, x2_inicial, tasa_aprendizaje, iteraciones):
  x1 = x1_inicial
  x2 = x2_inicial
  for i in range(iteraciones):
    dy_dx1, dy_dx2 = calculo_gradiente(x1, x2)
    x1 -= tasa_aprendizaje * dy_dx1
    x2 -= tasa_aprendizaje * dy_dx2

    norma = np.sqrt(dy_dx1**2 + dy_dx2**2)
    print(f"Norma del gradiente: {norma}")
    print(f"Iteración {i}: x1 = {x1}, x2 = {x2}");
  return x1, x2

def calcular_y(x1, x2):
    term1 = np.sin(x1 + x2)
    term2 = (x1 - x2) ** 2
    term3 = -1.5 * x1
    term4 = 2.5 * x2
    y = term1 + term2 + term3 + term4 + 1
    return y

print(gradiente_descendente(-1.5, 2.1, 0.01, 20));