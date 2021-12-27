# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 11:23:11 2021

@author: Мезенцева Ксения
"""

from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt
import scipy.constants as constants

r = 1.496e11 #km (1000 a.e)
vt = 3 * 1000 #m/s # скорость тела на расстоянии r от Солнц
G = constants.G #гравитационная постоянная
m1 = 1.98847e30 #kg #масса Солнца
m2 = 2.2e14 #kg #масса тела

# вектор-функция правых частей уравнений:
# полагается, что f зависит от (q, t), причём q - это
# список из двух чисел:
# q = [v, x, w, y], t - время
def f(q, t):
    
     x = q[0]
     v = q[1]
     y = q[2]
     w = q[3]
     
     fx = - G * (m1 + m2) * x / (x**2 + y**2)**1.5#часть уравнения для координаты х
     fy =  - G * (m1 + m2) * y / (y**2 + x**2)**1.5#часть уравения для координаты у
     f1x = v#dx/dt
     f1y = w#dy/dt
     return [f1x, fx, f1y, fy]
 
    #НУ
# массив точек интегрирования
ti = np.linspace(0, 100000000, 100)
         
# начальная координата х
x0 = r
# скорость тела в точке х=0
v0 = vt
# начальная координата у
y0 = 0
# скорость тела в точке у=0
w0 = vt
# список начальных условий
t0 = [x0, v0, y0, w0 ]


# решение ОДУ
sol1 = integrate.odeint(f, t0, ti)

x1 = sol1[:, 0]
y1 = sol1[:, 2]

#построение траектории движения тела
fig = plt.figure(figsize=(8,5))
plt.plot(x1, y1, 0, 0, 'ro')
plt.ylabel('y')
plt.xlabel('x')
plt.grid(True) #сетка на рисунке
plt.show()
