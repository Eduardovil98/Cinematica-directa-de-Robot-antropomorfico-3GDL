### Cinemática directa de un Robot antropomórfico 3GDL con Python



##### Lenguajes y Herramientas

<img src='https://ruslanmv.com/assets/images/posts/2022-01-24-How-to-connect-Google-Colab-to-your-computer/colab.png' width='150'> <img src='https://www.python.org/static/community_logos/python-logo-master-v3-TM-flattened.png' width='200'>


##### Introducción
La cinemática directa consiste en determinar cuál es la posición y orientación del extremo final del robot, con respecto a un sistema de coordenadas que se toma como referencia, conocidos los valores de las articulaciones y los parámetros geométricos de los elementos del robot.
- Se utiliza fundamentalmente el álgebra vectorial y matricial para representar y describir la localización de un objeto en el espacio tridimensional con respecto a un sistema de referencia fijo.
- Dado que un se robot puede considerar como una cadena cinemática formada por objetos rígidos o eslabones unidos entre sí mediante articulaciones, se puede establecer un sistema de referencia fijo situado en la base del robot y describir la localización de cada uno de los eslabones con respecto a dicho sistema de referencia.

El presente trabajo aporta un ejemplo de la aplicación de la cinemática directa a un robot de tres grados de libertad con la representación de Denavit-Hartenberg de las matrices de transformación homogénea, se tiene o se conocen los siguientes valores:  el ángulo medido entre los ejes articulares y la distancia entre los ejes articulares, con estas variable podremos programar un algoritmo que nos indique la posición final del robot y mostrarlo de manera gráfica, este proyecto fue desarrollado con el lenguaje de programación Python con IDE de Google Colab.

En la figura se muestran las características y dimensiones del brazo robótico con el cual se obtendrá la cinemática directa para determinar la posición y orientación del elemento terminal referido a la base.
<p align="center">
	<img src='https://scontent.fntr3-1.fna.fbcdn.net/v/t39.30808-6/317782345_2387475848086148_3634494119880566751_n.jpg?_nc_cat=103&ccb=1-7&_nc_sid=0debeb&_nc_eui2=AeFh-T0CGfNJ_QjnJXcg-qPzPu_rvC6HWPk-7-u8LodY-YzbJKdy0MRuccMUYFtTq_85JzXqjEwveUxyoCh8KSTz&_nc_ohc=AsXZFPuJEAoAX-Vx77O&_nc_ht=scontent.fntr3-1.fna&oh=00_AfDb2MYwko-DjCkRcJihsVr8gAV6nryLlCvTLJHZlEc2sw&oe=638DB7F3' width='300'>
</p>


### Programación

##### Paso 1)
Lo primero que realizaremos es sacar la representación de Denavit-Hartenberg con base a la imagen, la tabla queda de la siguiente manera:

<p align="center">
	<img src='https://scontent.fntr3-1.fna.fbcdn.net/v/t39.30808-6/317938354_2387491028084630_4684303261126530196_n.jpg?stp=dst-jpg_s960x960&_nc_cat=100&ccb=1-7&_nc_sid=0debeb&_nc_eui2=AeGTN0Eb-k3pVBpML1BnfQDoSYHqYpEKhi5JgepikQqGLtjb8gj2WI0PTniO4DEt7-aGXtAUQZ1qo5cpHrXjKmI6&_nc_ohc=OKJ54kENkGUAX8CIBNa&tn=p7f7HmXJygRraCre&_nc_ht=scontent.fntr3-1.fna&oh=00_AfBS8aiXdfMnpt7NKgfCYy7-hRkgMz0r1wQoO0siDqcxOg&oe=638E99F7' width='400'>
</p>


##### Paso 2)
Las librerías que usaremos en Python son lo siguiente:

```python
 %pip install sympy==1.10.1
 %pip install matplotlib
 %pip install numpy
```
La librería **sysmpy** nos ayuda a realizar operaciones con símbolos matemáticos, además de poder trabajar con matrices, otra librería muy utilizada en Python es **numpy**, esta librería nos permite crear matrices y realizar operaciones matriciales y finalmente matplotlib el cual nos ayudara a crear gráficos a partir de datos contenidos en lista o matriz.

Una vez instalas las librerías iniciaremos a programar las funciones que nos ayudaran a calcular la cinemática directa del robot.

##### Librerias Usadas:


```python
# importamos las librerías necesarias
import sympy as sp  # librería para cálculo simbólico
import threading    # librería para trabajar funciones en paralelo

```
##### Crear clase para D-H

Esta clase ayudara a realizar las operaciones de forma paralela para calcular la expresión de la matriz de transformación D-H.
```python
# Version 1.0
class InvKin():
    # Version 1.0
    def __init__(self, threads_num):
        """
        Parametros de la clase

        :param threads_num: Numero de subprocesos a ejecutar de forma paralela
        """
        self.threads_num = threads_num
		
    # Version 1.0
    def matrix_mult(self, init, final, row, col, matrixA, MatrixB, MatrixC):
        """
        Funcion para realizar multiplicacion de matrices por bloques
        :param init: Valor inicial del bloqe
        :param final: Valor final del bloque
        :param row: Numero de filas
        :param col: Numero de columnas
        :param matrixA: Valores de la matriz A
        :param MatrixB: Valores de la matriz B
        :param MatrixC: Matriz Nulo (Resultado)
        :return: NULL
        """
        for i in range(init, final):
            for j in range(col):  # Columnas
                result = 0
                for k in range(row):  # Filas
                    result += matrixA[i,k] * MatrixB[k,j]
                MatrixC[i,j] = result
				
    # version 1.0
    def start(self, matrix_A, matrix_B, matrix_C):
        """
        :param matrix_A: Conjunto bidimensional de valores de la matriz A
        :param matrix_B: Conjunto bidimensional de valores de la matriz B
        :param matrix_C: Matriz resultado
        :return: Matriz C
        """
        # print(matrix_A[0, 0])
        # print(matrix_B)
        # print(matrix_C)
        # Numero de filas
        size_rows, size_columns = sp.shape(matrix_A)

        # Numero de bloques
        nbloques = size_rows/self.threads_num

        # Lista para almencenar los hilos creados
        threads = list()

        for i in range(self.threads_num):
            # Asignar el punto o indice inicial y final del bloque
            initial = int(i * nbloques)
            if i <= self.threads_num - 2:
                final = int((i+1)*nbloques)
            else:
                final = int(size_rows)
			# Parametros de la instancia
            x = threading.Thread(target=self.matrix_mult, args=(initial, final, size_rows, size_columns, matrix_A, matrix_B, matrix_C)) 
            threads.append(x) # Agregar hilo a la lista
            x.start() # Iniciar hilo

        # Esperar a que terminen todos los hilos
        for index, thread in enumerate(threads):
            thread.join()

        # Restornar la matriz resultado
        return matrix_C
		
    # Version 2.0
    def symTfromDH(self, theta, d, a, alpha):
        """
      
        Definimos una función para construir las matrices de transformación
        en forma simbolica a partir de los parámetros D-H
        """
        # theta y alpha en radianes
        # d y a en metros
        T = sp.Matrix([[sp.cos(theta), -sp.sin(theta)*sp.cos(alpha), sp.sin(theta)*sp.cos(alpha), a*sp.cos(theta)],
              [sp.sin(theta), sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
              [0, sp.sin(alpha), sp.cos(alpha), d],
              [0, 0, 0, 1]])
        
        return T
```

##### Calcular parámetros de cada eslabón

Con la clase podremos calcular las matrices de Denavit Hartenberg, lo siguiente es definir las variables de los parametros del robot, sustiyuyendo los valores en la tabla de D-H quedaría de la siguiente manera.

<p align="center">
	<img src='https://scontent.fntr3-1.fna.fbcdn.net/v/t39.30808-6/317839370_2387535158080217_1642795907009008970_n.jpg?stp=dst-jpg_s960x960&_nc_cat=109&ccb=1-7&_nc_sid=0debeb&_nc_eui2=AeFJwkwOLBgs-636Ndwyb6EbQviYB-8kZuRC-JgH7yRm5HvQAwuM60pBV5zwrUt04zeeHCb16oMJfrFXgXoXUNGF&_nc_ohc=xJ7ZGo_K-08AX9GP5DI&tn=p7f7HmXJygRraCre&_nc_ht=scontent.fntr3-1.fna&oh=00_AfDsmXVmi5GHApw4-3E8IZ7S97wgQ4alxa4IApmPfN1Z_g&oe=638EBEFA' width='400'>
</p>

En la articulación 1 y 2 asignaremos a theta un ángulo de 90 grados para calcular en que punto en el espacio llegará el robot.

Lo siguiente es definir las variables en Python.


```python
# Valores de la primera articulación

alpha1 = sp.pi/2
d1 = 0.3
a1 = 0
theta1 = sp.pi/2

# Valores de la segunda articulación

alpha2 = 0
d2 = 0
a2 = 0.2
theta2 = sp.pi/2

# Valores de la tercera articulación

alpha3 = 0
d3 = 0
a3 = 0.25
theta3 = 0
```

Crearemos una instancia de la clase **InvKin**

```python
__instance_matrix = InvKin(threads_num=2)
```

Una vez obtenidos todos los parámetros de la tabla de Denavit-Hartenberg se procede a obtener las matrices de transformación, de las cuales se obtendrá la cinemática directa del robot.
La representación de Denavit-Hartenberg es a través del producto de cuatro transformaciones básicas.

<p align="center">
	<img src='https://scontent.fntr3-1.fna.fbcdn.net/v/t39.30808-6/317857926_2388144578019275_5951738726166414594_n.jpg?_nc_cat=100&ccb=1-7&_nc_sid=0debeb&_nc_eui2=AeF7iUc_A7kRjnsjhcSW-2Pj4ZPGGl5GAf7hk8YaXkYB_sZjO_ZkLZ7abUOBX4LIEEc5Mqfl8tVERA0HkT1-mOQr&_nc_ohc=QNVy6NW4jicAX9kdYw3&_nc_ht=scontent.fntr3-1.fna&oh=00_AfA7X12dQWjAF6D8dSbXhM4-ztVae-ldbGfDFd6fwzrtmw&oe=638E57DB' width='400'>
</p>

La función **symTfromDH** que se encuentra dentro de la clase **InvKin** ayuda a calcular la transformación que se muestra en la figura anterior.

Obtendremos la transformación homogénea para la primera articulacion.

```python
T01 = __instance_matrix.symTfromDH(theta1, d1, a1, alpha1)
T01


```
Matriz resultado de la primera articulación.

|     |     |     |       |
|-----|-----|-----|-----|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 |
| 0 | 1 | 0 | 0.3 |
| 0 | 0 | 0 | 1 |

Para la segunda articulación quedaría de la siguiente manera.

```python
T02 = __instance_matrix.symTfromDH(theta2, d2, a2, alpha2)
T02


```

Matriz resultado de la segunda articulación.

|     |     |     |       |
|-----|-----|-----|-----|
| 0 | -1 | 1 | 0 |
| 1 | 0 | 0 | 0.2 |
| 0 | 0 | 1 | 0 |
| 0 | 0 | 0 | 1 |

Por ultimo la tercera articulacion

```python
T03 = __instance_matrix.symTfromDH(theta3, d3, a3, alpha3)
T03


```
Matriz resultado de la tercera articulación.

|     |     |     |       |
|-----|-----|-----|-----|
| 1 | 0 | 0 | 0.25 |
| 0 | 1 | 0 | 0 |
| 0 | 0 | 1 | 0 |
| 0 | 0 | 0 | 1 |

##### Calcular la matriz T
Una vez calculados los parámetros de cada eslabón, se calcula la matriz **T** que indica la localización del sistema final con respecto al sistema de referencia de la base del robot.

$$ T = (T01)(T02)(T03) $$

Como todas la matrices son del mismo tamaño, basta con conocer el tamaño de una matriz, en este caso tomaremos a la variable **T02**

```python
__rows, __columns = sp.shape(T02)

 >> 4x4

```

Crearemos una matriz nula del mismo tamaño donde guardaremos el resultado de las multiplicaciones de las matrices homogéneas.

```python
T_aux = sp.Matrix(__rows, __columns, lambda i,j: 0)
T_aux
```
|     |     |     |       |
|-----|-----|-----|-----|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |

Realizaremos la multiplicación de la matriz **T01** y **T02** con la función **start** que se encuentra en la instancia de la clase creada al inicio del código.

```python
T_aux = __instance_matrix.start(matrix_A=T01, matrix_B=T02, matrix_C=T_aux)
T_aux
```
|     |     |     |       |
|-----|-----|-----|-----|
| 1 | 0 | 0 | 0.2 |
| 0 | 0 | -1 | 0 |
| 0 | 1 | 0 | 0.3 |
| 0 | 0 | 0 | 1 |

Ahora haremos la multiplicación de la matriz resultado **T_aux** y **T02**.

Creamos nuevamente una matriz nula.

```python
T_result = sp.Matrix(__rows, __columns, lambda i,j: 0)
T_result

```
|     |     |     |       |
|-----|-----|-----|-----|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |


```python
T_result = sp.Matrix(__rows, __columns, lambda i,j: 0)
T_result

```
Realizamos la multiplicación de las matrices

```python
T_result = __instance_matrix.start(matrix_A=T_aux, matrix_B=T03, matrix_C=T_result)
T_result

```
|     |     |     |       |
|-----|-----|-----|-----|
| 1 | 0 | 0 | 0.45 |
| 0 | 0 | -1 | 0 |
| 0 | 1 | 0 | 0.3 |
| 0 | 0 | 0 | 1 |

##### Resultado
```
Las coordenadas del punto final del robot son:


```

```python
print("Eje en X:", round(T_result[0,3], 2))
print("Eje en Y:", round(T_result[1,3], 2))
print("Eje en Z:", round(T_result[2,3], 2))

>> Eje en x: 0.45
>> Eje en y: 0.0
>> Eje en z: 0.3
```
```
Al rotar 90º (π/2 rad) a θ1 y θ2.

```
##### Visualización con Matplotlib

Lo primero es importar las librerias en Python

```python
import numpy as np  # librería para matrices
import matplotlib.pyplot as plt # librería para graficar
from mpl_toolkits.mplot3d import proj3d # librería para graficar 3D

```
Lo siguiente es escribir una clase para graficar los resultados, donde solo recibirá de parámetro una lista de las matrices de transformación de D-H.

```python
class Plot():
    # Version 1.0
    def __init__(self, mtxs):
        """
        Parametros de la clase

        :param mtxs: Matrices de transformacion(Resultado)
        """
        self.mtxs = mtxs
        # LLamar funcion para configurar el entorno donde se visualizará el robot
        self.__config_plt()
    def __config_plt(self):
        """
		Inicializar los parametros para graficar en Matplotlib

        :return: NULL
        """
        # configurar el espacio cartesiano donde se visualizará el robot
        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')

        # Limitar las coordenadas en el eje (x,y,z)
        self.ax.set_xlim(-0.2, 0.6)
        self.ax.set_ylim(-0.2, 0.6)
        self.ax.set_zlim(-0.2, 0.6)

        # Mostrar las etiquetas X, Y y Z
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        self.camera()
    def camera(self, azim=80, dist=10, elev=10):
		 """
		Configurar vista del entorno

        :param azim:  Rotacion del entono de la grafica
        :param dist: Distancia para la vista del entono de la grafica       
		:param elev: Altura para la vista del entono de la grafica
        :return: NULL
        """
        # Camara de la gráfica
        self.ax.azim = azim
        self.ax.dist = dist
        self.ax.elev = elev

    def update_position(self, e, labels_and_points):
        """
		Funcion que ayuda a actualizar la posicion de las etiquetas al detectar el evento de moviento en la grafica

        :param e: Detectar evento para actualiza la posicion
        :param labels_and_points: Tupla que contiene las etiquetas y Coordenadas
        :return: NULL
        """
        for label, x, y, z in labels_and_points:
            x2, y2, _ = proj3d.proj_transform(x, y, z, self.ax.get_proj())
            label.xy = x2,y2
            label.update_positions(self.fig.canvas.renderer)
        self.fig.canvas.draw()


    def __plot_text(self, points, labels, s):
        """
		Funcion para agregar texto en la grafica

        :param points:  Coordenadas de las etiquetas
        :param labels: Mensaje a mostrar
        :param s: Estilo
        :return: NULL
        """
        plotlabels = []
        # Coordenadas de etiquetas
        xs, ys, zs = np.split(points, 3, axis=1)

        # sc = self.ax.scatter(xs,ys,zs)

        for txt, x, y, z in zip(labels, xs, ys, zs):
            # Trasformacion de projeccion con respecto a la vista
            x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax.get_proj())

            # Estilo para mostrar texto
            if s:
                label = plt.annotate(
                    txt, xy = (x2, y2), xytext = (-10, 0),
                    textcoords = 'offset points', ha = 'right', va = 'bottom', fontsize=8, color="black")
            else:
                label = plt.annotate(
                    txt, xy=(x2, y2), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'), fontsize=8)
            # Agregar etiquetasa las lista
            plotlabels.append(label)

        # Actualzar posicion de las etiquetas con base a la visualizacion de la grafica
        self.fig.canvas.mpl_connect('motion_notify_event',
                               lambda event: self.update_position(event, zip(plotlabels, xs, ys, zs)))
    def start(self, theta1, theta2, theta3):
        """
		Mostrar grafica'
		
        :param theta1: Angulo de la primera union
        :param theta2: Angulo de la segunda union
        :param theta3: Angulo de la tercera union
        :return:
        """
        axis = [[0], [0], [0]]
        for j in range(3):
            for i, matx in enumerate(self.mtxs):
                axis[j].extend([matx[j, 3]])
        print(axis)

        self.ax.plot(axis[0], axis[1], axis[2], 'o-', markersize=6, markerfacecolor="red",
                linewidth=2, color="blue")

        self.ax.plot([axis[0][-1]], [axis[1][-1]], [axis[2][-1]], 'o-', markersize=8, markerfacecolor="white")


        points = np.array([(axis[0][-1], axis[1][-1], axis[2][-1])])
        labels = ['Punto final del robot ({}, {}, {})'.format(round(axis[0][-1], 2), round(axis[1][-1], 2), round(axis[2][-1], 2))]
        self.__plot_text(points=points, labels=labels, s=False)


        points = np.array([(axis[0][0], axis[1][0], axis[2][0]), (axis[0][1], axis[1][1], axis[2][1]), (axis[0][2], axis[1][2], axis[2][2])])
        labels = ['$\\theta$={}'.format(theta1), '$\\theta$={}'.format(theta2), '$\\theta$={}'.format(theta3)]
        self.__plot_text(points=points, labels=labels, s=False)

        plt.show()

```

Creamos la instancia de la clase **Plot** pasando las matrices como parámetro, y mostramos la grafica con la función **start**

```python
# Instancia
plot1 = Plot([T01, T_aux, T_result])
# Rotacion de la posicion para la camara
plot1.camera(80)
# Ejecutar grafica
plot1.start(theta1, theta2, theta3)
```

##### Visualizar resultados

###### Test 1
```python
Con valores de:
>> θ1 = 90
>> θ2 = 90
>> θ3 = 0
```
<p align="center">
	<img src='https://scontent.fntr3-1.fna.fbcdn.net/v/t39.30808-6/317921731_2388241728009560_7345649208629501699_n.jpg?_nc_cat=103&ccb=1-7&_nc_sid=0debeb&_nc_eui2=AeGU3sIyeMx8p9GOt6A8kZwW3GGrGX3nuVXcYasZfee5VcwUTduCBGL3e3QAZOo2SZBGtSql5Z7trHWwTHH1pLZa&_nc_ohc=WT1PTPWUiFkAX_6xBmp&_nc_ht=scontent.fntr3-1.fna&oh=00_AfBoF0YXVkuFIFznf5hC4DUBj4kL2ZdcjbHyusR8N06MKQ&oe=638FE9C3' width='400'>
</p>

```python
>> Eje en x: 0.45
>> Eje en y: 0.0
>> Eje en z: 0.3
```

###### Test 2
```python
Con valores de:
>> θ1 = 90
>> θ2 = 0
>> θ3 = 45
```
<p align="center">
	<img src='https://scontent.fntr3-1.fna.fbcdn.net/v/t39.30808-6/318199688_2388250764675323_2400862548932717296_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=0debeb&_nc_eui2=AeHTrf1R9JVBj2N2IcqC2UT5p2B2zoLsYJinYHbOguxgmBoLGa7vQcch2yVBS4svnSLVE748rz379-ouiEHJtVlv&_nc_ohc=xG3NZ0RA8O4AX9XdOKG&_nc_ht=scontent.fntr3-1.fna&oh=00_AfBcNT7VT8HizZHKwH8eGBLAY9c9y1kkucBFNd7l1DUutA&oe=638EA2E6' width='400'>
</p>

```python
>> Eje en x: 0
>> Eje en y: 0.38
>> Eje en z: 0.48
```

###### Test 3
```python
Con valores de:
>> θ1 = 0
>> θ2 = 0
>> θ3 = 0
```
<p align="center">
	<img src='https://scontent.fntr3-1.fna.fbcdn.net/v/t39.30808-6/317844283_2388240801342986_9205098720764980694_n.jpg?_nc_cat=104&ccb=1-7&_nc_sid=0debeb&_nc_eui2=AeG2Qs8lnHcMlkavRqwsK9b7daWaAssyCWp1pZoCyzIJag2YfMjkCriOBOHDuq8NxP36yfFh4jJg6JROyqVDzT6j&_nc_ohc=5_Ixxtod_-AAX_Y2rL2&_nc_ht=scontent.fntr3-1.fna&oh=00_AfDOyKj0AL2H6JN30J9y--kA747QC-pA7uaKCWVWiKMWkg&oe=638FF313' width='400'>
</p>

```python
>> Eje en x: 0.45
>> Eje en y: 0
>> Eje en z: 0.3
```

A lo largo del desarrollo del proyecto se lograron diferentes objetivos, como fue el iniciar con la creación de funciones en Python que sean lo suficientemente robustas para obtener los resultados de la cinemática directa del robot de 3GDL mediante matrices. Los resultados finales fueron satisfactorios de acuerdo con los objetivos planteados.
Se realizaron diferentes casos para validar que las funciones realizaban las operaciones de forma correcta, las pruebas fueron satisfactorios generando buenos resultados.

##### Algunas referencias

```python
https://ocw.ehu.eus/pluginfile.php/50445/mod_resource/content/8/T5%20CINEMATICA%20OCW_Revision.pdf
https://www.youtube.com/watch?v=V_IIeLJzR44
http://motion.cs.illinois.edu/RoboticSystems/CoordinateTransformations.html
https://stackoverflow.com/questions/48265646/rotation-of-a-vector-python
https://www.youtube.com/watch?v=7aTGEjDmvv4
https://www.kramirez.net/Robotica/Material/Presentaciones/CinematicaDirectaRobot.pdf
https://nbio.umh.es/files/2012/04/practica2.pdf
https://github.com/aakieu/3-dof-planar
https://dancasas.github.io/teaching/AC-2019/docs/2.1-Cinematica-directa-v2019.pdf
```
