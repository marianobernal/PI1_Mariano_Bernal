# Readme - Proyecto Integrador 1 


<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

##  <h2 align=center>**Mariano Bernal / Data Science FT - 13**

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

A continuación les compartiré un breve resumen de los principales puntos que se fueron trabajando en el proyecto.  

<hr>  

## **Descripción del problema**



## Rol a desarrollar

Comencé a trabajar como **`Data Scientist`** en Steam, una plataforma multinacional de videojuegos, y vamos a crear un primer modelo de ML que soluciona un problema de negocio: Steam pide que te encargues de predecir el precio de un videojuego. 

Al momento de analizar los sus datos vemos que la madurez de los mismos es poca: Datos anidados, sin transformar, no hay procesos automatizados para la actualización de nuevos productos, entre otras cosas….  haciendo tu trabajo imposible :weary: . 

El objetivo será primero haciendo un trabajo rápido de **`Data Engineer`** , lograr tener un **`MVP`** (_Minimum Viable Product_) para el cierre del proyecto! 
<hr> 

## **Etapas del Proyecto**
A continuación, pasaré a compartirles un breve resumen de las principales tareas realizadas.  
Lo primero a destacar y que podrán notar en el readme es que hemos dividido el trabajo en 3 etapas distintas y para las que podrán encontrar más documentación junto con el paso a paso de las operaciones realizadas.  
<br>
## 1-Transformaciones para funciones 
<br>
En esta etapa con el objetivo de disponibilizar la información para las consultas que se nos solicitaban, comenzamos con la extracción de la informaicón para luego pasar a las transformaciones a realizar con el objetivo de limpiar los datos.<br> Investigando, notamos que la información se nos entregaba en un formato json con información anidada. (La información anidada es una forma de estructurar datos en la que se utilizan objetos o elementos dentro de otros objetos o elementos)  
La mejor manera que encontramos para extraer dicha información fue con el función 'ast.literal_eval' recorriendo con un for por línea de archivo. Esta información, procedimos a cargarla en un dataframe para analizarla y sobre ella luego aplicar las transformaciones y funciones necesarias en su gran mayoría con Pandas.<br>


Una vez que logramos esto, comenzamos a realizar la limpieza. Las principales tareas realizadas fueron:  <br>
+ Transformaciones sobre relesase date: realizamos correcciones en los formatos en que estaba cargada la información para cambiar a tipo de dato fecha <br>
+ Limpieza de nan: reemplazamos los nan que se encontraban como strings para que pandas los tomer como nans correctamente<br>
+ Cambios de tipos de datos: realizamos varios cambios de tipos de datos, en algunos casos (sobre todo los numéricos y de fechas) forzando con coerce<br>
+ Faltantes en género: analizando, notamos que la columna tags contenía muchas veces los géneros. Por la tanto, aprovechamos esta información para completar los registros nulos de géneros con la información de la columna tags (solo trayendo la información que ya estaba en la columna género como géneros<br>
+ Faltantes en Título: operación similar a la anterior pero más directa, donde trajimos info de app_name para completar los faltantes en título<br>

Por último, en esta etapa realizamos el desarrollo de las funciones con las que disponibilizaremos esta información en endpoints de una API desarrollada con FastApi y deployada con Render. Las funciones son:<br>

+ def **genero( *`Año`: str* )**:
    Se ingresa un año y devuelve una lista con los 5 géneros más ofrecidos en el orden correspondiente.

+ def **juegos( *`Año`: str* )**:
    Se ingresa un año y devuelve una lista con los juegos lanzados en el año.

+ def **specs( *`Año`: str* )**:
    Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente. 

+ def **earlyacces( *`Año`: str* )**:
    Cantidad de juegos lanzados en un año con early access.

+ def **sentiment( *`Año`: str* )**:
    Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento. 


+ def **metascore( *`Año`: str* )**:
    Top 5 juegos según año con mayor metascore.



[Link documentación adicional](https://github.com/marianobernal/PI1_Mariano_Bernal/blob/master/1_Transformaciones_Funciones.ipynb)



## 2-EDA

Una vez que logramos disponibilizar esta información para que pueda ser consultada con las funciones que nos solicitó el cliente pasamos al análisis exploratorio de datos. <br>
Aquí, analizaremos más en detalle la información para comenzar a prepararla y desarrollar un primer modelo de machine learning que nos permita aproximarnos a una predicción. <br>
Las principales tareas que realizamos fueron: <br>

+ Reducción de cantidad de columnas: eliminamos varias ya que luego de un primer analisis entendimos que no sumarían información valiosa al modelo. Los principales motivos era la gran cantidad de nulos, la gran variabilidad de los valores o que eran registros categóricos que no aportarían nada al modelo.('Precio_descuento','Metascore','Url_reseñas','Título','Url','Temas','Desarrollador','Editor','Id') <br>
+ Reducción de registros: fuimos realizando la eliminación de registros por varios motivos, principalmente: seleccionamos un rango de fechas donde mayor cantidad de registros teníamos y más cercano a la actualizada, reducción por outliers (tomamos 3 sigma pero ampliandolo a 4 sigmas por la gran variabilidad de algunos registros) <br>
+ Analizamos la gran variabilidad de distribuciones con que cuenta el dataset según distintas variables categóricas (principalmente géneros, tags y acceso_temprano). Esta información nos ayudó a tomar decisiones como la comentada en el punto anterior  <br>
+ Generamos los one hot encoder mediante get dummies para las variables categóricas que decidimos incluir en el modelo <br>
+ Analizamos la existencia o no de variables redundantes mediante heat map <br>
Finalmente, luego de todos estos análisis obtuvimos como entregable el dataframe que utilizaremos como base para el modelo de machine learnin. La idea inicial es desarrollarlo con géneros, mes de lanzamiento y acceso temprarno como features del modelo pero para esta entrega de MVP no incluíremos todavía el mes ya que se encuentra en desarrollo.<br>
Con las definiciones que tomamos, nos quedamos con alrededor de 27.000 registros para entrenar el modelo de los 33.000 originales que teníamos en la base de datos.
<br>
[Link documentación adicional](https://github.com/marianobernal/PI1_Mariano_Bernal/blob/master/2_EDA.ipynb)


## `3-Modelo de Machine Learning` 

Ya contando con la información limpia y analizada de los puntos anteriores, pasamos al desarrollo del modelo. Las principales tareas que realizamos fueron (la gran mayoría con la biblioteca sklearn): <br>

+ Realizamos un random split de las bases de datos para entrenar dos modelos. Como primeros intentos, entrenamos un regrisión lineal multivarialbe y una regresión con un arbol de decisión <br>
+ Optimizamos el hiperparámetros profundidad del arbol mediante el análisis de la curva de validación <br>
+ Realizamos una validación cruzada de ambos modelos para terminar de definir cual era el más conveniente y que no tuviesemos ningún sesgo en las muestras aleatorias, decidiendonos por el arbol con una profundidad de 15 por tener menor RMSE <br>
Por último, desarrollamos la función que luego cargaremos en el archivo main con FASTAPI para deployar en con render.
+ def **predicción( *`genero, earlyaccess = True/False, (Variables que elijas)`* )**:
    Ingresando estos parámetros, deberíamos recibir el precio y **RMSE**.
<br>
[Link documentación adicional](https://github.com/marianobernal/PI1_Mariano_Bernal/blob/master/3_ML.ipynb)


## 4-Archivo main, github y Deployment  

Por último lo que hicimos para lograr deployar las funciones solicitadas en una página web fue lo siguiente: <br>
+ En un archivo llamado main.py consolidamos todas las funciones desarrolladas junto con los archivos necesarios para poder ejecutar deichas funciones <br>
+ Trabajamos con FASTAPI, generando un decorador para cada función<br>
+ Montamos un servidor con uvicorn para poder deployarlo primero localmente y analizar el funcionamiento y poder revisar la documentación generada por FAST API<br>
+ Una vez que quedo funcionando ok, genereamos un entorno virtual y creamos el archivo requirements.txt con las bibliotecas necesarias del proyecto. Luego cargamos el proyecto en un repositorio de github y los deployamos en una página web para que sea accesible mediante render<br>


## **Enlaces útiles**

**`Deployment web`**: https://fastapimarianobernal.onrender.com/docs 

**`Repositorio`**: https://github.com/marianobernal/PI1_Mariano_Bernal  

**`Video deployment`**: https://github.com/marianobernal/PI1_Mariano_Bernal 


