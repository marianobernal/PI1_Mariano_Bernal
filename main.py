# -*- coding: utf-8 -*-

from fastapi import FastAPI, Response
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

app = FastAPI()
dfsteam = pd.read_csv("Archivos/dfSteamFunciones.csv",encoding='utf-8')

dfsteam.rename(columns={'Año': 'Year'}, inplace=True)

dfsteam['Género'].fillna('[]',inplace=True)
dfsteam['Género'] = dfsteam['Género'].apply(ast.literal_eval)
dfsteam['Género'] = dfsteam['Género'].apply(lambda x: np.nan if not x else x)


dfsteam['Especificaciones'].fillna('[]',inplace=True)
dfsteam['Especificaciones'] = dfsteam['Especificaciones'].apply(ast.literal_eval)

@app.get("/genero/{Year}")
def genero(Year:str):
    "Se ingresa un Year y devuelve una lista con los 5 géneros más ofrecidos en el orden correspondiente."
    try:
        if int(Year) < dfsteam['Year'].min() or int(Year) > dfsteam['Year'].max():
            return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}
        else:
            df1= dfsteam[['Género','Year']]
            df2 = df1[df1['Year'] == int(Year)].explode('Género')
            serie = df2.groupby('Género')['Year'].count().sort_values(ascending = False).head(5)
            dicc = dict(serie)
            return {int(Year):list(dicc.keys())}
    except ValueError:
        return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}

@app.get("/juegos/{Year}")
def juegos (Year:str):
    "Se ingresa un Year y devuelve una lista con los juegos lanzados en el Year."
    try:    
        if int(Year) < dfsteam['Year'].min() or int(Year) > dfsteam['Year'].max():
            return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}
        else:
            lista = dfsteam['Título'][(dfsteam['Year']== int(Year))].unique()
            return {int(Year):list(lista)}
    except ValueError:
        return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}

@app.get("/specs/{Year}")
def specs (Year:str):
    "Se ingresa un Year y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente"
    try:    
        if int(Year) < dfsteam['Year'].min() or int(Year) > dfsteam['Year'].max():
            return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}
        else:
            df1= dfsteam[['Especificaciones','Year']]
            df2 = df1[df1['Year'] == int(Year)].explode('Especificaciones')
            lista = df2.groupby('Especificaciones')['Year'].count().sort_values(ascending = False).head(5)
            return {int(Year):list(dict(lista).keys())}
    except ValueError:
        return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}

@app.get("/earlyacces/{Year}")
def earlyacces (Year:int):
    "Cantidad de juegos lanzados en un Year con early access."
    try:    
        if int(Year) < dfsteam['Year'].min() or int(Year) > dfsteam['Year'].max():
            return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}
        else:      
            lista = dfsteam['Título'][(dfsteam['Year']== int(Year))&(dfsteam['Acceso_temprano'] == True)]
            return {int(Year):int(lista.count())}
    except ValueError:
        return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}

@app.get("/sentiment/{Year}")
def sentiment (Year:int):
    "Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento."
    try:    
        if int(Year) < dfsteam['Year'].min() or int(Year) > dfsteam['Year'].max():
            return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}
        else:
            df = dfsteam[(dfsteam['Year']==int(Year))&(dfsteam['Sentimiento'] != 'Sin Dato')]
            serie = dict(df[['Sentimiento','Título']].groupby('Sentimiento')['Título'].count().sort_values(ascending=False))
            val = list(serie.values())
            for i in range(0,len(val)):
                val[i] = int(val[i])
            ky = list(serie.keys())
            diccionario = {nombre: edad for nombre, edad in zip(ky, val)}
            return diccionario
    except ValueError:
        return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}


@app.get("/metascore/{Year}")
def metascore (Year:int):
    "Top 5 juegos según año con mayor metascore."
    try:    
        if int(Year) < dfsteam['Year'].min() or int(Year) > dfsteam['Year'].max():
            return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}
        else:   
            dic = dict(dfsteam[['Metascore','Título']][(dfsteam['Year']==int(Year))].groupby('Título')['Metascore'].mean().sort_values(ascending=False).head(5))
            return {Year:list(dic.keys())}
    except ValueError:
        return {'Por favor pruebe nuevamente ingresando un Year como números enteros entre': [int(dfsteam['Year'].min()),int(dfsteam['Year'].max())]}



dfml = pd.read_csv("Archivos/dfML.csv")
y=dfml['Precio'].round(1)
x=dfml.drop(columns=['Precio'])
Modelo_árbol = DecisionTreeRegressor(max_depth=6)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)
Modelo_árbol.fit(X_train,y_train)
y_train_pred = Modelo_árbol.predict(X_train)
y_test_pred = Modelo_árbol.predict(X_test)
rmse = (mean_squared_error(y_test, y_test_pred, squared = False))


@app.get("/predict/")
def predict (Mes:int,Géneros:str,Acceso_temprano):
    """
    Esta función deber recibir 
        - mes:el mes con números del 1 al 12
        - input_text: los géneros separados con coma seguido de un espacio y con el prefijo Género. Por ejemplo: 'Género_Action, Género_Accounting'
        - acceso_temprano: 1 para True o 0 para False
    La función devolverá el valor predicho seguido del RMSE del modelo
    Los géneros disponibles son : ['Género_Accounting',  'Género_Action', 'Género_Adventure',  
    'Género_Animation &amp; Modeling',  'Género_Audio Production',  'Género_Casual',  'Género_Design &amp; Illustration', 
    'Género_Early Access',  'Género_Education',  'Género_Free to Play',  'Género_Indie',  'Género_Massively Multiplayer', 
    'Género_Photo Editing',  'Género_RPG',  'Género_Racing',  'Género_Simulation',  'Género_Software Training', 
    'Género_Sports',  'Género_Strategy', 'Género_Utilities',  'Género_Video Production',  'Género_Web Publishing']
    """
    columnas_modelo = x.columns.tolist()[1:23]
    columnas_usuario_lista = Géneros.split(", ")
    # Crea un conjunto (set) con las columnas proporcionadas por el usuario
    columnas_usuario = set(columnas_usuario_lista)
    # Encuentra la intersección entre las columnas del modelo y las columnas proporcionadas por el usuario
    columnas_validas = columnas_usuario.intersection(columnas_modelo)
    # Crea un diccionario con las columnas proporcionadas por el usuario y sus respectivos valores (1 o 0)
    valores_usuario = {columna: 1 if columna in columnas_usuario else 0 for columna in columnas_modelo}
    # Crea un DataFrame a partir del diccionario
    df_usuario = pd.DataFrame(valores_usuario, index=[0])
    # Convierte el DataFrame en una lista ordenada
    género = df_usuario.to_numpy().tolist()[0]
    features = [Mes]+género+[Acceso_temprano]
    x_pred = np.array(features).reshape(1, -1)
    return {'Predicción de precio': float(Modelo_árbol.predict(x_pred)), 'RMSE':float(rmse)}