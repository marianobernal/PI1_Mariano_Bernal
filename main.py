# -*- coding: utf-8 -*-

from fastapi import FastAPI, Response
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

app = FastAPI()
dfsteam = pd.read_csv("PI MLOps - STEAM/dfSteamFunciones.csv",encoding='utf-8')

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
