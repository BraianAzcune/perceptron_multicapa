import pandas as pd
import numpy as np

def cargar_datos(path:str):
    """
    dado el path, retorna una tupla con= un numpy.narray primer array filas, segundo array valores de las columnas
    con= numpy.narray donde solo tiene valores de la ultima columna del dataset.
    """
    data_set_diabetes = pd.read_csv(path)

    data = data_set_diabetes.to_numpy()
    #numpy.ndarray

    respuestas = data[:, -1]
    respuestas = respuestas[:, np.newaxis]
    data = data[:, :-1]

    return data, respuestas
