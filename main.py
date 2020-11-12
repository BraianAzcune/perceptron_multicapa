#configurar red neuronal o cargar la red neuronal previa
from funcion_activacion import sigm
from estructura import crear_red_neuronal
from cargar_datos import cargar_datos
from persistencia_red_neuronal import *
import entrenamiento as en
from funcion_costo import error_cuadratico_medio as ecm

#Esto deberia o estar dentro un objeto, o algo, para cargarlo como archivo de configuracion
cantidad_de_entradas = 8
topologia = [cantidad_de_entradas, 8, 10, 6, 1]
tasa_aprendizaje = 0.2
entrenar = True
epocas = 1000
usar_red_neuronal_guardada = True
path_datos= "DataSets/datasetdiabetes.txt"
path_datos_evaluacion= "DataSets/datasetdiabetesEvalluar.txt"


if usar_red_neuronal_guardada:
    # cargar red_neuronal
    red_neuronal = cargar()

else:
    red_neuronal = crear_red_neuronal(topologia=topologia, act_f=sigm)



if entrenar:
    # cargar datos
    data, respuesta = cargar_datos(path_datos)

    #entrenamiento
    for i in range(epocas):
        prediccion = en.entrenar(red_neuronal=red_neuronal, Entrada=data, Respuesta=respuesta,
                          funcion_costo=ecm, tasa_aprendizaje=tasa_aprendizaje, entrenar=True)


    # guardar red_neuronal
    guardar(red_neuronal=red_neuronal)



else:
    # cargar datos
    data, respuesta = cargar_datos(path_datos_evaluacion)

    #predecir
    prediccion = en.entrenar(red_neuronal=red_neuronal, Entrada=data, Respuesta=respuesta,
                          funcion_costo=ecm, tasa_aprendizaje=tasa_aprendizaje, entrenar=False)



#mostrar performance de la red
