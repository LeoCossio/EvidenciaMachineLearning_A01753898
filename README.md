Este repositorio incluye todo lo necesario para crear y ejecutar una red neuronal que clasifica imágenes (dígitos del 0 al 9 escritos a mano).
Para eso es neesario descargar el dataset "train.csv" presente en el .zip de este repositorio o visitando https://www.kaggle.com/c/digit-recognizer e instalando el mismo archivo mencionado.

Una vez descargados los dos archivos .py de este repositorio y el dataset correspondiente, se puede ejecutar EvidenciaLeoC.py, el cual depende de las funciones dentro de NN_Functions.py.
Si se ha ejecutado correctamente, este archivo comenzará a entrenar una red neuronal para posteriormente mostrar los resultados de las métricas y la matriz de confusión para los sets de validación y prueba.
Después deberá mostrarse una interfaz gráfica donde se muestren 20 imágenes del dataset (test) seleccionadas al azar junto con su etiqueta y la predicción de la red neuronal.
