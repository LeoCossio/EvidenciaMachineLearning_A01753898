# Evidencia Redes neuronales
# Leonardo Cossío Dinorín

# NN_Functions:
# Código que contiene las funciones necesarias para crear, entrenar 
# y evaluar una red neuronal para la detección de digitos (0 al 9) 
# escritos a mano.

# Dataset obtenido de:
# https://www.kaggle.com/c/digit-recognizer
# El dataset es un csv que contiene los valores para 784 pixeles (Imágenes de 28x28)
# de números del 0 al 9 escritos a mano. A excepción de la etiqueta, los valores del
# dataset oscilan entre el 0 y el 255, representando un valor en escala de grises, siendo
# 0 = completamente negro y 255 = completamente blanco

# Librerías
import numpy as np # Arrays y matrices multidimensionales
import pandas as pd # Análisis y manipulación de datos
from matplotlib import pyplot as plt # Visualización de resultados

# ---------- Preparación del dataset ---------------------
"""
data_prep:
Función que se utiliza para preparar el set de datos:
- Revolver datos (shuffle) para evitar overfitting.
- Separar sets de train, validation y test.
- Normalizar datos -> Originalmente los datos son valores de entre 0 y 255 (Escala de grises),
                      por lo que se reaiza una normalización, de esta manera nos aseguramos que
                      ninguna característica domine debido a su escala.

    Args:
    - path (str): Ruta al archivo CSV que contiene los datos.
    - train_percentage (float): Porcentaje de datos a usar para entrenamiento.
    - valid_percentage (float): Porcentaje de datos a usar para validación dentro del conjunto de entrenamiento.

    Returns:
    - X_train (numpy array): Conjunto de atributos para entrenamiento.
    - Y_train (numpy array): Etiquetas para entrenamiento.
    - X_valid (numpy array): Conjunto de atributos para validación.
    - Y_valid (numpy array): Etiquetas para validación.
    - X_test (numpy array): Conjunto de atributos para prueba.
    - Y_test (numpy array): Etiquetas para prueba.
"""
def data_prep(path, train_percentage, valid_percentage):
    # Leer y cargar los datos de entrenamiento
    data = pd.read_csv(path)

    # Convertir los datos a un array de numpy
    data = np.array(data)
    # m: Filas
    # n: Columnas (Incluyendo el label)
    m, n = data.shape

    # Revolver los datos
    np.random.shuffle(data)

    # Dónde se separará el dataset para entrenamiento
    split_index = int(m * train_percentage)
    split_index_valid = int(m * train_percentage * valid_percentage)

    # Separar el dataset de train
    data_train = data[split_index_valid:split_index].T
    Y_train = data_train[0] # Label (Y)
    X_train = data_train[1:n] # Features (X)

    n_train, m_train = data_train.shape

    # Separar el dataset de validación
    data_valid = data[0:split_index_valid].T
    Y_valid = data_valid[0] # Label (Y)
    X_valid = data_valid[1:n_train] # Features (X)

    # Separar el dataset de test
    data_test = data[split_index:m].T
    Y_test = data_test[0] # Label (Y)
    X_test = data_test[1:n] # Features (X)
    
    # Normalizar datos
    X_train = X_train/255.
    X_valid = X_valid/255.
    X_test = X_test/255.

    # Regresa los 3 sets creados
    return X_train, Y_train,X_valid, Y_valid, X_test, Y_test

# --------- Inicialización de la red neuronal --------------------------
"""
init_params:
Define la forma de nuestra red neuronal e inicializa los pesos y sesgos de manera aleatoria.

    args:
    - input_dim (int): Dimensión de la capa de entrada (Imagenes de 28x28)
    - hidden1_dim (int): Dimensiones de la primera capa oculta
    - hidden2_dim (int): Dimensiones de la segunda capa oculta
    - output_dim (int): Dimensiones de la capa de salida (10 clases)

    returns:
    - w1 (numpy array): Arreglo de numpy inicializado con valores aleatorios con los pesos de la primera capa oculta.
    - b1 (numpy array): Arreglo de numpy inicializado con valores aleatorios con los sesgos de la primera capa oculta.
    - w2 (numpy array): Arreglo de numpy inicializado con valores aleatorios con los pesos de la segunda capa oculta.
    - b2 (numpy array): Arreglo de numpy inicializado con valores aleatorios con los sesgos de la segunda capa oculta.
    - w3 (numpy array): Arreglo de numpy inicializado con valores aleatorios con los pesos de la capa de salida.
    - b3 (numpy array): Arreglo de numpy inicializado con valores aleatorios con los sesgos de la capa de salida.

NOTA: Se les resta 0.5 para que los valores queden centrados en 0, 
lo cual ayuda a romper la simetría entre los pesos iniciales
"""
def init_params(input_dim, hidden1_dim, hidden2_dim, output_dim):
    w1 = np.random.rand(hidden1_dim, input_dim) - 0.5
    b1 = np.random.rand(hidden1_dim, 1) - 0.5
    w2 = np.random.rand(hidden2_dim, hidden1_dim) - 0.5
    b2 = np.random.rand(hidden2_dim, 1) - 0.5
    w3 = np.random.rand(output_dim, hidden2_dim) - 0.5
    b3 = np.random.rand(output_dim, 1) - 0.5

    return w1, b1, w2, b2, w3, b3

# --------------- Funciones de activación --------------------
"""
ReLU:
Aplica la función de activación ReLU (Rectified Linear Unit) a una matriz o vector:
Si Z > 0 --> Z
Si Z <= 0 --> 0

    Args:
    - Z (numpy array): Entrada, que puede ser una matriz o un vector.

    Returns:
    - numpy array: Resultado de aplicar la función ReLU a cada elemento de Z. 
"""
def ReLU(Z):
    return np.maximum(Z, 0)

"""
Función ReLU derivada:
Calcula la derivada de la función de activación ReLU (Rectified Linear Unit) para una matriz o vector.
Si Z > 0 --> 1
Si Z <= 0 --> 0

    Args:
    - Z (numpy array): Entrada, que puede ser una matriz o un vector.

    Returns:
    - numpy array: Resultado de la derivada de ReLU. 

"""
def deriv_ReLU(Z):
    return (Z > 0)

"""
Softmax: Genera un vector de probabilidades que representa 
la posibilidad de que la salida pertenezca a cada una de las 
clases en un problema de clasificación multiclase.
Utilizada en la capa de salida de esta red.

Softmax = exp(Zi)/sum(exp(Zi))

Cada elemento de la salida es un valor entre 0 y 1, y la suma de todos los valores en la misma columna es 1.

    Args:
    - Z (numpy array): Entrada, que puede ser una matriz o un vector. 

    Returns:
    - numpy array: Resultado de aplicar la función softmax. 
"""
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

# ------------ Feed Forward ----------------------------------
"""
Forward propagation:
Realiza la propagación hacia adelante a través de una red neuronal.
Calcula los valores lineales Z para cada capa usando los pesos y sesgos,
aplica las funciones de activación correspondientes, y devuelve los 
valores no activados y activados para cada capa.

Zi: Resultado de la ecuación de la neurona ANTES de pasar
por la función de activación

Ai: Resultado de Z DESPUÉS de pasar por la función de activación.

    Args:
    - w1 (numpy array): Matriz de pesos de la primera capa.
    - b1 (numpy array): Vector de sesgos de la primera capa.
    - w2 (numpy array): Matriz de pesos de la segunda capa.
    - b2 (numpy array): Vector de sesgos de la segunda capa.
    - w3 (numpy array): Matriz de pesos de la tercera capa.
    - b3 (numpy array): Vector de sesgos de la tercera capa.
    - X (numpy array): Datos de entrada (características).

    Returns:
    - Z1 (numpy array): Suma ponderada de entradas en la primera capa antes de la activación.
    - A1 (numpy array): Activación de la primera capa (ReLU).
    - Z2 (numpy array): Suma ponderada de entradas en la segunda capa antes de la activación.
    - A2 (numpy array): Activación de la segunda capa (ReLU).
    - Z3 (numpy array): Suma ponderada de entradas en la tercera capa antes de la activación.
    - A3 (numpy array): Salida final de la red (softmax).
"""
def forward_prop(w1, b1, w2, b2, w3, b3, X):
    # Z = Resultado antes de la activación
    Z1 = w1.dot(X) + b1
    # A = Resultado después de la activación (ReLU)
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = w3.dot(A2) + b3
    # Para la salida se utiliza softmax
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# ----------- One Hot Encoding ------------------------
"""
One hot encoding:
En este caso se aplica para la salida,
de esta manera en lugar de tener un array de 
10 posibles valores (0 al 9), se tienen 10 
arrays llenos de 0's y 1's.

    Args:
    - Y (numpy array): Vector de etiquetas de clase. Cada valor en Y representa la clase a la que pertenece una muestra.

    Returns:
    - one_hot_Y (numpy array): Matriz one-hot de tamaño (n_clases, n_muestras), 
      donde cada columna es una representación one-hot de la etiqueta correspondiente en Y.

"""
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T # Se utiliza para asegurarse que se encuentre en el formato correcto.
    return one_hot_Y

# ----------------- Back propagation --------------------
"""
Backpropagation:
Se encarga de calcular el error y
obtener los valores para actualizar los
pesos y bias de las neuronas

    Args:
    - Z1 (numpy array): Suma ponderada de entradas en la primera capa antes de la activación.
    - A1 (numpy array): Activación de la primera capa.
    - Z2 (numpy array): Suma ponderada de entradas en la segunda capa antes de la activación.
    - A2 (numpy array): Activación de la segunda capa.
    - Z3 (numpy array): Suma ponderada de entradas en la tercera capa antes de la activación.
    - A3 (numpy array): Salida final de la red (activaciones de la tercera capa).
    - w1 (numpy array): Matriz de pesos de la primera capa.
    - w2 (numpy array): Matriz de pesos de la segunda capa.
    - w3 (numpy array): Matriz de pesos de la tercera capa.
    - X  (numpy array): Datos de entrada (características).
    - Y  (numpy array): Etiquetas de clase verdaderas para las muestras de entrada.

    Returns:
    - dw1 (numpy array): Gradiente de la matriz de pesos de la primera capa.
    - db1 (numpy array): Gradiente del vector de sesgos de la primera capa.
    - dw2 (numpy array): Gradiente de la matriz de pesos de la segunda capa.
    - db2 (numpy array): Gradiente del vector de sesgos de la segunda capa.
    - dw3 (numpy array): Gradiente de la matriz de pesos de la tercera capa.
    - db3 (numpy array): Gradiente del vector de sesgos de la tercera capa.

"""
def back_prop(Z1, A1, Z2, A2, Z3, A3, w1, w2, w3, X, Y):
    m = Y.size
    # One hot encoding para la capa de salida
    one_hot_Y = one_hot(Y)

    # Error de la capa de salida
    dZ3 = A3 - one_hot_Y
    # Gradiente del peso
    dw3 = 1/m * dZ3.dot(A2.T)
    # Gradiente del sesgo
    db3 = 1/m * np.sum(dZ3, axis=1, keepdims=True)
    
    # Error propagado a la segunda capa oculta
    dZ2 = w3.T.dot(dZ3) * deriv_ReLU(Z2)
    # Gradiente de los pesos entre la primera y la segunda capa oculta
    dw2 = 1/m * dZ2.dot(A1.T)
    # Gradiente del sesgo de la segunda capa oculta
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    
    # Error propagado a la primera capa oculta
    dZ1 = w2.T.dot(dZ2) * deriv_ReLU(Z1)
    # Gradiente de los pesos entre la entrada y la primera capa oculta
    dw1 = 1/m * dZ1.dot(X.T)
    # Gradiente de sesgos de la primera capa oculta
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    return dw1, db1, dw2, db2, dw3, db3

# ----------- Actualizar parámetros (Descenso de gradiente) ------------------
"""
update_params:
Aquí se aplican las ecuaciones del algoritmo
de descenso de gradiente para poder actualizar 
los pesos y bias de la red neuronal tomando en 
cuenta el learning rate (alpha)

    Args:
    - w1 (numpy array): Matriz de pesos de la primera capa.
    - b1 (numpy array): Vector de sesgos de la primera capa.
    - w2 (numpy array): Matriz de pesos de la segunda capa.
    - b2 (numpy array): Vector de sesgos de la segunda capa.
    - w3 (numpy array): Matriz de pesos de la tercera capa.
    - b3 (numpy array): Vector de sesgos de la tercera capa.
    - dw1 (numpy array): Gradiente de la matriz de pesos de la primera capa.
    - db1 (numpy array): Gradiente del vector de sesgos de la primera capa.
    - dw2 (numpy array): Gradiente de la matriz de pesos de la segunda capa.
    - db2 (numpy array): Gradiente del vector de sesgos de la segunda capa.
    - dw3 (numpy array): Gradiente de la matriz de pesos de la tercera capa.
    - db3 (numpy array): Gradiente del vector de sesgos de la tercera capa.
    - alpha (float): Tasa de aprendizaje para la actualización de los parámetros.

    Returns:
    - w1 (numpy array): Matriz de pesos actualizada de la primera capa.
    - b1 (numpy array): Vector de sesgos actualizado de la primera capa.
    - w2 (numpy array): Matriz de pesos actualizada de la segunda capa.
    - b2 (numpy array): Vector de sesgos actualizado de la segunda capa.
    - w3 (numpy array): Matriz de pesos actualizada de la tercera capa.
    - b3 (numpy array): Vector de sesgos actualizado de la tercera capa.
"""
def update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
    # w = w - alpha * dw
    # b = b - alpha * db
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2
    w3 -= alpha * dw3
    b3 -= alpha * db3
    return w1, b1, w2, b2, w3, b3

# ---------------- Obtener predicciones -----------------------
"""
get_predictions:
Obtiene las predicciones realizadas por la red neuronal.
Como la salida de la red es un array que contiene probabilidades 
para cada clase, esta función selecciona la clase con la probabilidad 
más alta como el resultado final.

    args:
    A2 (numpy array): Vector de probabilidades (predicciones de la red neuronal)

    returns:
    Valor máximo del vector A2 (resultado con mayor probabilidad de ser correcto)

"""
def get_predictions(A2):
    # Regresa el valor más grande del vector (Mayor probabilidad)
    return np.argmax(A2, 0)

# -------------------------- Métricas -------------------------------
"""
get_accuracy:
Calcula la precisión del modelo comparando 
las predicciones con las etiquetas reales.

Accuracy: (TP + TN)/(TP + TN + FP + FN)

    Args:
    - predictions (numpy array): Vector de predicciones generadas por el modelo.
    - Y (numpy array): Vector de etiquetas verdaderas correspondientes a las muestras.

    Returns:
    - accuracy (float): Precisión del modelo, calculada como la proporción de predicciones correctas.

"""
def get_accuracy(predictions, Y):
    accuracy = np.sum(predictions == Y)/Y.size
    return accuracy

"""
get_precision:
Calcula la precisión del modelo comparando 
las predicciones con las etiquetas reales.
La precisión es la proporción de verdaderos 
positivos sobre el total de predicciones positivas.

Precision = TP/(TP+FP)

    Args:
    - predictions (numpy array): Vector de predicciones generadas por el modelo, con valores 0 o 1.
    - Y (numpy array): Vector de etiquetas verdaderas correspondientes a las muestras, con valores 0 o 1.

    Returns:
    - precision (float): Precisión del modelo, definida como la proporción de verdaderos positivos entre todos los positivos predichos.

"""
def get_precision(predictions, Y):
    true_positives = np.sum((predictions == 1) & (Y == 1))
    predicted_positives = np.sum(predictions == 1)
    if predicted_positives == 0:
        precision = 0
    else:
        precision = true_positives / predicted_positives
    return precision

"""
get_recall:
Calcula el recall del modelo comparando 
las predicciones con las etiquetas reales.
El recall es la proporción de verdaderos positivos 
sobre el total de casos que deberían haber sido positivos.

Recall = TP / (TP + FN)

    Args:
    - predictions (numpy array): Vector de predicciones generadas por el modelo, con valores 0 o 1.
    - Y (numpy array): Vector de etiquetas verdaderas correspondientes a las muestras, con valores 0 o 1.

    Returns:
    - recall (float): Recall del modelo, definido como la proporción de verdaderos positivos entre todos los verdaderos positivos y falsos negativos.

"""
def get_recall(predictions, Y):
    true_positives = np.sum((predictions == 1) & (Y == 1))
    false_negatives = np.sum((predictions == 0) & (Y == 1))
    if (true_positives + false_negatives) == 0:
        recall = 0
    else:
        recall = true_positives/(true_positives + false_negatives)
    return recall

"""
get_specificity:
Calcula la especificidad del modelo
comparando las predicciones con las etiquetas reales.
La especificidad es la proporción de verdaderos negativos 
sobre el total de casos que deberían haber sido negativos.

Specificity = TN / (TN + FP)

    Args:
    - predictions (numpy array): Vector de predicciones generadas por el modelo, con valores 0 o 1.
    - Y (numpy array): Vector de etiquetas verdaderas correspondientes a las muestras, con valores 0 o 1.

    Returns:
    - specificity (float): Especificidad del modelo, definida como la proporción de verdaderos negativos entre todos los verdaderos negativos y falsos positivos.

"""
def get_specificity(predictions, Y):
    true_negatives = np.sum((predictions == 0) & (Y == 0))
    false_positives = np.sum((predictions == 1) & (Y == 0))
    
    if (true_negatives + false_positives) == 0:
        specificity = 0
    else:
        specificity = true_negatives/(true_negatives + false_positives)
    return specificity

"""
get_F1:
Calcula el "F1 Score" del modelo, que nos da una idea 
del balance entre la precisión y el recall.

F1 Score = 2 * (Precisión * Recall) / (Precisión + Recall)

    Args:
    - predictions (numpy array): Vector de predicciones generadas por el modelo, con valores 0 o 1.
    - Y (numpy array): Vector de etiquetas verdaderas correspondientes a las muestras, con valores 0 o 1.

    Returns:
    - F1 (float): Puntaje F1 del modelo, que es la media armónica de la precisión y el recall.

"""
def get_F1(predictions, Y):
    presicion = get_precision(predictions, Y)
    recall = get_recall(predictions, Y)

    if (presicion + recall) == 0:
        F1 = 0
    else:
        F1 = 2*((presicion * recall)/(presicion + recall))
    return F1


# ---------- Entrenamiento del modelo ----------------------

"""
train_model:
Se implementa el algoritmo de descenso de gradiente
para entrenar a la red neuronal.
1) Inicializa los parámetros (Pesos y bias)
2) Inicia un ciclo donde:
    2.1) Realiza un forward propagation
    2.2) Retropropaga los errores para obtener
    los gradientes
    2.3) Actualiza los parámetros
3) Imprime el valor de accuracy obtenido cada cierto
número de ciclos

    Args:
    - X_train (numpy array): Matriz de características de entrada del set de entrenamiento (dimensión: [n_features, n_samples]).
    - Y_train (numpy array): Vector de etiquetas verdaderas del set de entrenamiento (dimensión: [n_samples]).
    - X_val (numpy array): Matriz de características de entrada del set de validación (dimensión: [n_features, n_samples]).
    - Y_val (numpy array): Vector de etiquetas verdaderas del set de validación(dimensión: [n_samples]).
    - iterations (int): Número de iteraciones para entrenar el modelo.
    - alpha (float): Learning rate para la actualización de los parámetros.
    - input_dim (int): Número de características de entrada (dimensión de la capa de entrada).
    - hidden1_dim (int): Número de neuronas en la primera capa oculta.
    - hidden2_dim (int): Número de neuronas en la segunda capa oculta.
    - output_dim (int): Número de clases de salida (dimensión de la capa de salida).

    Returns:
    - w1 (numpy array): Matriz de pesos actualizada de la primera capa.
    - b1 (numpy array): Vector de sesgos actualizado de la primera capa.
    - w2 (numpy array): Matriz de pesos actualizada de la segunda capa.
    - b2 (numpy array): Vector de sesgos actualizado de la segunda capa.
    - w3 (numpy array): Matriz de pesos actualizada de la tercera capa.
    - b3 (numpy array): Vector de sesgos actualizado de la tercera capa.

"""
def train_model(X_train, Y_train, X_val, Y_val, iterations, alpha, input_dim, hidden1_dim, hidden2_dim, output_dim):
    w1, b1, w2, b2, w3, b3 = init_params(input_dim, hidden1_dim, hidden2_dim, output_dim)

    for i in range(iterations):
        # Forward propagation aplicado con el set de entrenamiento
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(w1, b1, w2, b2, w3, b3, X_train)

        # Backward propagation aplicado con el set de entrenamiento
        dw1, db1, dw2, db2, dw3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, w1, w2, w3, X_train, Y_train)

        # Actualizar parámetros
        w1, b1, w2, b2, w3, b3 = update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)

        # Evaluar el desempeño del modelo cada 50 iteraciones
        if i % 50 == 0:
            print("Iteration: ", i)
            
            # Accuracy con el set de entrenamiento
            train_accuracy = get_accuracy(get_predictions(A3), Y_train)
            print("Training Accuracy: ", train_accuracy)
            
            # Forward propagation con el set de validación
            _, _, _, _, _, A3_val = forward_prop(w1, b1, w2, b2, w3, b3, X_val)

            # Accuracy con el set de validación
            val_accuracy = get_accuracy(get_predictions(A3_val), Y_val)
            print("Validation Accuracy: ", val_accuracy)
    
    return w1, b1, w2, b2, w3, b3

# --------- Predicciones (Una vez obtenidos los pesos) ---------------------
"""
make_predictions:
Hace las predicciones utilizando datos de entrada
y pasando a través de la red neuronal YA ENTRENADA

    Args:
    - X (numpy array): Matriz de características de entrada para las cuales se realizarán las predicciones (dimensión: [n_features, n_samples]).
    - w1 (numpy array): Matriz de pesos de la primera capa del modelo.
    - b1 (numpy array): Vector de sesgos de la primera capa del modelo.
    - w2 (numpy array): Matriz de pesos de la segunda capa del modelo.
    - b2 (numpy array): Vector de sesgos de la segunda capa del modelo.
    - w3 (numpy array): Matriz de pesos de la tercera capa del modelo.
    - b3 (numpy array): Vector de sesgos de la tercera capa del modelo.

    Returns:
    - predictions (numpy array): Vector de predicciones generadas por el modelo para las muestras de entrada.

"""
def make_predictions(X, w1, b1, w2, b2, w3, b3):
    _, _, _, _, _, A3 = forward_prop(w1, b1, w2, b2, w3, b3, X)
    predictions = get_predictions(A3)
    return predictions

# ---------- Visualización de resultados ---------------------------
"""
visualization:
Elige una cantidad de objetos al azar, hace las predicciones
y muestra los resultados de manera gráfica

    Args:
    - num_images (int): Número de imágenes a visualizar. Se limita a un máximo de 20.
    - w1 (numpy array): Matriz de pesos de la primera capa del modelo.
    - b1 (numpy array): Vector de sesgos de la primera capa del modelo.
    - w2 (numpy array): Matriz de pesos de la segunda capa del modelo.
    - b2 (numpy array): Vector de sesgos de la segunda capa del modelo.
    - w3 (numpy array): Matriz de pesos de la tercera capa del modelo.
    - b3 (numpy array): Vector de sesgos de la tercera capa del modelo.
    - X (numpy array): Matriz de características de entrada (dimensión: [n_features, n_samples]).
    - Y (numpy array): Vector de etiquetas verdaderas (dimensión: [n_samples]).

    Returns:
    - None: Muestra una visualización de las imágenes y sus predicciones.

"""
def visualization(num_images, w1, b1, w2, b2, w3, b3, X, Y):
    # Limitar las imágenes mostradas a 20
    num_images = min(num_images, 20)
    
    indices = np.random.choice(X.shape[1], num_images, replace=False)
    
    # Calcular el tamaño de la cuadrícula
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Ajustar el tamaño de la figura
    plt.figure(figsize=(15, 15))
    
    for i, index in enumerate(indices):
        current_image = X[:, index].reshape((28, 28)) * 255
        prediction = make_predictions(X[:, index, None], w1, b1, w2, b2, w3, b3)
        label = Y[index]

        plt.subplot(grid_size, grid_size, i+1)
        plt.imshow(current_image, cmap='gray', interpolation='nearest')
        plt.title(f"Predicción: {prediction[0]}\nEtiqueta: {label}", fontsize=10)
        plt.axis('off')
    
    # Ajustar los espacios entre subplots con un padding mayor
    plt.tight_layout(pad=3.0)
    plt.show()

"""
compute_confusion_matrix:
Calcula la matriz de confusión para las etiquetas verdaderas y las predicciones.

    Args:
    - Y_true (numpy array): Vector de etiquetas verdaderas (dimensión: [n_samples]).
    - Y_pred (numpy array): Vector de predicciones realizadas por el modelo (dimensión: [n_samples]).
    - num_classes (int): Número total de clases en el problema de clasificación.

    Returns:
    - cm (numpy array): Matriz de confusión (dimensión: [num_classes, num_classes]).
"""
def compute_confusion_matrix(Y_true, Y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(Y_true, Y_pred):
        cm[true, pred] += 1
    return cm

"""
plot_confusion_matrix:
Visualiza la matriz de confusión usando un mapa de calor.

    Args:
    - cm (numpy array): Matriz de confusión (dimensión: [num_classes, num_classes]).
    - Set (str): Nombre del conjunto de datos (por ejemplo, 'Train' o 'Test'), que se usará en el título del gráfico.
    - classes (list): Lista de nombres de las clases, que se usarán para las etiquetas del eje.

    Returns:
    - None: Muestra un gráfico de la matriz de confusión.
"""
def plot_confusion_matrix(cm, Set, classes):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusión ({Set} set)')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.show()

# --- Automatización de cálculo de métricas --------------------
"""
metrics:
Evalúa el rendimiento del modelo en un conjunto de datos dado y muestra métricas de evaluación.

    Args:
    - X (numpy array): Matriz de características del conjunto de datos en el que se evaluará el modelo (dimensión: [n_features, n_samples]).
    - Y (numpy array): Vector de etiquetas verdaderas del conjunto de datos (dimensión: [n_samples]).
    - w1 (numpy array): Matriz de pesos de la primera capa del modelo.
    - b1 (numpy array): Vector de sesgos de la primera capa del modelo.
    - w2 (numpy array): Matriz de pesos de la segunda capa del modelo.
    - b2 (numpy array): Vector de sesgos de la segunda capa del modelo.
    - w3 (numpy array): Matriz de pesos de la tercera capa del modelo.
    - b3 (numpy array): Vector de sesgos de la tercera capa del modelo.
    - Set (str): Nombre del conjunto de datos que se está evaluando ('Train' o 'Test').

    Returns:
    - None: Muestra las métricas de evaluación y la matriz de confusión.
"""
def metrics(X, Y, w1, b1, w2, b2, w3, b3, Set):
    # Evaluar el modelo en datos de prueba
    predictions_valid = make_predictions(X, w1, b1, w2, b2, w3, b3)
    valid_accuracy = get_accuracy(predictions_valid, Y)
    valid_precision = get_precision(predictions_valid, Y)
    valid_recall = get_recall(predictions_valid, Y)
    valid_specificity = get_specificity(predictions_valid, Y)
    valid_F1 = get_F1(predictions_valid, Y)

    print("\n************************")
    print(f"{Set} Set Accuracy: ", valid_accuracy)
    print(f"{Set} Set Precision: ", valid_precision)
    print(f"{Set} Set Recall: ", valid_recall)
    print(f"{Set} Set Specificity: ", valid_specificity)
    print(f"{Set} Set F1 Score: ", valid_F1)
    print("************************")
    
    # Visualizar matriz de confusión
    num_classes = 10  # Ajusta esto si el número de clases cambia
    cm = compute_confusion_matrix(Y, predictions_valid, num_classes)
    plot_confusion_matrix(cm, Set, classes=np.arange(num_classes))