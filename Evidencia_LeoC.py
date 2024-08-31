# Evidencia Redes neuronales
# Leonardo Cossío Dinorín

# EvidenciaLeoC:
# Código que utiliza las funciones definidas en NN_Functions.py para 
# crear, entrenar y evaluar una red neuronal para la clasificación de 
# digitos (0 al 9) escritos a mano.

# Dataset obtenido de:
# https://www.kaggle.com/c/digit-recognizer

# Importación del archivo que contiene las funciones necesarias
from NN_Functions import *

def main():
    # Ruta al dataset
    dataset_path = "train.csv"

    # train_percentage: Porcentaje del data set para entrenamiento (El resto será para test)
    # valid_percentage: Porcentaje del train set destinado para validation
    train_percentage = 0.7
    valid_percentage = 0.15

    # Definir hiperparámetros
    iterations = 300 # "epochs"
    alpha = 0.3 # Learning rate

    # Dimensiones de la red neuronal
    # NOTA: 
    # Input debe ser de ese tamaño debido
    # a los datos utilizados (Imágenes de 28 x 28)
    # Output debe ser 10 debido al número de 
    # clases (0 al 9)
    input_dim = 784
    hidden1_dim = 32 # Neuronas de la primera capa oculta
    hidden2_dim = 16 # Neuronas de la segunda capa oculta
    output_dim = 10

    # Preparación de los datos (División y normalización)
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = data_prep(dataset_path, train_percentage, valid_percentage)
    print("Train shape: ", X_train.shape)
    print("Valid shape: ", X_valid.shape)
    print("Test shape: ", X_test.shape)
    
    # Entrenar el modelo
    w1, b1, w2, b2, w3, b3 = train_model(X_train, Y_train, X_valid, Y_valid, iterations, alpha, input_dim, hidden1_dim, hidden2_dim, output_dim)
    
    # Métricas para evaluar al modelo con el set de validación
    metrics(X_valid, Y_valid, w1, b1, w2, b2, w3, b3, Set="Validation")

    # Métricas para evaluar al modelo con el set de prueba
    metrics(X_test, Y_test, w1, b1, w2, b2, w3, b3, Set="Test")

    # Probar predicciones individuales
    visualization(20, w1, b1, w2, b2, w3, b3, X_test, Y_test)

if __name__ == "__main__":
    main()
