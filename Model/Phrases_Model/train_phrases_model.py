import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from Utils.paths import phrases_model_json_data_path, phrases_model_converted_data_path, generated_models_path
from Utils.config import phrases_model_keras_name
from Model.model_utils import plot_history


def train_and_save_model(model_path, epochs=500, max_length_frames=15, max_length_keypoints=1662):
    """
    Entrena y guarda un modelo LSTM basado en los datos proporcionados.

    Parámetros:
    model_path -- Ruta donde se guardará el modelo entrenado.
    epochs -- Número de épocas para el entrenamiento (por defecto 500).

    Funcionalidad:
    - Carga los identificadores de palabras desde un archivo JSON.
    - Recupera las secuencias de keypoints y sus etiquetas desde los archivos HDF5.
    - Define la arquitectura del modelo LSTM.
    - Preprocesa las secuencias de entrada y las etiquetas.
    - Divide los datos en entrenamiento y validación.
    - Entrena el modelo utilizando EarlyStopping.
    - Guarda el modelo entrenado en la ruta especificada.
    """
    # Cargar identificadores de palabras desde el archivo JSON
    with open(phrases_model_json_data_path, 'r') as json_file:
        data = json.load(json_file)
        word_ids = data.get('word_ids', [])  # Ejemplo: ['word1', 'word2', 'word3']
        if not word_ids:
            raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

    # Obtener las secuencias de keypoints y etiquetas desde los archivos HDF5
    sequences, labels = [], []
    for word_index, word_id in enumerate(word_ids):
        hdf_path = os.path.join(phrases_model_converted_data_path, f"{word_id}.h5")
        if not os.path.exists(hdf_path):
            print(f"[WARNING] Archivo HDF5 no encontrado: {hdf_path}. Saltando...")
            continue
        data = pd.read_hdf(hdf_path, key='data')
        for _, df_sample in data.groupby('sample'):
            seq_keypoints = [fila['keypoints'] for _, fila in df_sample.iterrows()]
            sequences.append(seq_keypoints)
            labels.append(word_index)

    if not sequences or not labels:
        raise ValueError("[ERROR] No se encontraron secuencias ni etiquetas para entrenar el modelo.")

    # Ajustar las secuencias al tamaño máximo definido por el modelo
    sequences = pad_sequences(sequences, maxlen=int(max_length_frames), padding='pre', truncating='post',
                              dtype='float16')

    # Convertir las etiquetas a formato categórico (one-hot encoding)
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

    # Definir la arquitectura del modelo
    model = Sequential()

    # Capa LSTM inicial con regularización L2
    model.add(LSTM(64, return_sequences=True, input_shape=(int(max_length_frames), max_length_keypoints),
                   kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    # Segunda capa LSTM sin devolver secuencias
    model.add(LSTM(128, return_sequences=False, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))

    # Capas densas totalmente conectadas
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))

    # Capa de salida con activación softmax
    model.add(Dense(len(word_ids), activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Configurar EarlyStopping para detener el entrenamiento si no hay mejoras
    early_stopping = EarlyStopping(monitor='accuracy', patience=20, restore_best_weights=True)

    # Entrenar el modelo
    print("[INFO] Iniciando entrenamiento del modelo...")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=8, callbacks=[early_stopping])

    # Guardar el modelo entrenado
    model.save(model_path)
    print(f"[INFO] Modelo guardado en: {model_path}")

    # Mostrar el resumen del modelo
    print("[INFO] Resumen del modelo:")
    model.summary()

    # Evaluar el modelo
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[INFO] Evaluación en datos de validación - Pérdida: {val_loss:.4f}, Precisión: {val_acc:.4f}")

    plot_history(history)



if __name__ == "__main__":
    """
    Script principal para entrenar y guardar el modelo LSTM.
    """
    # Entrenar el modelo y guardar el resultado en la ruta especificada
    train_and_save_model(os.path.join(generated_models_path, phrases_model_keras_name))
