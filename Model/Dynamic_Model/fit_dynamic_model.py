import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from Utils.paths import dynamic_model_json_data_path, dynamic_model_converted_data_path, generated_models_path, \
    dynamic_model_keras_path

from Utils.config import dynamic_model_name
from Model.model_utils import plot_history


def train_and_save_model(model_path, epochs=300, max_length_frames=15, max_length_keypoints=1662):
    # Cargar identificadores de palabras desde el archivo JSON
    with open(dynamic_model_json_data_path, 'r') as json_file:
        data = json.load(json_file)
        word_ids = data.get('word_ids', [])
        if not word_ids:
            raise ValueError("[ERROR] No se encontraron identificadores de palabras en el archivo JSON.")

        # Imprimir las palabras cargadas
        print(f"[INFO] Se encontraron {len(word_ids)} palabras para entrenar: {word_ids}")

    # Obtener las secuencias de keypoints y etiquetas desde los archivos HDF5
    sequences, labels = [], []
    for word_index, word_id in enumerate(word_ids):
        hdf_path = os.path.join(dynamic_model_converted_data_path, f"{word_id}.h5")
        if not os.path.exists(hdf_path):
            print(f"[WARNING] Archivo HDF5 no encontrado: {hdf_path}. Saltando...")
            continue
        data = pd.read_hdf(hdf_path, key='data')
        for _, df_sample in data.groupby('sample'):
            seq_keypoints = np.stack(df_sample['keypoints'].values)
            sequences.append(seq_keypoints)
            labels.append(word_index)

    if not sequences or not labels:
        raise ValueError("[ERROR] No se encontraron secuencias ni etiquetas para entrenar el modelo.")

    # Verificar cuántos datos se cargaron por palabra
    print(f"[INFO] Se cargaron {len(sequences)} secuencias para entrenamiento.")

    # Convertir las secuencias a numpy arrays
    sequences = np.array(sequences)

    # Aplanar los keypoints por frame para normalización
    num_samples, seq_len, num_keypoints = sequences.shape
    sequences_flat = sequences.reshape(-1, num_keypoints)

    # Normalizar los keypoints
    scaler = StandardScaler()
    sequences_flat = scaler.fit_transform(sequences_flat)

    # Restaurar la forma original
    sequences = sequences_flat.reshape(num_samples, seq_len, num_keypoints)

    # Ajustar las secuencias al tamaño máximo definido por el modelo
    sequences = pad_sequences(
        sequences, maxlen=max_length_frames, padding='post', truncating='post', dtype='float32'
    )

    # Convertir las etiquetas a formato categórico (one-hot encoding)
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    # Mezclar los datos antes de dividir
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Dividir los datos en conjuntos de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    # Definir la arquitectura del modelo
    model = Sequential()

    # Primera capa LSTM con BatchNormalization
    model.add(
        LSTM(
            128,
            return_sequences=True,
            input_shape=(max_length_frames, max_length_keypoints),
            kernel_regularizer=l2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Segunda capa LSTM
    model.add(
        LSTM(
            64,
            return_sequences=False,
            kernel_regularizer=l2(0.001),
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Capas densas
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Capa de salida
    model.add(Dense(len(word_ids), activation='softmax'))

    # Compilar el modelo
    from keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    from keras.callbacks import ReduceLROnPlateau

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )

    # Configurar EarlyStopping para monitorizar la pérdida de validación
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=40,
        restore_best_weights=True
    )

    # Entrenar el modelo
    print("[INFO] Iniciando entrenamiento del modelo...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        shuffle=True
    )

    # Guardar el modelo entrenado
    model.save(model_path)

    # Scaler en el entrenamiento
    import joblib
    from Utils.paths import dynamic_model_keras_path
    scaler_save_path = os.path.join(dynamic_model_keras_path, 'scaler.save')
    joblib.dump(scaler, scaler_save_path)

    print(f"[INFO] Modelo guardado en: {model_path}")

    # Mostrar el resumen del modelo
    print("[INFO] Resumen del modelo:")
    model.summary()

    # Evaluar el modelo
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"[INFO] Evaluación en datos de validación - Pérdida: {val_loss:.4f}, Precisión: {val_acc:.4f}")

    # Verificar las clases del modelo
    expected_classes = set(range(len(word_ids)))
    model_classes = set(np.argmax(y_train, axis=1))
    if expected_classes == model_classes:
        print(
            f"[INFO] Todas las {len(expected_classes)} palabras esperadas están correctamente clasificadas en el modelo.")
    else:
        missing_classes = expected_classes - model_classes
        print(f"[WARNING] Las siguientes palabras no están clasificadas en el modelo: {missing_classes}")

    plot_history(history)


if __name__ == "__main__":
    # Entrenar el modelo y guardar el resultado en la ruta especificada
    train_and_save_model(os.path.join(dynamic_model_keras_path, dynamic_model_name))
