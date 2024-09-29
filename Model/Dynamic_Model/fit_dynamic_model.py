import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2

# Rutas y variables
from Utils.paths import generated_models_path, dynamic_model_converted_data_path
from Utils.config import dynamic_model_name

# Funciones auxiliares
from Model.model_utils import plot_history

# Configuración
num_epoch = 100  # Número de épocas para entrenar el modelo
batch_size = 64  # Muestras por iteración
max_length_frames = 15  # Cantidad de frames maxima para entrenamiento
length_keypoints = 63  # Longitud de los key points


# Crea y compila el modelo dinámico
def get_model(max_length_frames, output_length: int):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(max_length_frames, length_keypoints),
                   kernel_regularizer=l2(0.001)))
    model.add(LSTM(128, return_sequences=True, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(LSTM(128, return_sequences=False, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(output_length, activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Entrena el modelo dinámico
def training_model(data_path, dynamic_model_path):
    actions = [os.path.splitext(action)[0] for action in os.listdir(data_path) if os.path.splitext(action)[1] == ".h5"]
    sequences, labels = [], []

    for label, action in enumerate(actions):
        hdf_path = os.path.join(data_path, f"{action}.h5")
        data = pd.read_hdf(hdf_path, key='data')
        for _, data_filtered in data.groupby('sample'):
            sequences.append([fila['keypoints'] for _, fila in data_filtered.iterrows()])
            labels.append(label)

    sequences = pad_sequences(sequences, maxlen=max_length_frames, padding='post', truncating='post', dtype='float32')
    x = np.array(sequences)
    y = to_categorical(labels).astype(int)

    print("Distribución de las etiquetas en el conjunto de datos:")
    print(np.unique(labels, return_counts=True))

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80)

    model = get_model(max_length_frames, len(actions))

    cp_callback = ModelCheckpoint(dynamic_model_path, verbose=1, save_weights_only=False, save_best_only=True)
    es_callback = EarlyStopping(patience=20, verbose=1, restore_best_weights=True)
    tensorboard_callback = TensorBoard(log_dir='./logs')

    history = model.fit(x_train, y_train, epochs=num_epoch, batch_size=batch_size, validation_data=(x_test, y_test),
                        callbacks=[cp_callback, es_callback, tensorboard_callback])

    val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

    model.save(dynamic_model_path)
    print(f'Model saved to {dynamic_model_path}')

    if os.path.exists(dynamic_model_path):
        print(f"Model successfully saved at {dynamic_model_path}")
    else:
        print("Model saving failed")

    plot_history(history)


if __name__ == "__main__":
    model_path = os.path.join(generated_models_path, dynamic_model_name)
    if not os.path.exists(dynamic_model_converted_data_path):
        print(f"Error: the path {dynamic_model_converted_data_path} does not exist.")
        exit(1)
    training_model(dynamic_model_converted_data_path, model_path)
