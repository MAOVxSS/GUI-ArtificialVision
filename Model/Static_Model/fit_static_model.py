import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Rutas
from Utils.paths import static_model_data_path, generated_models_path
from Utils.config import static_model_name

# Funciones auxiliares
from Model.model_utils import plot_history

# Configuración de semilla aleatoria
RANDOM_SEED = 42

# Define las rutas para los datos y el modelo
model_save_path = os.path.join(generated_models_path, static_model_name)
NUM_CLASSES = 21  # Letras

# Lectura de datos
X_dataset = np.loadtxt(static_model_data_path, delimiter=',', dtype='float32',
                       usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(static_model_data_path, delimiter=',', dtype='int32', usecols=(0))

# Verificar el balance de los datos
print("Distribución de las etiquetas en el conjunto de datos:")
print(np.unique(y_dataset, return_counts=True))

# Asegurarse de que se hayan leído todos los datos
print(f"Dimensiones de X_dataset: {X_dataset.shape}")
print(f"Dimensiones de y_dataset: {y_dataset.shape}")

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.80, random_state=RANDOM_SEED)

# Construcción del modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Callbacks
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False,
                                                 save_best_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=70, verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')

# Compilación del modelo
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Definir la tasa de aprendizaje
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_data=(X_test, y_test),
                    callbacks=[cp_callback, es_callback, tensorboard_callback])

# Evaluación del modelo
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)
print(f'Validation Loss: {val_loss}, Validation Accuracy: {val_acc}')

# Guardar modelo en formato .keras
model.save(model_save_path)
print(f'Model saved to {model_save_path}')

# Verificar si el modelo fue guardado correctamente
if os.path.exists(model_save_path):
    print(f"Model successfully saved at {model_save_path}")
else:
    print("Model saving failed")

# Llamada a la función de graficar
plot_history(history)
