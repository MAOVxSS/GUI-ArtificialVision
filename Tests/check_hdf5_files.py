import os
from Model.model_utils import analyze_h5_keypoints
from Utils.paths import phrases_model_converted_data_path
# Script principal para analizar todos los archivos HDF5 en una carpeta
if __name__ == "__main__":
    """
    Itera sobre todos los archivos HDF5 en una carpeta y los analiza.
    """
    # Carpeta donde se encuentran los archivos HDF5

    if not os.path.exists(phrases_model_converted_data_path):
        print(f"[ERROR] La carpeta especificada no existe: {phrases_model_converted_data_path}")
    else:
        hdf5_files = [os.path.join(phrases_model_converted_data_path, f) for f in os.listdir(phrases_model_converted_data_path) if f.endswith('.h5')]

        if not hdf5_files:
            print(f"[ERROR] No se encontraron archivos HDF5 en la carpeta: {phrases_model_converted_data_path}")
        else:
            print(f"[INFO] Se encontraron {len(hdf5_files)} archivos HDF5. Iniciando análisis...\n")
            for h5_file in hdf5_files:
                analyze_h5_keypoints(h5_file)
