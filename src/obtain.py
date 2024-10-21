# scripts/01_obtain_data.py

import pandas as pd
import kagglehub
import logging
from config import RAW_DATA_PATH  # Importamos la ruta desde config.py
from utils import create_directory  # Reutilizamos la función create_directory

def download_data():
    """Descarga los datos desde Kaggle y retorna la ruta."""
    logging.info("Descargando dataset desde Kaggle...")
    path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
    logging.info(f"Dataset descargado en: {path}")
    return path

def load_and_save_data(download_path, save_path):
    """Carga los datos descargados y los guarda en la ruta especificada."""
    df = pd.read_csv(f"{download_path}/diabetes.csv")
    create_directory('dataset/raw')  # Asegura que el directorio exista
    df.to_csv(save_path, index=False)
    logging.info(f"Datos guardados en: {save_path}")
    return df

def main():
    """Función principal para obtener y guardar los datos."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Iniciando proceso de obtención de datos")

    # Descargar y guardar los datos
    download_path = download_data()
    df = load_and_save_data(download_path, RAW_DATA_PATH)

    # Imprimir información básica
    logging.info(f"Primeras filas del dataset:\n{df.head()}")
    logging.info(f"Tamaño del dataset: {df.shape}")

if __name__ == "__main__":
    main()
