# scripts/utils.py

import os
import logging
import pandas as pd
from joblib import dump,load

def create_directory(path):
    """Crea un directorio si no existe."""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Directorio creado: {path}")
    else:
        logging.info(f"El directorio ya existe: {path}")


def load_data(filepath):
    """Carga un conjunto de datos desde el archivo especificado."""
    if not os.path.exists(filepath):
        logging.error(f'El archivo no existe: {filepath}')
        raise FileNotFoundError(f'El archivo no existe: {filepath}')
    
    logging.info(f'Cargando datos desde {filepath}')
    return pd.read_csv(filepath)

def save_dataframe(df, filepath):
    """Guarda un DataFrame en un archivo CSV."""
    output_dir = os.path.dirname(filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(filepath, index=False)
    logging.info(f'Datos guardados en: {filepath}')
    
    
def save_model(model, filepath):
    """Guarda el modelo entrenado en la ruta especificada."""
    output_dir = os.path.dirname(filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dump(model, filepath)
    logging.info(f'Modelo guardado en: {filepath}')
    
def load_model(filepath):
    """Carga un modelo guardado desde un archivo."""
    if not os.path.exists(filepath):
        logging.error(f'El modelo no existe: {filepath}')
        raise FileNotFoundError(f'El modelo no existe: {filepath}')
    logging.info(f'Cargando modelo desde {filepath}')
    return load(filepath)