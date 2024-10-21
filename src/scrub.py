# scripts/02_scrub_data.py

import numpy as np
import logging
from utils import load_data, save_dataframe
from config import RAW_DATA_PATH, PREPROCESSED_DATA_PATH

def replace_zero_with_nan(df, columns):
    """Reemplaza los ceros por NaN en las columnas especificadas."""
    for column in columns:
        df[column].replace(0, np.nan, inplace=True)
    return df

def impute_missing_values(df, strategy='median'):
    """Imputa los valores faltantes utilizando la estrategia especificada."""
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy=strategy)
    df[:] = imputer.fit_transform(df)
    return df

def preprocess_data():
    """Función que realiza todo el preprocesamiento del dataset."""
    # Cargar los datos
    df = load_data(RAW_DATA_PATH)

    # Identificar las columnas con ceros que representan valores faltantes
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Reemplazar ceros por NaN
    df = replace_zero_with_nan(df, zero_not_accepted)

    # Imputar los valores faltantes
    df = impute_missing_values(df, strategy='median')

    # Guardar los datos preprocesados
    save_dataframe(df, PREPROCESSED_DATA_PATH)

def main():
    """Función principal que ejecuta la limpieza y preprocesamiento."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Iniciando preprocesamiento de datos')

    preprocess_data()

    logging.info('Preprocesamiento completado')

if __name__ == '__main__':
    main()
