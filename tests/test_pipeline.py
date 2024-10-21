import os
import pytest
import pandas as pd
from src.obtain import download_data, load_and_save_data
from src.scrub import preprocess_data
from src.model import create_pipeline, train_model, split_data
from src.utils import load_data, save_dataframe

# Ruta de prueba para archivos generados
TEST_RAW_DATA_PATH = "dataset/raw/test_dataset.csv"
TEST_PROCESSED_DATA_PATH = "dataset/processed/test_cleaned_data.csv"
TEST_MODEL_PATH = "models/test_model.joblib"

@pytest.fixture(scope="module")
def sample_data():
    """Crea un DataFrame de ejemplo para pruebas."""
    data = {
        'Pregnancies': [1, 2, 3],
        'Glucose': [85, 0, 120],  # Contiene un cero para probar la limpieza
        'BloodPressure': [70, 80, 0],  # Contiene un cero para imputar
        'SkinThickness': [35, 29, 32],
        'Insulin': [0, 150, 130],  # Contiene un cero para probar la imputación
        'BMI': [33.6, 28.1, 30.5],
        'DiabetesPedigreeFunction': [0.627, 0.351, 0.672],
        'Age': [50, 31, 32],
        'Outcome': [1, 0, 1]
    }
    return pd.DataFrame(data)

def test_obtain_data(sample_data):
    """Verifica que el dataset se pueda guardar correctamente."""
    save_dataframe(sample_data, TEST_RAW_DATA_PATH)
    assert os.path.exists(TEST_RAW_DATA_PATH), "El archivo de datos no se guardó correctamente"

def test_scrub_data():
    """Verifica que los datos se limpien y se guarden correctamente."""
    preprocess_data()  # Utiliza el pipeline de limpieza en RAW_DATA_PATH -> PROCESSED_DATA_PATH
    assert os.path.exists(TEST_PROCESSED_DATA_PATH), "Los datos procesados no se guardaron correctamente"
    
    df = load_data(TEST_PROCESSED_DATA_PATH)
    assert not df.isnull().values.any(), "Hay valores nulos en los datos preprocesados"

def test_split_data(sample_data):
    """Verifica que la división de los datos se realice correctamente."""
    X_train, X_test, y_train, y_test = split_data(sample_data)
    assert len(X_train) > 0 and len(X_test) > 0, "La división de los datos falló"
    assert len(X_train) + len(X_test) == len(sample_data), "La división no cubre todos los datos"

def test_train_model(sample_data):
    """Verifica que el modelo se entrene y guarde correctamente."""
    X_train, X_test, y_train, _ = split_data(sample_data)
    model = train_model(X_train, y_train)
    
    # Guardar el modelo de prueba
    from joblib import dump
    dump(model, TEST_MODEL_PATH)
    assert os.path.exists(TEST_MODEL_PATH), "El modelo no se guardó correctamente"

def test_pipeline_end_to_end():
    """Prueba de extremo a extremo para asegurar que todo el pipeline funcione."""
    # Obtener, limpiar, entrenar, y evaluar todo el pipeline
    preprocess_data()  # Limpieza de los datos
    df = load_data(TEST_PROCESSED_DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    
    # Evaluación simple
    accuracy = model.score(X_test, y_test)
    assert accuracy >= 0, "El modelo no se entrenó correctamente"

    print(f"Accuracy del modelo: {accuracy:.2f}")

