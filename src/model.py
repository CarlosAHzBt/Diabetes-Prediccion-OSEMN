# scripts/04_model.py

import logging
from config import PREPROCESSED_DATA_PATH, MODEL_PATH, MODEL_PARAMS
from utils import load_data, save_model
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def create_pipeline():
    """Crea el pipeline de preprocesamiento y modelo."""
    return Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(random_state=42))
    ])

def split_data(df):
    """Separa los datos en entrenamiento y prueba."""
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def train_model(X_train, y_train):
    """Entrena el modelo utilizando GridSearchCV."""
    pipeline = create_pipeline()

    # Validación cruzada inicial
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    logging.info(f'Accuracy promedio en validación cruzada inicial: {cv_scores.mean():.4f}')

    # Ajuste de hiperparámetros
    grid_search = GridSearchCV(pipeline, MODEL_PARAMS, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logging.info(f'Mejores hiperparámetros: {grid_search.best_params_}')
    logging.info(f'Mejor accuracy en validación cruzada: {grid_search.best_score_:.4f}')

    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo en el conjunto de prueba y muestra las métricas."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f'Accuracy en el conjunto de prueba: {accuracy:.4f}')
    logging.info('Reporte de clasificación:')
    logging.info('\n' + classification_report(y_test, y_pred))

def main():
    """Función principal para ejecutar el modelado."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Iniciando el modelado')

    # Cargar los datos
    df = load_data(PREPROCESSED_DATA_PATH)

    # Separar los datos
    X_train, X_test, y_train, y_test = split_data(df)

    # Entrenar el modelo
    best_model = train_model(X_train, y_train)

    # Evaluar el modelo
    evaluate_model(best_model, X_test, y_test)

    # Guardar el modelo entrenado
    save_model(best_model, MODEL_PATH)

    logging.info('Modelado completado')

if __name__ == '__main__':
    main()
