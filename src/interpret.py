# scripts/05_interpret.py

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from config import (
    PREPROCESSED_DATA_PATH, MODEL_PATH, FIGURES_DIR, CLASSIFICATION_REPORT_PATH
)
from utils import load_data, load_model, create_directory

def plot_confusion_matrix(y_test, y_pred, output_dir):
    """Genera y guarda la matriz de confusión."""
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()
    logging.info('Matriz de confusión generada y guardada')

def plot_feature_importance(model, X, output_dir):
    """Genera y guarda el gráfico de importancia de características."""
    feature_importances = model.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Importancia de Características')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()
    logging.info('Gráfico de importancia de características generado y guardado')

def save_classification_report(y_test, y_pred, report_path):
    """Genera y guarda el reporte de clasificación."""
    report = classification_report(y_test, y_pred)
    logging.info('Reporte de clasificación:')
    logging.info('\n' + report)

    with open(report_path, 'w') as f:
        f.write(report)
    logging.info(f'Reporte de clasificación guardado en {report_path}')

def interpret_results():
    """Realiza la interpretación de los resultados del modelo."""
    # Cargar datos y modelo
    df = load_data(PREPROCESSED_DATA_PATH)
    model = load_model(MODEL_PATH)

    # Separar datos en entrenamiento y prueba
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Crear directorio para las figuras
    create_directory(FIGURES_DIR)

    # Generar y guardar interpretaciones
    plot_confusion_matrix(y_test, y_pred, FIGURES_DIR)
    plot_feature_importance(model, X, FIGURES_DIR)
    save_classification_report(y_test, y_pred, CLASSIFICATION_REPORT_PATH)

def main():
    """Función principal para ejecutar la interpretación."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Iniciando la interpretación de resultados')

    interpret_results()

    logging.info('Interpretación de resultados completada')

if __name__ == '__main__':
    main()
