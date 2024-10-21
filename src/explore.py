# scripts/03_explore_data.py

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import load_data, create_directory
from config import PREPROCESSED_DATA_PATH, FIGURES_DIR

def plot_histograms(df, output_dir):
    """Genera histogramas para todas las variables y los guarda en archivos."""
    for column in df.columns:
        plt.figure()
        df[column].hist(bins=20)
        plt.title(f'Histograma de {column}')
        plt.xlabel(column)
        plt.ylabel('Frecuencia')
        plt.savefig(os.path.join(output_dir, f'hist_{column}.png'))
        plt.close()

def plot_boxplots(df, output_dir):
    """Genera boxplots de cada variable vs la variable objetivo y los guarda."""
    for column in df.columns[:-1]:  # Excluye 'Outcome'
        plt.figure()
        sns.boxplot(x='Outcome', y=column, data=df)
        plt.title(f'{column} vs Outcome')
        plt.savefig(os.path.join(output_dir, f'boxplot_{column}.png'))
        plt.close()

def plot_correlation_matrix(df, output_dir):
    """Genera y guarda una matriz de correlación."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

def perform_eda():
    """Realiza el análisis exploratorio de datos (EDA)."""
    # Cargar los datos
    df = load_data(PREPROCESSED_DATA_PATH)

    # Crear el directorio de figuras
    create_directory(FIGURES_DIR)

    # Generar histogramas
    plot_histograms(df, FIGURES_DIR)
    logging.info('Histogramas generados y guardados')

    # Generar boxplots
    plot_boxplots(df, FIGURES_DIR)
    logging.info('Boxplots generados y guardados')

    # Generar matriz de correlación
    plot_correlation_matrix(df, FIGURES_DIR)
    logging.info('Matriz de correlación generada y guardada')

def main():
    """Función principal para ejecutar el análisis exploratorio."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Iniciando el análisis exploratorio de datos')

    perform_eda()

    logging.info('Análisis exploratorio completado')

if __name__ == '__main__':
    main()
