
import logging
from obtain import main as obtain_data
from scrub import main as scrub_data
from explore import main as explore_data
from model import main as model_data
from interpret import main as interpret_data

def run_pipeline():
    """Ejecuta todos los pasos del flujo de trabajo en orden."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    logging.info('==== INICIANDO FLUJO DE TRABAJO ====')

    try:
        # Paso 1: Obtener los datos
        logging.info('Ejecutando paso 1: Obtener Datos')
        obtain_data()

        # Paso 2: Limpiar y preprocesar los datos
        logging.info('Ejecutando paso 2: Limpieza y Preprocesamiento')
        scrub_data()

        # Paso 3: Exploración de los datos
        logging.info('Ejecutando paso 3: Exploración de Datos')
        explore_data()

        # Paso 4: Modelado de los datos
        logging.info('Ejecutando paso 4: Modelado')
        model_data()

        # Paso 5: Interpretación de los resultados
        logging.info('Ejecutando paso 5: Interpretación de Resultados')
        interpret_data()

        logging.info('==== FLUJO DE TRABAJO COMPLETADO CON ÉXITO ====')

    except Exception as e:
        logging.error(f'Ocurrió un error durante la ejecución: {str(e)}')

if __name__ == '__main__':
    run_pipeline()
