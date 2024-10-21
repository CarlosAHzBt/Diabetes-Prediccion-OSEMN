# Diabetes Binary Prediction

## Descripción del Proyecto

Este proyecto es un estudio enfocado en la predicción de si un paciente desarrollará diabetes. El principal objetivo es aplicar la metodología OSEMN pipeline para estructurar el flujo de trabajo de análisis de datos.

Pima Indians Diabetes Database
El dataset corresponde al siguiente enlace [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database]
## Metodología OSEMN

La metodología OSEMN se compone de las siguientes etapas:

1. **Obtain (Obtener)**: Recolección de datos.
2. **Scrub (Limpiar)**: Limpieza y preparación de los datos.
3. **Explore (Explorar)**: Análisis exploratorio de los datos.
4. **Model (Modelar)**: Construcción y evaluación de modelos predictivos.
5. **Interpret (Interpretar)**: Interpretación de los resultados.

## Automatización con Archivos .py

A diferencia de los notebooks, este proyecto utiliza archivos `.py` para facilitar la automatización de la ejecución del código. Esto permite una integración más sencilla en pipelines de CI/CD y una mayor reproducibilidad de los resultados.

## Estructura del Proyecto

- `data/`: Contiene los datos utilizados para el análisis.
- `scripts/`: Scripts en Python para cada etapa de la metodología OSEMN.
- `models/`: Modelos entrenados y sus evaluaciones.
- `results/`: Resultados y visualizaciones generadas.

## Requisitos

- Python 3.x
- Bibliotecas: pandas, numpy, scikit-learn, matplotlib, seaborn

## Ejecución

Para ejecutar el proyecto, sigue estos pasos:

1. Clona el repositorio.
2. Instala las dependencias necesarias.
3. Ejecuta los scripts en el orden adecuado.

```bash
git clone <URL_DEL_REPOSITORIO>
cd Diabetes-binary-prediction
pip install -r requirements.txt
python scripts/obtain.py
python scripts/scrub.py
python scripts/explore.py
python scripts/model.py
python scripts/interpret.py
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o un pull request para discutir cualquier cambio.


## Rendimiento del Modelo

A continuación se muestra el rendimiento del modelo cargado en esta practica en términos de precisión, recall y f1-score:

```
              precision    recall  f1-score   support

           0       0.80      0.84      0.82       100
           1       0.67      0.61      0.64        54

    accuracy                           0.76       154
   macro avg       0.74      0.73      0.73       154
weighted avg       0.76      0.76      0.76       154
```

### Conclusión del Reporte
Clase desbalanceada: La clase 0 (sin diabetes) tiene más ejemplos que la clase 1 (con diabetes). Esto puede afectar el rendimiento del modelo, ya que es más difícil para el modelo detectar correctamente los casos de clase 1 (diabetes).

Precision vs Recall:

Para la clase 1, la precisión (0.67) es mayor que el recall (0.61). Esto significa que, aunque la mayoría de las predicciones positivas de diabetes fueron correctas, el modelo no identificó todos los casos reales de diabetes, lo que podría ser preocupante en un contexto médico.
F1-Score más bajo para la clase 1: Indica que el modelo tiene dificultades para equilibrar precisión y recall para esta clase. En el contexto de salud, donde es crucial minimizar los falsos negativos (personas con diabetes no diagnosticadas), sería necesario mejorar el recall de esta clase.

Accuracy Global (0.76): Aunque el modelo tiene un rendimiento aceptable en términos de precisión general, es importante considerar que la accuracy puede ser engañosa en problemas con clases desbalanceadas.
