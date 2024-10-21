# scripts/config.py

# Rutas de archivos
RAW_DATA_PATH  = 'dataset/raw/dataset.csv'
PREPROCESSED_DATA_PATH = 'dataset/preprocessed/dataset_clean.csv'
FIGURES_DIR  = 'reports/figures'
MODEL_PATH = 'models/best_model.joblib'
CLASSIFICATION_REPORT_PATH = 'reports/classification_report.txt'

# Hiperparámetros para el modelo dentro del pipeline
MODEL_PARAMS = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0]
}