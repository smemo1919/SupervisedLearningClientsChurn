# SupervisedLearningClientsChurn

Proyecto del curso **EPG4001 — Aprendizaje Supervisado** (Magíster en
Inteligencia Artificial, Pontificia Universidad Católica de Chile).

Predicción de **fuga de clientes** (*customer churn*) mediante clasificación
supervisada, aplicando el flujo completo del curso: EDA → preprocesamiento →
modelamiento → evaluación con validación cruzada y métricas → manejo de clases
desbalanceadas.

## Contenido

- **`main.ipynb`** — Notebook principal con el análisis completo (con outputs).
- **`build_notebook.py`** — Script que genera `main.ipynb` celda por celda.
- **`requirements.txt`** — Dependencias de Python.

## Datasets

Se descargan automáticamente con [`kagglehub`](https://github.com/Kaggle/kagglehub)
(cacheados localmente tras la primera ejecución):

| Dataset | Kaggle ID | Rol |
|---------|-----------|-----|
| Customer Churn | `muhammadshahidazeem/customer-churn-dataset` | Principal (train/test provisto) |
| Telco Customer Churn | `blastchar/telco-customer-churn` | Secundario (features ricas + desbalance) |

## Modelos y técnicas

- **Clasificadores:** Regresión Logística, LDA, QDA, Naive Bayes, *k*-NN,
  Árbol de Decisión, Random Forest, SVM (RBF).
- **Evaluación:** *accuracy*, *precision*, *recall*, F1, matriz de confusión,
  curva ROC / AUC, validación cruzada estratificada (5-fold).
- **Desbalance:** `class_weight`, undersampling, oversampling, SMOTE y
  selección de la **probabilidad de corte óptima**.

## Uso

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# registrar el kernel para el notebook
python -m ipykernel install --user --name churn-venv \
  --display-name "Python (SupervisedLearningClientsChurn)"

# opción A: regenerar y ejecutar el notebook desde el script
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=churn-venv main.ipynb

# opción B: abrir de forma interactiva
jupyter lab main.ipynb
```

> **Nota:** el dataset principal tiene cientos de miles de filas; el notebook
> entrena sobre una submuestra estratificada (`SAMPLE_SIZE`) para acotar el
> tiempo de cómputo. Ajusta esa constante en la celda de configuración.
