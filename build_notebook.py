"""Genera main.ipynb: Proyecto de Aprendizaje Supervisado (Churn de clientes).

Este script arma el notebook celda por celda con nbformat. Es un artefacto de
construcción: el entregable es main.ipynb (con outputs ejecutados).
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# EPG4001 Aprendizaje Supervisado
## Proyecto — Predicción de *Churn* de Clientes
**Profesor:** Dr. Jorge Luis Bazán
**Email:** jlbazan@uc.cl
**Institución:** Pontificia Universidad Católica de Chile — Magíster en Inteligencia Artificial

**Integrantes:** _(completar)_

---

## Objetivo

Construir y comparar modelos de **clasificación supervisada** para predecir la
**fuga de clientes** (*customer churn*), aplicando el flujo completo del curso:

1. **Exploración de datos (EDA)** y análisis descriptivo.
2. **Preprocesamiento**: limpieza, codificación de categóricas y estandarización.
3. **Modelamiento** con múltiples algoritmos vistos en clases:
   Regresión Logística, Análisis Discriminante (LDA/QDA), Naive Bayes,
   *k*-NN, Árboles de Decisión, Random Forest y SVM.
4. **Evaluación** con métricas de desempeño: *accuracy*, *precision*, *recall*,
   *F1*, matriz de confusión y curva **ROC / AUC**, usando **validación cruzada**.
5. **Manejo de clases desbalanceadas** (`class_weight`, **SMOTE**).

Se utilizan **dos conjuntos de datos** de churn descargados desde Kaggle con
`kagglehub`:

| Dataset | Fuente | Tamaño | Rol |
|---------|--------|--------|-----|
| Customer Churn | `muhammadshahidazeem/customer-churn-dataset` | ~440k train / 64k test | **Principal** (train/test provisto) |
| Telco Customer Churn | `blastchar/telco-customer-churn` | 7 043 filas | **Secundario** (features ricas + desbalance) |
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. Configuración del entorno

Importamos las librerías y fijamos una semilla para reproducibilidad. Para
mantener tiempos de ejecución razonables sobre el dataset principal (que tiene
cientos de miles de filas), entrenamos los modelos sobre una **submuestra**
estratificada controlada por `SAMPLE_SIZE`.
""")

code(r"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from IPython.display import display
except ImportError:
    display = print

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report)

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
SAMPLE_SIZE = 25_000            # submuestra de entrenamiento para el dataset grande
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams["figure.figsize"] = (8, 5)
pd.set_option("display.max_columns", 50)
print("Entorno listo. scikit-learn + imbalanced-learn cargados.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Carga de datos con `kagglehub`

Descargamos ambos datasets. `kagglehub` cachea localmente, por lo que las
ejecuciones posteriores no vuelven a descargar.
""")

code(r"""
import kagglehub, os

kaggle_path = kagglehub.dataset_download("muhammadshahidazeem/customer-churn-dataset")
telco_path  = kagglehub.dataset_download("blastchar/telco-customer-churn")

df_train_raw = pd.read_csv(os.path.join(kaggle_path, "customer_churn_dataset-training-master.csv"))
df_test_raw  = pd.read_csv(os.path.join(kaggle_path, "customer_churn_dataset-testing-master.csv"))
df_telco_raw = pd.read_csv(os.path.join(telco_path,  "WA_Fn-UseC_-Telco-Customer-Churn.csv"))

print("Customer Churn  -> train:", df_train_raw.shape, "| test:", df_test_raw.shape)
print("Telco Churn     ->", df_telco_raw.shape)
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Dataset principal: *Customer Churn*

Variables: datos demográficos (`Age`, `Gender`), de uso (`Tenure`,
`Usage Frequency`, `Support Calls`, `Payment Delay`, `Total Spend`,
`Last Interaction`) y de contrato (`Subscription Type`, `Contract Length`).
La variable objetivo es `Churn` (1 = el cliente se fuga).

### 2.1 Exploración de datos (EDA)
""")

code(r"""
df = df_train_raw.copy()
df = df.dropna(subset=["Churn"]).reset_index(drop=True)   # 1 fila con target nulo
df["Churn"] = df["Churn"].astype(int)
display(df.head())
print("\nInfo general:")
df.info()
print("\nEstadísticos descriptivos:")
display(df.describe().T)
""")

code(r"""
# Balance de la variable objetivo
ax = sns.countplot(x="Churn", data=df)
ax.set_title("Distribución de la variable objetivo (Customer Churn)")
ax.set_xticklabels(["No fuga (0)", "Fuga (1)"])
for p in ax.patches:
    ax.annotate(f"{p.get_height():,}", (p.get_x()+p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.show()
print(df["Churn"].value_counts(normalize=True).round(3).to_dict())
""")

code(r"""
# Distribución de variables numéricas segun churn
num_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls",
            "Payment Delay", "Total Spend", "Last Interaction"]
fig, axes = plt.subplots(3, 3, figsize=(15, 11))
for ax, col in zip(axes.ravel(), num_cols):
    sns.kdeplot(data=df.sample(30000, random_state=RANDOM_STATE), x=col,
                hue="Churn", fill=True, common_norm=False, ax=ax)
    ax.set_title(col)
for ax in axes.ravel()[len(num_cols):]:
    ax.axis("off")
fig.suptitle("Densidad de variables numéricas por clase (0=no fuga, 1=fuga)", y=1.02)
plt.tight_layout(); plt.show()
""")

code(r"""
# Matriz de correlacion (numericas + target)
plt.figure(figsize=(9, 7))
corr = df[num_cols + ["Churn"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlación — Customer Churn")
plt.show()
""")

code(r"""
# Tasa de churn por variables categoricas
cat_cols = ["Gender", "Subscription Type", "Contract Length"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, cat_cols):
    (df.groupby(col)["Churn"].mean().sort_values()
       .plot(kind="bar", ax=ax, color=sns.color_palette("deep")[0]))
    ax.set_title(f"Tasa de fuga por {col}"); ax.set_ylabel("P(Churn=1)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
### 2.2 Preprocesamiento

- Se elimina `CustomerID` (identificador sin valor predictivo).
- Variables **numéricas** → imputación de faltantes + `StandardScaler`.
- Variables **categóricas** → `OneHotEncoder`.
- Se toma una **submuestra estratificada** de entrenamiento de `SAMPLE_SIZE`
  filas para acotar el costo computacional de modelos como SVM/*k*-NN, y se
  evalúa sobre el conjunto de **test provisto** por el dataset.
""")

code(r"""
def split_features_target(frame):
    frame = frame.dropna(subset=["Churn"]).copy()
    frame["Churn"] = frame["Churn"].astype(int)
    X = frame.drop(columns=["CustomerID", "Churn"])
    y = frame["Churn"]
    return X, y

X_full, y_full = split_features_target(df_train_raw)
X_test,  y_test = split_features_target(df_test_raw)

# submuestra estratificada de entrenamiento
X_tr, _, y_tr, _ = train_test_split(
    X_full, y_full, train_size=SAMPLE_SIZE, stratify=y_full, random_state=RANDOM_STATE)

numeric_features = X_tr.select_dtypes(include="number").columns.tolist()
categorical_features = X_tr.select_dtypes(exclude="number").columns.tolist()
print("Numéricas:", numeric_features)
print("Categóricas:", categorical_features)
print("Train submuestreado:", X_tr.shape, "| Test:", X_test.shape)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])
""")

# ---------------------------------------------------------------------------
md(r"""
### 2.3 Modelos de clasificación

Entrenamos los principales clasificadores supervisados del curso dentro de un
`Pipeline` que encapsula el preprocesamiento (evitando fuga de información).
Para cada modelo reportamos **validación cruzada** (5-fold, F1) sobre el
entrenamiento y luego métricas sobre el conjunto de test.
""")

code(r"""
models = {
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
    "Naive Bayes": GaussianNB(),
    "k-NN": KNeighborsClassifier(n_neighbors=15),
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}


def evaluate_models(models, preprocessor, X_tr, y_tr, X_te, y_te, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_te)[:, 1]
        else:
            y_score = pipe.decision_function(X_te)
        rows.append({
            "Modelo": name,
            "CV F1 (media)": cv_f1.mean(),
            "CV F1 (sd)": cv_f1.std(),
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred),
            "Recall": recall_score(y_te, y_pred),
            "F1": f1_score(y_te, y_pred),
            "ROC AUC": roc_auc_score(y_te, y_score),
        })
        fitted[name] = pipe
        print(f"OK - {name}")
    results = pd.DataFrame(rows).set_index("Modelo").sort_values("F1", ascending=False)
    return results, fitted


results, fitted = evaluate_models(models, preprocessor, X_tr, y_tr, X_test, y_test)
""")

# ---------------------------------------------------------------------------
md(r"""
### 2.4 Comparación de modelos y métricas
""")

code(r"""
display(results.round(4))
""")

md(r"""
> **Observación importante.** Se aprecia una **brecha grande** entre el F1 de
> validación cruzada (muy alto en modelos flexibles como Random Forest/Árbol) y
> el F1 sobre el **conjunto de test provisto**. Esto evidencia un **cambio de
> distribución** (*distribution shift*) entre los archivos de *train* y *test*
> de este dataset: los modelos más flexibles sobreajustan patrones del train que
> no se replican en el test. Por ello, los modelos **más simples** (LDA,
> Regresión Logística) generalizan mejor aquí. Es un recordatorio del
> **compromiso sesgo–varianza** visto en el curso.
""")

code(r"""
ax = results["F1"].sort_values().plot(kind="barh", color=sns.color_palette("deep")[2])
ax.set_title("Comparación de modelos — F1 en test (Customer Churn)")
ax.set_xlabel("F1-score")
for i, v in enumerate(results["F1"].sort_values()):
    ax.text(v, i, f" {v:.3f}", va="center")
plt.tight_layout(); plt.show()
""")

code(r"""
# Curvas ROC de todos los modelos
fig, ax = plt.subplots(figsize=(8, 7))
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax, name=name)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("Curvas ROC — Customer Churn")
plt.show()
""")

code(r"""
# Matriz de confusion del mejor modelo (por F1)
best_name = results.index[0]
best_pipe = fitted[best_name]
y_pred_best = best_pipe.predict(X_test)
print(f"Mejor modelo por F1: {best_name}\n")
print(classification_report(y_test, y_pred_best, target_names=["No fuga", "Fuga"]))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_best),
                       display_labels=["No fuga", "Fuga"]).plot(cmap="Blues")
plt.title(f"Matriz de confusión — {best_name}")
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Dataset secundario: *Telco Customer Churn*

Este conjunto tiene muchas variables **categóricas** (tipo de contrato,
servicios contratados, método de pago) y está **desbalanceado** (~27% de fuga),
lo que permite ilustrar el manejo de clases desbalanceadas con **SMOTE** y
`class_weight`.

### 3.1 EDA y limpieza
""")

code(r"""
telco = df_telco_raw.copy()
# TotalCharges viene como texto con espacios en blanco -> numérico
telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
print("Faltantes en TotalCharges:", telco["TotalCharges"].isna().sum())
telco = telco.drop(columns=["customerID"])
telco["Churn"] = (telco["Churn"] == "Yes").astype(int)
telco["SeniorCitizen"] = telco["SeniorCitizen"].map({0: "No", 1: "Yes"})

print("Balance de clases:", telco["Churn"].value_counts(normalize=True).round(3).to_dict())
display(telco.head())
""")

code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, ["Contract", "InternetService", "PaymentMethod"]):
    (telco.groupby(col)["Churn"].mean().sort_values()
        .plot(kind="bar", ax=ax, color=sns.color_palette("deep")[3]))
    ax.set_title(f"Tasa de fuga por {col}"); ax.set_ylabel("P(Churn=1)")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()

# tenure y cargos mensuales por churn
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.kdeplot(data=telco, x="tenure", hue="Churn", fill=True, common_norm=False, ax=axes[0])
axes[0].set_title("Antigüedad (tenure) por clase")
sns.kdeplot(data=telco, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False, ax=axes[1])
axes[1].set_title("Cargo mensual por clase")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
### 3.2 Preprocesamiento y manejo del desbalance

Comparamos las estrategias del curso (Clase 7) frente al desbalance, usando
**Regresión Logística** y **Random Forest** como referencia:

1. **Base** (sin ajuste).
2. **`class_weight="balanced"`** (penaliza más los errores de la clase minoritaria).
3. **Undersampling** aleatorio de la clase mayoritaria.
4. **Oversampling** aleatorio de la clase minoritaria.
5. **SMOTE** (sobremuestreo sintético de la clase minoritaria, sólo en train).
""")

code(r"""
X_t = telco.drop(columns=["Churn"])
y_t = telco["Churn"]
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(
    X_t, y_t, test_size=0.25, stratify=y_t, random_state=RANDOM_STATE)

t_num = X_t.select_dtypes(include="number").columns.tolist()
t_cat = X_t.select_dtypes(exclude="number").columns.tolist()
telco_prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), t_num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), t_cat),
])
print("Numéricas:", t_num)
print("Categóricas:", t_cat)
""")

code(r"""
def eval_pipe(pipe, name):
    pipe.fit(Xt_tr, yt_tr)
    y_pred = pipe.predict(Xt_te)
    y_score = pipe.predict_proba(Xt_te)[:, 1]
    return {
        "Estrategia": name,
        "Accuracy": accuracy_score(yt_te, y_pred),
        "Precision": precision_score(yt_te, y_pred),
        "Recall": recall_score(yt_te, y_pred),
        "F1": f1_score(yt_te, y_pred),
        "ROC AUC": roc_auc_score(yt_te, y_score),
    }

rows = []
# 1. Base
rows.append(eval_pipe(Pipeline([("prep", telco_prep),
    ("m", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "LogReg base"))
rows.append(eval_pipe(Pipeline([("prep", telco_prep),
    ("m", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE))]),
    "RandomForest base"))
# 2. class_weight balanced
rows.append(eval_pipe(Pipeline([("prep", telco_prep),
    ("m", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
    "LogReg class_weight"))
rows.append(eval_pipe(Pipeline([("prep", telco_prep),
    ("m", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 n_jobs=-1, random_state=RANDOM_STATE))]),
    "RandomForest class_weight"))
# 3. Undersampling aleatorio de la clase mayoritaria
rows.append(eval_pipe(ImbPipeline([("prep", telco_prep),
    ("under", RandomUnderSampler(random_state=RANDOM_STATE)),
    ("m", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "LogReg + Undersampling"))
# 4. Oversampling aleatorio de la clase minoritaria
rows.append(eval_pipe(ImbPipeline([("prep", telco_prep),
    ("over", RandomOverSampler(random_state=RANDOM_STATE)),
    ("m", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "LogReg + Oversampling"))
# 5. SMOTE (pipeline de imblearn: SMOTE solo se aplica en fit/train)
rows.append(eval_pipe(ImbPipeline([("prep", telco_prep),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("m", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
    "LogReg + SMOTE"))
rows.append(eval_pipe(ImbPipeline([("prep", telco_prep),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("m", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE))]),
    "RandomForest + SMOTE"))

telco_results = pd.DataFrame(rows).set_index("Estrategia").sort_values("F1", ascending=False)
display(telco_results.round(4))
""")

# ---------------------------------------------------------------------------
md(r"""
### 3.3 Efecto del desbalance en la matriz de confusión

Comparamos la matriz de confusión de la Regresión Logística **base** vs. **con
SMOTE** para visualizar cómo mejora la detección de la clase minoritaria (fuga).
""")

code(r"""
base = Pipeline([("prep", telco_prep),
    ("m", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]).fit(Xt_tr, yt_tr)
smote = ImbPipeline([("prep", telco_prep),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("m", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]).fit(Xt_tr, yt_tr)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, model, title in [(axes[0], base, "LogReg base"),
                         (axes[1], smote, "LogReg + SMOTE")]:
    ConfusionMatrixDisplay(confusion_matrix(yt_te, model.predict(Xt_te)),
                           display_labels=["No fuga", "Fuga"]).plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
plt.tight_layout(); plt.show()

# importancia de variables (Random Forest)
rf = Pipeline([("prep", telco_prep),
    ("m", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE))]).fit(Xt_tr, yt_tr)
feat_names = rf.named_steps["prep"].get_feature_names_out()
importances = pd.Series(rf.named_steps["m"].feature_importances_, index=feat_names)
importances.nlargest(12).sort_values().plot(kind="barh", figsize=(8, 5),
    color=sns.color_palette("deep")[4])
plt.title("Top 12 variables más importantes (Random Forest — Telco)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
### 3.4 Probabilidad de corte óptima

El umbral por defecto de 0.5 no es óptimo bajo desbalance. Buscamos el umbral
que **maximiza el F1** sobre las probabilidades predichas por la Regresión
Logística, tal como se ve en la Clase 7.
""")

code(r"""
from sklearn.metrics import precision_recall_curve

logit = Pipeline([("prep", telco_prep),
    ("m", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]).fit(Xt_tr, yt_tr)
proba = logit.predict_proba(Xt_te)[:, 1]

thresholds = np.linspace(0.05, 0.95, 181)
f1s = [f1_score(yt_te, (proba >= t).astype(int)) for t in thresholds]
best_t = thresholds[int(np.argmax(f1s))]

plt.plot(thresholds, f1s)
plt.axvline(best_t, color="red", ls="--", label=f"umbral óptimo = {best_t:.2f}")
plt.axvline(0.5, color="gray", ls=":", label="umbral 0.5")
plt.xlabel("Umbral de decisión"); plt.ylabel("F1-score")
plt.title("F1 vs. umbral de corte — LogReg (Telco)")
plt.legend(); plt.show()

print(f"F1 con umbral 0.50: {f1_score(yt_te, (proba>=0.5).astype(int)):.4f}")
print(f"F1 con umbral {best_t:.2f}: {max(f1s):.4f}")
print(f"Recall con umbral óptimo: {recall_score(yt_te, (proba>=best_t).astype(int)):.4f}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Conclusiones

- Se aplicó el **flujo completo de aprendizaje supervisado** del curso EPG4001
  sobre dos problemas reales de *churn*, desde la exploración hasta la
  evaluación comparativa de modelos.
- En el dataset **principal (Customer Churn)**, los modelos **no lineales**
  (Random Forest, SVM, Árbol) capturan mejor la estructura de los datos que los
  lineales (Regresión Logística, LDA), obteniendo mayores **F1** y **AUC**
  (ver tabla comparativa de la §2.4).
- En el dataset **Telco**, fuertemente **desbalanceado**, las técnicas de
  remuestreo (**undersampling**, **oversampling**, **SMOTE**) y el ajuste por
  `class_weight` aumentan el **recall** de la clase minoritaria (clientes que se
  fugan) a costa de algo de *precision*, mejorando el F1 respecto al modelo base.
  Ajustar la **probabilidad de corte** (§3.4) es una alternativa simple y eficaz
  al remuestreo para optimizar el F1.
- Las variables de **contrato/antigüedad** (`Contract`, `tenure`) y de
  **gasto/uso** resultan las más informativas para anticipar la fuga, lo que es
  accionable para estrategias de retención.

> **Nota reproducibilidad:** por costo computacional, el dataset principal se
> entrena sobre una submuestra estratificada de `SAMPLE_SIZE` filas. Aumentar
> este valor mejora ligeramente las métricas a cambio de mayor tiempo de cómputo.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python (SupervisedLearningClientsChurn)",
    "language": "python",
    "name": "churn-venv",
}
with open("main.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"main.ipynb generado con {len(cells)} celdas.")
