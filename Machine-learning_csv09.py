import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Chargement des données
df = pd.read_csv("MEF_PY_V09_H.csv")

# Nettoyage : suppression des lignes avec valeurs manquantes sur les colonnes clés
df = df.dropna(subset=[
    'Structure_Niv1', 'Structure_Niv2', 'LIBCADEMP',
    'CDLQUA', 'TRANCHE_AGE', 'Sexe_ano', 'Brut', 'Net a payer'
])

# Création de variables dérivées
df['Structure_Combo'] = df['Structure_Niv1'] + "_" + df['Structure_Niv2']
df['Sexe_AGE'] = df['Sexe_ano'] + "_" + df['TRANCHE_AGE']

# Sélection des variables explicatives et cibles
X = df[['Structure_Combo', 'LIBCADEMP', 'CDLQUA', 'TRANCHE_AGE', 'Sexe_AGE']]
y = df[['Brut', 'Net a payer']]

# Colonnes catégorielles
cat_cols = X.select_dtypes(include='object').columns.tolist()

# Prétraitement
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Grille d'hyperparamètres
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2]
}

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Meilleur modèle
model = grid_search.best_estimator_
y_pred = model.predict(X_test)

# Évaluation
print("Meilleurs paramètres :", grid_search.best_params_)
print("R² (test) :", r2_score(y_test, y_pred))
print("RMSE (test) :", np.sqrt(mean_squared_error(y_test, y_pred)))

# Sauvegarde du modèle
joblib.dump(model, "model_random_forest.pkl")
print("Modèle sauvegardé sous 'model_random_forest.pkl'")


