# Prédiction de la Masse Salariale (Fonction Publique)

## Objectif de la webapp

Cette application web permet d'estimer la masse salariale brute et nette à payer du mois suivant pour les agents de la fonction publique. 
En sélectionnant plusieurs critères liés à la structure administrative, au poste, au type de contrat, à la tranche d’âge et au sexe, l’utilisateur obtient une prédiction personnalisée et une visualisation de l’historique salarial.

---

## Choix du dataset

Le dataset utilisé est le fichier **`MEF_PY_V09_H.csv`**, qui contient les données salariales historiques des agents, notamment :
- Structure administrative (niveau 1 et 2)
- Poste (`LIBCADEMP`)
- Type de contrat (`CDLQUA`)
- Tranche d’âge (`TRANCHE_AGE`)
- Sexe (`Sexe_ano`)
- Salaires brut et net
- Dates de paiement (`MOIPAI`)

Le dataset, constitué de données réelles, a d'abord été anonymisé, puis nettoyé afin de supprimer les lignes contenant des valeurs manquantes sur les variables clés.

---

## Choix du modèle

Un modèle **Random Forest Regressor** est utilisé pour la prédiction, car il :
- Gère efficacement les variables catégorielles après encodage
- Supporte la non-linéarité et la complexité des données salariales
- Offre de bonnes performances de généralisation.

Le modèle est intégré dans un pipeline comprenant :
- Un préprocesseur avec `OneHotEncoder` pour les variables catégorielles
- Une recherche d’hyperparamètres (`GridSearchCV`) pour optimiser les paramètres du Random Forest (nombre d’arbres, profondeur, etc.)
- Une validation croisée `5-fold` pour nous assurer de la robustesse du modèle.

Le modèle entraîné a été sauvegardé dans le fichier `model_random_forest.pkl`.

---

## Fonctionnement global de l’application

### Interface utilisateur
- Sélection dynamique des critères : choix de la structure administrative (niveau 1 filtre niveau 2, qui filtre le poste)
- Sélection du type de contrat, tranche d’âge, et sexe
- Bouton pour lancer la prédiction

### Prédiction
- Transformation des entrées utilisateur pour créer les variables dérivées (`Structure_Combo` et `Sexe_AGE`) nécessaires au modèle
- Chargement du modèle pré-entraîné
- Prédiction du salaire brut et net à payer pour le mois suivant

### Visualisation
- Affichage d’un tableau combinant l’historique mensuel des salaires moyens de 2025 et la prédiction
- Graphique comparatif des salaires brut et net montrant l’évolution et la projection

### Messages d’alerte
- Si les données sont insuffisantes pour une prédiction fiable, un message d’alerte est affiché

---

## Utilisation

Installez les dépendances nécessaires :  
   ```bash
   pip install streamlit scikit-learn joblib pandas matplotlib numpy
