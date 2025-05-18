import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prédiction masse salariale", page_icon="💰", layout="centered")
st.title("💼 Prédiction de la masse salariale (Fonction publique)")
st.markdown("Sélectionnez les critères pour estimer la masse salariale du mois suivant.")

@st.cache_data
def charger_donnees():
    df = pd.read_csv("MEF_PY_V09_H.csv")
    df['MOIPAI'] = pd.to_datetime(df['MOIPAI'], errors='coerce')
    df = df.dropna(subset=['MOIPAI', 'Structure_Niv1', 'Structure_Niv2', 'LIBCADEMP'])
    return df

df = charger_donnees()

# --- Champs dynamiques ---
# Structure_Niv1 (niveau 1)
niv1_options = sorted(df['Structure_Niv1'].unique())
Structure_Niv1 = st.selectbox("Structure_Niv1", [''] + niv1_options)

# Si Structure_Niv1 sélectionné, on filtre les autres champs
if Structure_Niv1:
    filtered_df = df[df['Structure_Niv1'] == Structure_Niv1]

    # Filtrage dynamique du niveau 2
    niv2_options = sorted(filtered_df['Structure_Niv2'].dropna().unique())
    Structure_Niv2 = st.selectbox("Structure_Niv2", [''] + niv2_options)

    # Filtrage dynamique de LIBCADEMP basé sur Structure_Niv2
    if Structure_Niv2:
        filtered_df = filtered_df[filtered_df['Structure_Niv2'] == Structure_Niv2]

    # Maintenant on filtre LIBCADEMP selon Structure_Niv1 (et Structure_Niv2 si dispo)
    libcademp_options = sorted(filtered_df['LIBCADEMP'].dropna().unique())
    LIBCADEMP = st.selectbox("Poste", [''] + libcademp_options)

else:
    Structure_Niv2 = st.selectbox("Structure_Niv2", [''])
    LIBCADEMP = st.selectbox("Poste", [''])

# Autres champs non liés à Structure_Niv1
CDLQUA = st.selectbox("Type de contrat", ['', 'Fonctionnaire', 'Non titulaire'])
TRANCHE_AGE = st.selectbox("Tranche d'âge", ['','15 a 19 ans','20 a 24 ans', '25 a 29 ans','30 a 34 ans','35 a 39 ans',
                                             '40 a 44 ans','45 a 49 ans','50 a 54 ans','55 a 59 ans','60 a 64 ans'])
Sexe_ano = st.selectbox("Sexe", ['','Feminin', 'Masculin'])

# --- Prédiction + Visualisation ---
def afficher_2025_et_prediction(df, prediction_brut, prediction_net, Structure_Niv1, Structure_Niv2,   
                                 LIBCADEMP, CDLQUA,TRANCHE_AGE, Sexe_ano):

    filtre_agent = (
        (df['Structure_Niv1'] == Structure_Niv1) &
        (df['Structure_Niv2'] == Structure_Niv2) &
        (df['LIBCADEMP'] == LIBCADEMP) &
        (df['CDLQUA'] == CDLQUA) &
        (df['TRANCHE_AGE'] == TRANCHE_AGE) &
        (df['Sexe_ano'] == Sexe_ano)
    )

    df_filtre = df[filtre_agent & (df['MOIPAI'].dt.year == 2025)]

    historique = df_filtre.groupby('MOIPAI')[['Brut', 'Net a payer']].mean().round(0)

    if not historique.empty:
        dernier_mois = historique.index.sort_values().max()
    else:
        st.warning("⚠️ ALERTE : Les données sont insuffisantes pour produire une prédiction fiable.")
        dernier_mois = pd.Timestamp('2024-12-01')
        return

    mois_suivant = (dernier_mois + pd.DateOffset(months=1)).replace(day=1)

    prediction_df = pd.DataFrame({
        'Brut': [prediction_brut],
        'Net a payer': [prediction_net]
    }, index=[mois_suivant]).round(0)

    final_df = pd.concat([historique, prediction_df])
    final_df = final_df.sort_index()

    st.subheader(f"📉 Historique 2025 et Prédiction ({mois_suivant.strftime('%B %Y')})")
    st.write("🧾 **Historique 2025 :**")
    st.dataframe(final_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(final_df.index))
    bar_width = 0.35

    couleurs_brut = ['skyblue'] * (len(final_df) - 1) + ['orange']
    couleurs_net = ['lightgreen'] * (len(final_df) - 1) + ['yellow']

    ax.bar(x - bar_width / 2, final_df['Brut'], width=bar_width, color=couleurs_brut, label='Brut')
    ax.bar(x + bar_width / 2, final_df['Net a payer'], width=bar_width, color=couleurs_net, label='Net à payer')

    ax.set_ylabel("Montant (€)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime('%b %Y') for d in final_df.index], rotation=45)
    ax.legend()
    st.pyplot(fig)

# --- Bouton de prédiction ---
if st.button('Prédire la masse salariale'):
    if '' in [Structure_Niv1, Structure_Niv2, LIBCADEMP, CDLQUA, 
              TRANCHE_AGE, Sexe_ano]:
        st.error("❌ Merci de remplir tous les champs avant de prédire.")
    else:
        input_data = pd.DataFrame([{
            'Structure_Niv1': Structure_Niv1,
            'Structure_Niv2': Structure_Niv2,
            'LIBCADEMP': LIBCADEMP,
            'CDLQUA': CDLQUA,
            'TRANCHE_AGE': TRANCHE_AGE,
            'Sexe_ano': Sexe_ano
        }])


        # Ajout des colonnes dérivées attendues par le modèle
        input_data['Structure_Combo'] = input_data['Structure_Niv1'] + "_" + input_data['Structure_Niv2']
        input_data['Sexe_AGE'] = input_data['Sexe_ano'] + "_" + input_data['TRANCHE_AGE']

        # Prédiction
        model = joblib.load('model_random_forest.pkl')
        prediction = model.predict(input_data)
        prediction_brut, prediction_net = prediction[0]

        st.success(f"💰 Masse salariale estimée (Brut) : **{prediction_brut:,.2f} €**")
        st.success(f"💶 Masse salariale estimée (Net à payer) : **{prediction_net:,.2f} €**")

        st.markdown("---")

        afficher_2025_et_prediction(df, prediction_brut, prediction_net,
                            Structure_Niv1, Structure_Niv2, LIBCADEMP, CDLQUA,  
                            TRANCHE_AGE, Sexe_ano)
