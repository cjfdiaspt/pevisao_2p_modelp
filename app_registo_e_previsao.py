
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import io

st.set_page_config(page_title="Previsão e Registo de Jogos", layout="centered")

st.title("Previsão e Registo - Over 4.5 na 2ª Parte")

# Inicializar base
if "df_base" not in st.session_state:
    st.session_state.df_base = pd.DataFrame(columns=["Total 1ª parte", "Total 2ª parte", "Total jogo"])

# Mostrar base atual
st.subheader("Base de Dados Atual")
st.dataframe(st.session_state.df_base)

# Treinar modelo (se possível)
def train_model(df):
    df = df.copy()
    df["Total 1P Anterior"] = df["Total 1ª parte"].shift(1)
    df["Total 2P Anterior"] = df["Total 2ª parte"].shift(1)
    df["Total jogo anterior"] = df["Total jogo"].shift(1)
    df = df.dropna()
    df["Target"] = (df["Total 2ª parte"] > 4.5).astype(int)
    X = df[["Total 1ª parte", "Total 1P Anterior", "Total 2P Anterior", "Total jogo anterior"]]
    y = df["Target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, df

if len(st.session_state.df_base) >= 5:
    model, scaler, df_model = train_model(st.session_state.df_base)
    st.success("Modelo treinado com sucesso!")
else:
    model, scaler, df_model = None, None, None
    st.info("Adiciona pelo menos 5 jogos completos para ativar a previsão.")

# Prever a 2ª parte
st.subheader("1. Prever 2ª Parte com base na 1ª Parte")

total_1p_atual = st.number_input("Total 1ª Parte (atual)", step=1, min_value=0, value=3)

if st.button("Prever 2ª Parte"):
    if model:
        ult = st.session_state.df_base.iloc[-1]
        entrada = np.array([[total_1p_atual, ult["Total 1ª parte"], ult["Total 2ª parte"], ult["Total jogo"]]])
        entrada_scaled = scaler.transform(entrada)
        pred = model.predict(entrada_scaled)[0]
        prob = model.predict_proba(entrada_scaled)[0][1]
        if pred == 1:
            st.markdown(f"<span style='color:darkgreen'><strong>OVER 4.5 (2ª parte)</strong> – {prob*100:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'><strong>UNDER 4.5 (2ª parte)</strong> – {(1-prob)*100:.1f}%</span>", unsafe_allow_html=True)
    else:
        st.warning("Ainda não há dados suficientes para prever.")

# Registar resultado completo
st.subheader("2. Registar Resultado Final do Jogo")

col1, col2 = st.columns(2)
with col1:
    novo_1p = st.number_input("Total 1ª Parte (novo jogo)", key="novo1p", step=1, min_value=0)
with col2:
    novo_2p = st.number_input("Total 2ª Parte (novo jogo)", key="novo2p", step=1, min_value=0)

if st.button("Adicionar Jogo Completo"):
    total = novo_1p + novo_2p
    novo_jogo = pd.DataFrame([[novo_1p, novo_2p, total]], columns=["Total 1ª parte", "Total 2ª parte", "Total jogo"])
    st.session_state.df_base = pd.concat([st.session_state.df_base, novo_jogo], ignore_index=True)
    st.success("Jogo adicionado à base com sucesso!")

# Exportar base
csv = st.session_state.df_base.to_csv(index=False).encode('utf-8')
st.download_button("📥 Baixar base atualizada", data=csv, file_name="base_dados.csv", mime="text/csv")
