
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Previsão 2ª Parte + Registo", layout="centered")
st.title("Sistema de Previsão com Alertas Ativos")

# Inicialização da base em memória
if "df_base" not in st.session_state:
    st.session_state.df_base = pd.DataFrame(columns=[
        "Home 1P", "Away 1P", "Home 2P", "Away 2P", "Total 1ª parte", "Total 2ª parte", "Total jogo",
        "alerta_4", "alerta_14", "alerta_15", "alerta_16", "alerta_17", "cluster_externo"
    ])

# Funções dos alertas
def alerta_4(h1p, a1p, cluster):
    return int(h1p == a1p and h1p + a1p <= 2 and cluster == "A")

def alerta_14(t1p, d1p, cluster):
    if cluster != "C":
        return 0
    return int(
        (t1p >= 6 and d1p <= 2) or (t1p >= 5 and d1p <= 2) or
        (t1p >= 4 and d1p <= 2) or (t1p >= 3 and d1p <= 2) or
        (t1p >= 3 and d1p <= 1)
    )

def alerta_15(cluster_ant, t2p_ant):
    return int(cluster_ant == "C" and t2p_ant <= 4)

def alerta_16(t1p_ant, tj_ant, t1p, d1p, cluster):
    if not (t1p_ant > 4.5 and tj_ant > 12):
        return 0
    return int(
        (t1p >= 5 and d1p <= 2 and cluster in ["B", "C"]) or
        (t1p >= 4 and d1p <= 2 and cluster in ["B", "C"]) or
        (t1p >= 3 and d1p <= 2 and cluster == "C")
    )

def alerta_17(t1p, d1p, cluster):
    if cluster != "C":
        return 0
    return int(
        (t1p >= 2 and d1p <= 3) or (t1p >= 3 and d1p <= 3) or
        (t1p >= 4 and d1p <= 3) or (t1p >= 5 and d1p <= 3) or
        (t1p >= 2 and d1p <= 2) or (t1p >= 3 and d1p <= 2) or
        (t1p >= 4 and d1p <= 2) or (t1p >= 5 and d1p <= 2)
    )

# Cálculo do cluster externo com base nos últimos 10 jogos
def obter_cluster_externo(base):
    base = base.copy()
    base["Total jogo"] = base["Total 1ª parte"] + base["Total 2ª parte"]
    media_10 = base["Total jogo"].rolling(10).mean().shift(1)
    pct_over_10 = base["Total jogo"].rolling(10).apply(lambda x: (x > 10.5).mean()).shift(1)
    return np.where(
        (media_10 > 10.5) | (pct_over_10 > 0.6), "C",
        np.where((media_10 < 9.5) | (pct_over_10 < 0.4), "A", "B")
    )


# Treinar modelo com Random Forest incluindo os alertas como features
def treinar_modelo(df):
    if len(df) < 10:
        return None, None
    df = df.dropna()
    df["Target"] = (df["Total 2ª parte"] > 4.5).astype(int)
    features = ["Total 1ª parte", "alerta_4", "alerta_14", "alerta_17"]
    X = df[features]
    y = df["Target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

# Interface de previsão
st.subheader("1. Previsão após 1ª Parte")

col1, col2 = st.columns(2)
with col1:
    home_1p = st.number_input("Golos Casa 1P", min_value=0, value=2, step=1)
    away_1p = st.number_input("Golos Fora 1P", min_value=0, value=1, step=1)

if st.button("Ativar Previsão + Alertas"):
    total_1p = home_1p + away_1p
    diff_1p = abs(home_1p - away_1p)
    df_base = st.session_state.df_base.copy()
    cluster_externo = obter_cluster_externo(df_base).tolist()[-1] if len(df_base) >= 10 else "B"

    a4 = alerta_4(home_1p, away_1p, cluster_externo)
    a14 = alerta_14(total_1p, diff_1p, cluster_externo)
    a17 = alerta_17(total_1p, diff_1p, cluster_externo)

    # Treinar modelo se possível
    modelo, scaler = treinar_modelo(df_base)
    if modelo:
        X_new = np.array([[total_1p, a4, a14, a17]])
        Xs = scaler.transform(X_new)
        pred = modelo.predict(Xs)[0]
        prob = modelo.predict_proba(Xs)[0][1]
        st.markdown(f"**Cluster:** {cluster_externo}")
        st.markdown(f"**Alerta 4:** {a4}, **Alerta 14:** {a14}, **Alerta 17:** {a17}")
        if pred == 1:
            st.markdown(f"<span style='color:darkgreen'><strong>OVER 4.5 (2ª parte)</strong> – {prob*100:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'><strong>UNDER 4.5 (2ª parte)</strong> – {(1-prob)*100:.1f}%</span>", unsafe_allow_html=True)
    else:
        st.warning("Insere pelo menos 10 jogos completos na base para ativar o modelo.")


st.subheader("2. Registar Resultado Final do Jogo")

col3, col4 = st.columns(2)
with col3:
    home_2p = st.number_input("Golos Casa 2P", min_value=0, value=3, step=1)
    away_2p = st.number_input("Golos Fora 2P", min_value=0, value=2, step=1)

if st.button("Adicionar Jogo Completo"):
    total_1p = home_1p + away_1p
    total_2p = home_2p + away_2p
    total_jogo = total_1p + total_2p
    diff_1p = abs(home_1p - away_1p)
    df = st.session_state.df_base.copy()
    cluster_externo = obter_cluster_externo(df).tolist()[-1] if len(df) >= 10 else "B"

    t1p_ant = df["Total 1ª parte"].iloc[-1] if len(df) > 0 else 0
    t2p_ant = df["Total 2ª parte"].iloc[-1] if len(df) > 0 else 0
    tj_ant = df["Total jogo"].iloc[-1] if len(df) > 0 else 0

    a4 = alerta_4(home_1p, away_1p, cluster_externo)
    a14 = alerta_14(total_1p, diff_1p, cluster_externo)
    a17 = alerta_17(total_1p, diff_1p, cluster_externo)
    a15 = alerta_15(cluster_externo, t2p_ant)
    a16 = alerta_16(t1p_ant, tj_ant, total_1p, diff_1p, cluster_externo)

    novo_jogo = pd.DataFrame([{
        "Home 1P": home_1p, "Away 1P": away_1p,
        "Home 2P": home_2p, "Away 2P": away_2p,
        "Total 1ª parte": total_1p, "Total 2ª parte": total_2p, "Total jogo": total_jogo,
        "alerta_4": a4, "alerta_14": a14, "alerta_15": a15,
        "alerta_16": a16, "alerta_17": a17,
        "cluster_externo": cluster_externo
    }])

    st.session_state.df_base = pd.concat([st.session_state.df_base, novo_jogo], ignore_index=True)
    st.success("Jogo adicionado à base com sucesso.")

st.markdown("**Alertas pós-jogo:**")
    if a15:
st.markdown("- Alerta 15 ativo para o próximo jogo")
    if a16:
st.markdown("- Alerta 16 ativo para o próximo jogo")

st.subheader("3. Base de Dados Completa")
st.dataframe(st.session_state.df_base)

st.download_button(
    label="📥 Baixar base de dados",
    data=st.session_state.df_base.to_csv(index=False).encode("utf-8"),
    file_name="base_completa_alertas.csv",
    mime="text/csv"
)


st.markdown("**Alertas pós-jogo:**")
st.markdown(f"- Alerta 15: {'ATIVO' if a15 else 'NÃO ATIVO'}")
st.markdown(f"- Alerta 16: {'ATIVO' if a16 else 'NÃO ATIVO'}")


import matplotlib.pyplot as plt

st.subheader("4. Evolução do Acerto do Modelo")

def calcular_evolucao_acerto(base):
    base = base.copy()
    base["Target"] = (base["Total 2ª parte"] > 4.5).astype(int)
    base["Predito"] = np.nan

    modelo, scaler = treinar_modelo(base)
    if modelo:
        for i in range(10, len(base)):
            treino = base.iloc[:i]
            if len(treino) < 10:
                continue
            X_train = treino[["Total 1ª parte", "alerta_4", "alerta_14", "alerta_17"]]
            y_train = treino["Target"]
            modelo_i = RandomForestClassifier(n_estimators=100, random_state=42)
            scaler_i = StandardScaler()
            X_scaled = scaler_i.fit_transform(X_train)
            modelo_i.fit(X_scaled, y_train)

            X_teste = base[["Total 1ª parte", "alerta_4", "alerta_14", "alerta_17"]].iloc[i:i+1]
            Xs_teste = scaler_i.transform(X_teste)
            pred = modelo_i.predict(Xs_teste)[0]
            base.at[i, "Predito"] = pred

        base = base.dropna(subset=["Predito"])
        base["Acerto"] = (base["Predito"] == base["Target"]).astype(int)
        base["Acerto Móvel"] = base["Acerto"].rolling(10).mean()

        fig, ax = plt.subplots()
        ax.plot(base.index, base["Acerto Móvel"] * 100, marker="o")
        ax.set_ylim(0, 100)
        ax.set_title("Taxa de Acerto Móvel (últimos 10 jogos)")
        ax.set_ylabel("% Acerto")
        st.pyplot(fig)
    else:
        st.warning("São necessários pelo menos 10 jogos para calcular a evolução.")

calcular_evolucao_acerto(st.session_state.df_base)
