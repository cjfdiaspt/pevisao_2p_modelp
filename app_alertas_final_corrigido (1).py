
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Previsão 2ª Parte + Registo", layout="centered")
st.title("Previsão 2ª Parte + Alertas")

if "df_base" not in st.session_state:
    st.session_state.df_base = pd.DataFrame(columns=[
        "Home 1P", "Away 1P", "Home 2P", "Away 2P",
        "Total 1ª parte", "Total 2ª parte", "Total jogo",
        "alerta_4", "alerta_14", "alerta_15", "alerta_16", "alerta_17", "cluster_externo"
    ])

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

def obter_cluster_externo(base):
    base = base.copy()
    base["Total jogo"] = base["Total 1ª parte"] + base["Total 2ª parte"]
    media_10 = base["Total jogo"].rolling(10).mean().shift(1)
    pct_over_10 = base["Total jogo"].rolling(10).apply(lambda x: (x > 10.5).mean()).shift(1)
    return np.where(
        (media_10 > 10.5) | (pct_over_10 > 0.6), "C",
        np.where((media_10 < 9.5) | (pct_over_10 < 0.4), "A", "B")
    )

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


st.subheader("1. Previsão após 1ª Parte")

col1, col2 = st.columns(2)
with col1:
    home_1p = st.number_input("Golos Casa 1P", min_value=0, value=2, step=1)
with col2:
    away_1p = st.number_input("Golos Fora 1P", min_value=0, value=2, step=1)

if st.button("Prever 2ª Parte"):
    total_1p = home_1p + away_1p
    diff_1p = abs(home_1p - away_1p)
    base = st.session_state.df_base.copy()
    cluster_externo = obter_cluster_externo(base).tolist()[-1] if len(base) >= 10 else "B"

    a4 = alerta_4(home_1p, away_1p, cluster_externo)
    a14 = alerta_14(total_1p, diff_1p, cluster_externo)
    a17 = alerta_17(total_1p, diff_1p, cluster_externo)

    modelo, scaler = treinar_modelo(base)
    if modelo:
        X_novo = np.array([[total_1p, a4, a14, a17]])
        X_scaled = scaler.transform(X_novo)
        pred = modelo.predict(X_scaled)[0]
        prob = modelo.predict_proba(X_scaled)[0][1]

        if pred == 1:
            st.markdown(f"<span style='color:darkgreen'><strong>PREVISÃO: OVER 4.5 na 2ª parte</strong> – {prob*100:.1f}%</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:red'><strong>PREVISÃO: UNDER 4.5 na 2ª parte</strong> – {(1-prob)*100:.1f}%</span>", unsafe_allow_html=True)

    st.markdown("**Alertas Ativos:**")
    if a4:
        st.markdown("- Alerta 4: Jogo truncado – tendência de UNDER 4.5 na 2ª parte")
    if a14:
        st.markdown("- Alerta 14: Jogo com padrão explosivo – OVER 4.5 provável")
    if a17:
        st.markdown("- Alerta 17: Cluster C com diferença equilibrada – OVER 4.5 provável")
    if not any([a4, a14, a17]):
        st.markdown("- Nenhum alerta ativado nesta 1ª parte.")

st.subheader("2. Registar Resultado Final do Jogo")

col3, col4 = st.columns(2)
with col3:
    home_2p = st.number_input("Golos Casa 2P", min_value=0, value=2, step=1)
with col4:
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

    novo = pd.DataFrame([{
    # Validação combinada dos alertas (exceto alerta 17)
    # Validação do alerta 17
        "Home 1P": home_1p, "Away 1P": away_1p,
        "Home 2P": home_2p, "Away 2P": away_2p,
        "Total 1ª parte": total_1p, "Total 2ª parte": total_2p,
        "Total jogo": total_jogo,
        "alerta_4": a4, "alerta_14": a14, "alerta_15": a15,
        "alerta_16": a16, "alerta_17": a17,
        "cluster_externo": cluster_externo
    }])
    st.session_state.df_base = pd.concat([st.session_state.df_base, novo], ignore_index=True)


    # Validação combinada dos alertas (exceto 17 que já está separado)
    alertas_ativos = [a4, a14, a15, a16]
    acerto = int(total_2p > 4.5)
    alerta_geral_ativo = int(any(alertas_ativos))
    alerta_geral_correto = int(alerta_geral_ativo and acerto)
    alerta_geral_errado = int(alerta_geral_ativo and not acerto)

    novo["alerta_geral_correto"] = alerta_geral_correto
    novo["alerta_geral_errado"] = alerta_geral_errado


    # Validação do alerta 17

    # Validação do alerta 17 (acertos em OVER ou UNDER com alta confiança)
    if a17 == 1:
        if total_2p > 4.5:
            alerta_17_correto = 1
            alerta_17_errado = 0
        elif total_2p <= 4.5:
            alerta_17_correto = 1
            alerta_17_errado = 0
        else:
            alerta_17_correto = 0
            alerta_17_errado = 1
    else:
        alerta_17_correto = 0
        alerta_17_errado = 0

    novo["alerta_17_correto"] = alerta_17_correto
    novo["alerta_17_errado"] = alerta_17_errado
    alerta_17_errado = int(a17 == 1 and total_2p <= 4.5)

    novo["alerta_17_correto"] = alerta_17_correto
    novo["alerta_17_errado"] = alerta_17_errado

    st.success("Jogo adicionado à base com sucesso.")
    st.markdown("**Alertas pós-jogo (para o próximo jogo):**")
    if a15:
        st.markdown("- ✅ Reação esperada: Aposta recomendada **OVER 4.5 na 2ª parte** (alerta 15)")
    else:
        st.markdown("- ⛔ Nenhuma reação esperada (alerta 15 não ativo)")

    if a16:
        st.markdown("- ✅ Correção ofensiva provável: Aposta recomendada **OVER 4.5 na 1ª parte** (alerta 16)")
    else:
        st.markdown("- ⛔ Nenhuma indicação de correção ofensiva (alerta 16 não ativo)")

st.subheader("3. Base de Dados Completa")
st.dataframe(st.session_state.df_base)

st.download_button(
    label="📥 Baixar base de dados",
    data=st.session_state.df_base.to_csv(index=False).encode("utf-8"),
    file_name="base_completa_alertas.csv",
    mime="text/csv"
)
