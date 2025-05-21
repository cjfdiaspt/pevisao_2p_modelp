
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
