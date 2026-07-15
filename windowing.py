"""
windowing.py — Découpe le signal de puissance en fenêtres analysables.

Deux stratégies :
  1. slice_fixed   : fenêtres de durée fixe avec chevauchement (pour CNN1D)
  2. slice_on_edges: fenêtres déclenchées par un saut de puissance (pour DTW, RF)

Une fenêtre est un dict :
    {
        "timestamps": list[str],   # horodatages ISO
        "values":     list[float], # puissances en W
        "tag":        str|None,    # label si annoté
        "n_points":   int,
    }
"""

import numpy as np
import pandas as pd
from typing import Optional

# ── Helpers ────────────────────────────────────────────────────────────────────

def _rows_to_window(rows: pd.DataFrame) -> dict:
    return {
        "timestamps": rows["timestamp"].tolist(),
        "values":     rows["valeur_W"].tolist(),
        "tag":        rows["tag"].iloc[0] if "tag" in rows.columns else None,
        "n_points":   len(rows),
    }


def _majority_tag(series: pd.Series) -> Optional[str]:
    """Retourne le tag majoritaire (hors None) dans une fenêtre, ou None."""
    counts = series.dropna().value_counts()
    return counts.index[0] if len(counts) > 0 else None


# ── Stratégie 1 : fenêtres fixes ──────────────────────────────────────────────

def slice_fixed(
    df: pd.DataFrame,
    window_s: int = 300,   # durée de la fenêtre en secondes (défaut 5 min)
    step_s:   int = 150,   # pas de glissement (chevauchement 50%)
    min_points: int = 10,  # fenêtres trop courtes ignorées
) -> list[dict]:
    """
    Découpe le signal en fenêtres de durée fixe avec chevauchement.

    Utile pour CNN1D qui exige une taille d'entrée constante.
    Le tag est attribué par vote majoritaire dans la fenêtre.

    Exemple : window_s=300, step_s=150 → une fenêtre toutes les 2.5 min,
    chaque point est couvert par 2 fenêtres → meilleure couverture.
    """
    if df.empty:
        return []

    # Convertir les timestamps en secondes pour les calculs
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["t_s"] = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype(int)

    t_start = int(df["t_s"].iloc[0])
    t_end   = int(df["t_s"].iloc[-1])

    windows = []
    t = t_start

    while t + window_s <= t_end:
        mask = (df["t_s"] >= t) & (df["t_s"] < t + window_s)
        chunk = df[mask]

        if len(chunk) >= min_points:
            tag = _majority_tag(chunk["tag"]) if "tag" in chunk.columns else None
            windows.append({
                "timestamps": chunk["timestamp"].tolist(),
                "values":     chunk["valeur_W"].tolist(),
                "tag":        tag,
                "n_points":   len(chunk),
            })

        t += step_s

    return windows


# ── Stratégie 2 : fenêtres sur fronts de puissance ────────────────────────────

def detect_edges(
    df: pd.DataFrame,
    threshold_W: float = 100.0,  # saut minimum pour déclencher un front
    min_gap_s:   int   = 30,     # ignorer les fronts trop proches
) -> list[int]:
    """
    Détecte les indices où la puissance saute de plus de threshold_W.

    Indépendant des tags : permet de repérer les événements même
    sur des données non annotées.

    Retourne une liste d'indices dans le DataFrame.
    """
    if df.empty:
        return []

    values = df["valeur_W"].to_numpy(dtype=float)
    ts     = pd.to_datetime(df["timestamp"], utc=True)
    t_s    = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype(int).to_numpy()

    edges     = []
    last_edge = -min_gap_s  # permet de déclencher dès le début

    for i in range(1, len(values)):
        delta = abs(values[i] - values[i - 1])
        gap   = t_s[i] - last_edge

        if delta >= threshold_W and gap >= min_gap_s:
            edges.append(i)
            last_edge = t_s[i]

    return edges


def slice_on_edges(
    df: pd.DataFrame,
    threshold_W: float = 100.0,  # seuil de détection de front
    pre_s:       int   = 30,     # secondes avant le front incluses
    post_s:      int   = 270,    # secondes après le front incluses (défaut 4.5 min)
    min_points:  int   = 6,      # fenêtres trop courtes ignorées
) -> list[dict]:
    """
    Crée une fenêtre autour de chaque saut de puissance détecté.

    Avantages :
    - Les fenêtres ont une durée variable → adapté au DTW et RandomForest
    - Centré sur l'événement → moins de bruit de fond
    - Fonctionne aussi sur données non annotées (exploration)

    Le tag est attribué par vote majoritaire dans la fenêtre.
    """
    if df.empty:
        return []

    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["t_s"] = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype(int)

    edge_indices = detect_edges(df, threshold_W=threshold_W)
    windows = []

    for idx in edge_indices:
        t_edge = df["t_s"].iloc[idx]
        mask   = (df["t_s"] >= t_edge - pre_s) & (df["t_s"] <= t_edge + post_s)
        chunk  = df[mask]

        if len(chunk) < min_points:
            continue

        tag = _majority_tag(chunk["tag"]) if "tag" in chunk.columns else None
        windows.append({
            "timestamps": chunk["timestamp"].tolist(),
            "values":     chunk["valeur_W"].tolist(),
            "tag":        tag,
            "n_points":   len(chunk),
            "edge_idx":   idx,  # utile pour le débogage
        })

    return windows


# ── Utilitaire : pad/truncate pour CNN ────────────────────────────────────────

def to_fixed_length(windows: list[dict], length: int) -> np.ndarray:
    """
    Convertit une liste de fenêtres en tableau numpy (N, length) pour CNN1D.

    Les fenêtres trop courtes sont paddées à zéro à droite.
    Les fenêtres trop longues sont tronquées.
    """
    X = np.zeros((len(windows), length), dtype=np.float32)
    for i, w in enumerate(windows):
        vals = np.array(w["values"], dtype=np.float32)
        n    = min(len(vals), length)
        X[i, :n] = vals[:n]
    return X


# ── Test rapide ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Données synthétiques : signal plat à 500W avec un pic à 1400W
    import pandas as pd

    n = 200
    t = pd.date_range("2026-01-01", periods=n, freq="10s")
    w = [500] * n
    w[80:100] = [1400] * 20  # pic 200s

    df = pd.DataFrame({
        "timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valeur_W":  w,
        "tag":       ["micro_ondes"] * n,
    })

    fixed  = slice_fixed(df, window_s=120, step_s=60)
    edges  = slice_on_edges(df, threshold_W=100, pre_s=20, post_s=60)

    print(f"Fenêtres fixes   : {len(fixed)}")
    print(f"Fenêtres sur fronts : {len(edges)}")
    if edges:
        print(f"  1ère fenêtre : {edges[0]['n_points']} pts, tag={edges[0]['tag']}")
