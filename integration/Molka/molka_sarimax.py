# =============================================================
# molka_sarimax.py — Module SARIMAX / STL
# Molka Module | ESPRIT 3A IA
#
# Fonctions exportées :
#   - creer_exog(index)        : variables exogènes ramadan/été/hiver
#   - train_all(df)            : entraîne SARIMAX sur les 6 quartiers
#   - forecast(modeles, preds_base, quartier, horizon)
#   - get_stl(df, quartier)    : décomposition STL
#   - get_metrics(modeles)     : MAE / RMSE par quartier
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from molka_data import QUARTIERS, RAMADAN_MOIS


def _get_sarimax():
    """Import lazy de SARIMAX — permet l'installation en cours de session."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    return SARIMAX

def _get_stl_cls():
    from statsmodels.tsa.seasonal import STL
    return STL

def _get_metrics_fns():
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    return mean_absolute_error, mean_squared_error

# ── Horizon de prévision ──────────────────────────────────────
HORIZON   = 26   # semaines
N_TEST    = 13   # semaines tenu de côté pour évaluation

# ── Variables exogènes ────────────────────────────────────────
def creer_exog(index):
    """
    Crée le DataFrame d'exogènes (ramadan / été / hiver) pour un index temporel.

    Args:
        index : pd.DatetimeIndex (hebdomadaire)

    Returns:
        pd.DataFrame avec colonnes [ramadan, ete, hiver]
    """
    ram = (
        ((index.year == 2021) & (index.month == 4)) |
        ((index.year == 2022) & (index.month == 4)) |
        ((index.year == 2023) & (index.month == 3)) |
        ((index.year == 2024) & (index.month == 3))
    )
    return pd.DataFrame({
        'ramadan': ram.astype(int),
        'ete':     index.month.isin([6, 7, 8]).astype(int),
        'hiver':   index.month.isin([12, 1, 2]).astype(int),
    }, index=index)


# ── Entraînement SARIMAX ─────────────────────────────────────
def train_all(df, horizon=HORIZON, n_test=N_TEST, verbose=True, quartiers=None):
    """
    Entraîne un modèle SARIMAX(1,0,1)(1,0,1,52) par quartier.

    Args:
        df        : DataFrame issu de molka_data.generate_dataset()
        horizon   : nombre de semaines à prédire (défaut 26)
        n_test    : taille du split test pour les métriques (défaut 13)
        verbose   : affiche le progression
        quartiers : liste de quartiers à entraîner (None = tous)

    Returns:
        dict {
            'modeles':     {quartier: SARIMAXResults},
            'series_q':    {quartier: pd.Series},
            'preds_base':  {quartier: {'dates', 'mean', 'ci_lo', 'ci_hi'}},
            'mae_scores':  {quartier: float},
            'rmse_scores': {quartier: float},
        }
    """
    SARIMAX = _get_sarimax()
    mean_absolute_error, mean_squared_error = _get_metrics_fns()

    if quartiers is None:
        quartiers = QUARTIERS

    if verbose:
        print(f"Entraînement SARIMAX ({len(quartiers)} quartier(s))...")

    modeles     = {}
    series_q    = {}
    preds_base  = {}
    mae_scores  = {}
    rmse_scores = {}

    for q in quartiers:
        serie = (
            df[df['quartier'] == q]
            .set_index('date')['densite_chiens']
            .asfreq('W')
        )
        series_q[q] = serie
        exog = creer_exog(serie.index)

        # Split train / test
        s_train, s_test = serie.iloc[:-n_test], serie.iloc[-n_test:]
        e_train, e_test = exog.iloc[:-n_test],  exog.iloc[-n_test:]

        model = SARIMAX(
            s_train, exog=e_train,
            order=(1, 0, 1), seasonal_order=(1, 0, 1, 52),
            enforce_stationarity=False, enforce_invertibility=False
        ).fit(disp=False, method='lbfgs', maxiter=200)
        modeles[q] = model

        # Métriques sur test set
        fc_test = model.get_forecast(steps=n_test, exog=e_test)
        y_pred  = fc_test.predicted_mean.values
        mae_scores[q]  = round(mean_absolute_error(s_test, y_pred), 3)
        rmse_scores[q] = round(np.sqrt(mean_squared_error(s_test, y_pred)), 3)

        # Prévisions futures (horizon semaines)
        futur_dates = pd.date_range(
            serie.index[-1] + pd.Timedelta(weeks=1),
            periods=horizon, freq='W'
        )
        exog_fut = creer_exog(futur_dates)
        fc_fut   = model.get_forecast(steps=horizon, exog=exog_fut)
        ci       = fc_fut.conf_int(alpha=0.05)

        preds_base[q] = {
            'dates': futur_dates,
            'mean':  fc_fut.predicted_mean.values,
            'ci_lo': ci.iloc[:, 0].values,
            'ci_hi': ci.iloc[:, 1].values,
        }

        if verbose:
            print(f"  ✅ {q:12s} | AIC={model.aic:.0f} | MAE={mae_scores[q]} | RMSE={rmse_scores[q]}")

    if verbose:
        print(f"\n✅ SARIMAX entraîné ({len(quartiers)} quartier(s)) — moteur du Digital Twin prêt")

    return {
        'modeles':     modeles,
        'series_q':    series_q,
        'preds_base':  preds_base,
        'mae_scores':  mae_scores,
        'rmse_scores': rmse_scores,
    }


# ── Prévision pour un quartier ────────────────────────────────
def forecast(preds_base, quartier):
    """
    Retourne les prévisions SARIMAX pour un quartier.

    Args:
        preds_base : dict issu de train_all()
        quartier   : nom du quartier

    Returns:
        dict {dates (ISO strings), mean, ci_lo, ci_hi}
    """
    pb = preds_base[quartier]
    return {
        'quartier': quartier,
        'dates':    [d.isoformat() for d in pb['dates']],
        'mean':     pb['mean'].tolist(),
        'ci_lo':    pb['ci_lo'].tolist(),
        'ci_hi':    pb['ci_hi'].tolist(),
    }


# ── Décomposition STL ─────────────────────────────────────────
def get_stl(df, quartier, period=52):
    """
    Décompose la série temporelle d'un quartier en tendance/saison/résidu.

    Returns:
        dict {dates, observed, trend, seasonal, resid}
    """
    STL = _get_stl_cls()

    serie = (
        df[df['quartier'] == quartier]
        .set_index('date')['densite_chiens']
        .asfreq('W')
        .dropna()
    )

    stl    = STL(serie, period=period, robust=True)
    result = stl.fit()

    return {
        'quartier': quartier,
        'dates':    [d.isoformat() for d in serie.index],
        'observed': serie.values.tolist(),
        'trend':    result.trend.tolist(),
        'seasonal': result.seasonal.tolist(),
        'resid':    result.resid.tolist(),
    }


# ── Métriques ─────────────────────────────────────────────────
def get_metrics(sarimax_result):
    """
    Retourne un dict {quartier: {mae, rmse, aic}} depuis le résultat train_all().
    """
    out = {}
    for q in QUARTIERS:
        out[q] = {
            'mae':  sarimax_result['mae_scores'].get(q),
            'rmse': sarimax_result['rmse_scores'].get(q),
            'aic':  round(sarimax_result['modeles'][q].aic, 1)
                    if q in sarimax_result['modeles'] else None,
        }
    return out


# ── Test rapide ───────────────────────────────────────────────
if __name__ == '__main__':
    from molka_data import generate_dataset
    df = generate_dataset()
    res = train_all(df, verbose=True)
    print("\nMétriques :")
    for q, m in get_metrics(res).items():
        print(f"  {q:12s} | MAE={m['mae']} | RMSE={m['rmse']} | AIC={m['aic']}")
