# =============================================================
# molka_api.py — API Flask pour le module Molka
# DOGSIM-TN | ESPRIT 3A IA
#
# Architecture : calcul à la demande par quartier
#   → Serveur prêt en ~1s (dataset uniquement)
#   → SARIMAX + RL entraînés au premier appel pour un quartier donné
#   → Résultats mis en cache en mémoire pour les appels suivants
#
# Endpoints :
#   GET  /molka/health              : état + quartiers déjà entraînés
#   GET  /molka/quartiers           : liste des quartiers + métadonnées
#   POST /molka/sarimax             : prévisions SARIMAX (calcul si besoin)
#   GET  /molka/stl/<quartier>      : décomposition STL (calcul si besoin)
#   GET  /molka/metrics             : MAE/RMSE/AIC des quartiers entraînés
#   POST /molka/simulate            : simulation Digital Twin (calcul si besoin)
#   POST /molka/budget              : stratégie RL avec budget
#   POST /molka/simulate/all        : simulation 4 scénarios × 6 quartiers
#   GET  /molka/geo                 : coordonnées GPS
#   POST /molka/reset/<quartier>    : vider le cache d'un quartier
# =============================================================

import time
import threading
import traceback
import subprocess
import sys
from pathlib import Path

from flask import Blueprint, jsonify, request

# ── Installation automatique des dépendances ─────────────────
def _ensure_deps():
    pkgs = {'statsmodels': 'statsmodels>=0.14.0', 'sklearn': 'scikit-learn>=1.3.0'}
    missing = [pkg for imp, pkg in pkgs.items()
               if not __import__(imp, globals(), locals(), [], 0) and True
               or False]
    # Vérification propre
    missing = []
    for imp, pkg in pkgs.items():
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[Molka] Installation : {missing} …")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet'] + missing)
        print("[Molka] ✅ Dépendances installées")

_ensure_deps()

# ── Import modules molka ──────────────────────────────────────
try:
    from molka_data    import generate_dataset, QUARTIERS, QUARTIERS_CFG, GEO
    from molka_sarimax import train_all, forecast, get_stl, get_metrics, creer_exog
    from molka_rl      import (
        train_all_agents, simuler_twin, generer_strategie_budget,
        valider_budget, ACTIONS, COUT_ACTION,
        BUDGET_MIN_ABSOLU, BUDGET_RECOMMANDE, BUDGET_OPTIMAL,
        HORIZON_RL, color_risque,
    )
    _IMPORT_OK = True
except ImportError as e:
    _IMPORT_OK = False
    _IMPORT_ERROR = str(e)

# ── Blueprint Flask ───────────────────────────────────────────
molka_bp = Blueprint('molka', __name__, url_prefix='/molka')

# ── État global ───────────────────────────────────────────────
# Dataset partagé (généré une fois au démarrage, ~1s)
_DF        = None
_DF_READY  = False
_DF_LOCK   = threading.Lock()

# Cache par quartier — entraîné à la demande
# Structure : { quartier: {'sarimax': ..., 'rl': ..., 'trained_at': ...} }
_CACHE      = {}
_CACHE_LOCK = threading.Lock()


# ── Initialisation du dataset ─────────────────────────────────
def _init_dataset():
    global _DF, _DF_READY
    print("[Molka] Génération du dataset (2015-2023)…")
    _DF = generate_dataset()
    _DF_READY = True
    print(f"[Molka] ✅ Dataset prêt — {len(_DF)} lignes · {_DF['quartier'].nunique()} quartiers")


def init_molka(app=None, eager=False):
    """
    Initialise le module Molka.
    Ne fait que générer le dataset (~1s).
    Les modèles SARIMAX/RL sont entraînés à la demande par quartier.
    """
    if not _IMPORT_OK:
        print(f"[Molka] ⚠ Import échoué : {_IMPORT_ERROR}")
        return
    if eager:
        _init_dataset()
    else:
        t = threading.Thread(target=_init_dataset, daemon=True)
        t.start()


# ── Entraînement à la demande ─────────────────────────────────
def _get_or_train(quartier):
    """
    Retourne les modèles (sarimax_res, rl_res) pour un quartier.
    Entraîne si pas encore fait, sinon retourne le cache.
    Thread-safe.
    """
    with _CACHE_LOCK:
        if quartier in _CACHE:
            return _CACHE[quartier]['sarimax'], _CACHE[quartier]['rl']

    # Pas encore dans le cache → entraîner (sans verrou pour ne pas bloquer les autres quartiers)
    if not _DF_READY:
        raise RuntimeError("Dataset non prêt — réessayez dans quelques secondes")

    print(f"[Molka] ⚙ Entraînement pour '{quartier}'…")
    t0 = time.time()

    # SARIMAX pour CE quartier uniquement
    sarimax_res = train_all(_DF, quartiers=[quartier], verbose=True)

    # Q-Learning pour CE quartier uniquement
    rl_res = train_all_agents(_DF, sarimax_res['preds_base'],
                               quartiers=[quartier], verbose=True)

    elapsed = time.time() - t0
    print(f"[Molka] ✅ '{quartier}' entraîné en {elapsed:.1f}s")

    with _CACHE_LOCK:
        _CACHE[quartier] = {
            'sarimax':    sarimax_res,
            'rl':         rl_res,
            'trained_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }

    return sarimax_res, rl_res


# ── Helper : réponse d'erreur si dataset pas prêt ────────────
def _check_dataset():
    if not _IMPORT_OK:
        return jsonify({'error': f'Import échoué : {_IMPORT_ERROR}'}), 503
    if not _DF_READY:
        return jsonify({'status': 'loading', 'message': 'Dataset en cours de génération…'}), 202
    return None


# ── Routes ────────────────────────────────────────────────────

@molka_bp.route('/health')
def health():
    trained = {}
    with _CACHE_LOCK:
        for q, v in _CACHE.items():
            trained[q] = v['trained_at']
    return jsonify({
        'module':         'molka',
        'import_ok':      _IMPORT_OK,
        'dataset_ready':  _DF_READY,
        'ready':          _DF_READY,       # compat avec les tabs HTML
        'loading':        not _DF_READY,
        'quartiers':      QUARTIERS if _IMPORT_OK else [],
        'trained':        trained,          # quartiers déjà calculés
        'n_trained':      len(trained),
    })


@molka_bp.route('/quartiers')
def quartiers():
    err = _check_dataset()
    if err: return err
    out = []
    for q in QUARTIERS:
        last = _DF[_DF['quartier'] == q].sort_values('date').tail(4)
        with _CACHE_LOCK:
            is_trained = q in _CACHE
        out.append({
            'name':            q,
            'base_d':          QUARTIERS_CFG[q]['base_d'],
            'base_b':          QUARTIERS_CFG[q]['base_b'],
            'densite_actuelle': round(float(last['densite'].mean()), 1),
            'risque_actuel':    round(float(last['risque_global'].mean()), 3),
            'trained':          is_trained,
            'geo': {'center': GEO[q]['center'], 'color': GEO[q]['color']},
        })
    return jsonify({'quartiers': out})


@molka_bp.route('/sarimax', methods=['POST'])
def sarimax_forecast():
    err = _check_dataset()
    if err: return err

    data     = request.get_json(silent=True) or {}
    quartier = data.get('quartier', QUARTIERS[0])
    if quartier not in QUARTIERS:
        return jsonify({'error': f"Quartier inconnu : {quartier}"}), 400

    try:
        sarimax_res, _ = _get_or_train(quartier)
        result = forecast(sarimax_res['preds_base'], quartier)

        serie = _DF[_DF['quartier'] == quartier].sort_values('date').tail(52)
        result['historique'] = {
            'dates':   [d.isoformat() for d in serie['date']],
            'valeurs': serie['densite_chiens'].round(2).tolist(),
        }
        result['mae']  = sarimax_res['mae_scores'].get(quartier)
        result['rmse'] = sarimax_res['rmse_scores'].get(quartier)
        result['aic']  = round(sarimax_res['modeles'][quartier].aic, 1)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@molka_bp.route('/stl/<quartier>')
def stl(quartier):
    err = _check_dataset()
    if err: return err
    if quartier not in QUARTIERS:
        return jsonify({'error': f"Quartier inconnu : {quartier}"}), 400
    try:
        result = get_stl(_DF, quartier)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@molka_bp.route('/metrics')
def metrics():
    err = _check_dataset()
    if err: return err
    with _CACHE_LOCK:
        trained_quartiers = list(_CACHE.keys())

    if not trained_quartiers:
        return jsonify({'metrics': {}, 'message': 'Aucun quartier entraîné pour le moment'})

    try:
        # Agréger les métriques de tous les quartiers déjà entraînés
        out = {}
        with _CACHE_LOCK:
            for q in trained_quartiers:
                sr = _CACHE[q]['sarimax']
                out[q] = {
                    'mae':  sr['mae_scores'].get(q),
                    'rmse': sr['rmse_scores'].get(q),
                    'aic':  round(sr['modeles'][q].aic, 1) if q in sr['modeles'] else None,
                }
        return jsonify({'metrics': out})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@molka_bp.route('/simulate', methods=['POST'])
def simulate():
    err = _check_dataset()
    if err: return err

    data     = request.get_json(silent=True) or {}
    quartier = data.get('quartier', QUARTIERS[0])
    mode     = data.get('mode', 'rl')
    steps    = int(data.get('steps', HORIZON_RL))

    if quartier not in QUARTIERS:
        return jsonify({'error': f"Quartier inconnu : {quartier}"}), 400
    if mode not in ('rien', 'vacc', 'cnvr', 'rl'):
        return jsonify({'error': f"Mode invalide : {mode}"}), 400

    try:
        sarimax_res, rl_res = _get_or_train(quartier)
        ds, rs, acts = simuler_twin(
            quartier, mode=mode, steps=steps,
            df=_DF, rl_result=rl_res,
            preds_base=sarimax_res['preds_base'],
        )
        return jsonify({
            'quartier':    quartier,
            'mode':        mode,
            'steps':       steps,
            'densites':    [round(v, 2) for v in ds.tolist()],
            'risques':     [round(v, 4) for v in rs.tolist()],
            'actions':     [ACTIONS[a] for a in acts],
            'actions_idx': acts,
            'couts':       [COUT_ACTION[a] for a in acts],
            'cout_total':  sum(COUT_ACTION[a] for a in acts),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@molka_bp.route('/simulate/all', methods=['POST'])
def simulate_all():
    """Simule les 4 scénarios pour tous les quartiers."""
    err = _check_dataset()
    if err: return err

    data  = request.get_json(silent=True) or {}
    steps = int(data.get('steps', HORIZON_RL))

    try:
        resultats = {}
        for q in QUARTIERS:
            sarimax_res, rl_res = _get_or_train(q)
            resultats[q] = {}
            for mode in ['rien', 'vacc', 'cnvr', 'rl']:
                ds, rs, acts = simuler_twin(
                    q, mode=mode, steps=steps,
                    df=_DF, rl_result=rl_res,
                    preds_base=sarimax_res['preds_base'],
                )
                resultats[q][mode] = {
                    'densite_finale': round(float(ds[-1]), 1),
                    'risque_final':   round(float(rs[-1]), 3),
                    'densites':       [round(v, 2) for v in ds.tolist()],
                    'risques':        [round(v, 4) for v in rs.tolist()],
                    'color':          color_risque(float(rs[-1])),
                    'geo': {
                        'center':     GEO[q]['center'],
                        'color_base': GEO[q]['color'],
                        'radius':     GEO[q]['radius'],
                    },
                }
        return jsonify({'resultats': resultats, 'quartiers': QUARTIERS})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@molka_bp.route('/budget', methods=['POST'])
def budget():
    err = _check_dataset()
    if err: return err

    data         = request.get_json(silent=True) or {}
    quartier     = data.get('quartier', QUARTIERS[0])
    budget_total = float(data.get('budget', 6240))
    nb_semaines  = int(data.get('semaines', HORIZON_RL))

    if quartier not in QUARTIERS:
        return jsonify({'error': f"Quartier inconnu : {quartier}"}), 400
    try:
        valider_budget(budget_total, nb_semaines)
    except ValueError as e:
        return jsonify({'error': str(e), 'valid': False}), 400

    try:
        sarimax_res, rl_res = _get_or_train(quartier)
        res = generer_strategie_budget(
            quartier, budget_total,
            df=_DF, rl_result=rl_res,
            preds_base=sarimax_res['preds_base'],
            nb_semaines=nb_semaines,
        )
        res['quartier']     = quartier
        res['budget_total'] = budget_total
        res['valid']        = True
        res['densites']     = [round(v, 2) for v in res['densites']]
        res['risques']      = [round(v, 4) for v in res['risques']]
        res['budget_seuils'] = {
            'min_absolu': BUDGET_MIN_ABSOLU,
            'recommande': BUDGET_RECOMMANDE,
            'optimal':    BUDGET_OPTIMAL,
        }
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@molka_bp.route('/geo')
def geo():
    out = {q: {'center': cfg['center'], 'bounds': cfg['bounds'],
               'color': cfg['color'], 'radius': cfg['radius']}
           for q, cfg in GEO.items()}
    return jsonify({'geo': out})


@molka_bp.route('/reset/<quartier>', methods=['POST'])
def reset_quartier(quartier):
    """Vide le cache d'un quartier pour forcer un réentraînement."""
    with _CACHE_LOCK:
        if quartier in _CACHE:
            del _CACHE[quartier]
            return jsonify({'message': f"Cache '{quartier}' supprimé"})
    return jsonify({'message': f"'{quartier}' non en cache"})


# ── App standalone ────────────────────────────────────────────
if __name__ == '__main__':
    from flask import Flask
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    app = Flask(__name__)
    app.register_blueprint(molka_bp)
    init_molka(eager=True)

    print("[Molka] Démarrage sur http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=False)
