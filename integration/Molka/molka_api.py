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


@molka_bp.route('/correlations')
def correlations():
    """Matrice de corrélation + distribution densité par saison (Cell 6)."""
    err = _check_dataset()
    if err: return err
    try:
        import numpy as np
        cols   = ['densite', 'taux_rage', 'taux_parvo', 'taux_lepto', 'risque_global']
        labels = ['Densité', 'Rage', 'Parvovirus', 'Lepto', 'Risque global']
        corr   = _DF[cols].corr().values.tolist()

        SAISONS = ['Hiver', 'Ramadan', 'Printemps', 'Été', 'Automne']
        COULEURS = {'Hiver':'#378ADD','Ramadan':'#7F77DD','Printemps':'#1D9E75','Été':'#E24B4A','Automne':'#EF9F27'}
        seasonal = {}
        for s in SAISONS:
            sub = _DF[_DF['saison'] == s]['densite']
            if len(sub) == 0:
                continue
            seasonal[s] = {
                'q1':    float(sub.quantile(0.25)),
                'median':float(sub.median()),
                'q3':    float(sub.quantile(0.75)),
                'min':   float(sub.min()),
                'max':   float(sub.max()),
                'mean':  float(sub.mean()),
                'color': COULEURS[s],
                'count': int(len(sub)),
            }
        return jsonify({'labels': labels, 'corr_matrix': corr, 'seasonal': seasonal, 'saisons': SAISONS})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@molka_bp.route('/timeseries')
def timeseries():
    """Serie temporelle historique : densite 6 quartiers + maladies (Cell 6)."""
    err = _check_dataset()
    if err: return err
    try:
        # Dates uniques triees
        df_sorted = _DF.sort_values('date')
        dates_list = df_sorted['date'].drop_duplicates().tolist()
        dates_all  = [d.strftime('%Y-%m-%d') for d in dates_list]

        # Densite par quartier (meme ordre que dates_all)
        quartiers_ts = {}
        for q in QUARTIERS:
            sub = df_sorted[df_sorted['quartier'] == q]['densite'].tolist()
            quartiers_ts[q] = [round(float(v), 2) for v in sub]

        # Maladies : moyenne tous quartiers, meme ordre que dates_all
        avg = df_sorted.groupby('date', sort=False)[
            ['taux_rage','taux_parvo','taux_lepto','risque_global']
        ].mean()
        # reindex sur dates_list pour garantir l ordre
        avg = avg.reindex(dates_list)
        maladies = {
            'rage':   [round(float(v), 4) for v in avg['taux_rage'].tolist()],
            'parvo':  [round(float(v), 4) for v in avg['taux_parvo'].tolist()],
            'lepto':  [round(float(v), 4) for v in avg['taux_lepto'].tolist()],
            'risque': [round(float(v), 4) for v in avg['risque_global'].tolist()],
        }

        # Shading : plages Ete et Ramadan (comparaison ASCII-safe)
        saison_sub = df_sorted[df_sorted['quartier'] == QUARTIERS[0]][['date','saison']]
        shading = []
        current = None
        for _, row in saison_sub.iterrows():
            saison_name = str(row['saison'])
            date_str    = row['date'].strftime('%Y-%m-%d')
            is_special  = saison_name in ('Ete', 'Ramadan') or saison_name.startswith('\xc9t')  # Ete / Été
            if is_special:
                color = 'rgba(255,184,0,0.10)' if 't' in saison_name.lower() else 'rgba(127,119,221,0.10)'
                if current and current['saison'] == saison_name:
                    current['end'] = date_str
                else:
                    if current:
                        shading.append(current)
                    current = {'saison': saison_name, 'start': date_str, 'end': date_str, 'color': color}
            else:
                if current:
                    shading.append(current)
                    current = None
        if current:
            shading.append(current)

        colors = {q: GEO[q]['color'] for q in QUARTIERS}

        return jsonify({
            'dates':     dates_all,
            'quartiers': quartiers_ts,
            'maladies':  maladies,
            'shading':   shading,
            'colors':    colors,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@molka_bp.route('/politique', methods=['POST'])
def politique():
    """Politique optimale Q-Learning : grille Densité × Risque par quartier (Cell 16)."""
    err = _check_dataset()
    if err: return err

    data      = request.get_json(silent=True) or {}
    quartiers = data.get('quartiers', QUARTIERS)

    try:
        import numpy as np
        AC_COLORS = {0:'#27ae60', 1:'#f39c12', 2:'#e67e22', 3:'#c0392b'}
        AC_LABELS = {0:'Rien', 1:'Vacc.', 2:'CNVR', 3:'Combo'}
        DISC_LABELS_D = ['Dens. Faible', 'Dens. Moy.', 'Dens. Élev.']
        DISC_LABELS_R = ['Risque Faible', 'Risque Moyen', 'Risque Élevé']

        result = {}
        for q in quartiers:
            _, rl_res = _get_or_train(q)
            Q   = rl_res['agents'][q]
            grid = []
            for d_disc in range(3):
                row = []
                for r_disc in range(3):
                    # saison=3 = Été (le plus critique)
                    state  = (d_disc, r_disc, 3)
                    q_vals = Q[state]
                    best_a = int(np.argmax(q_vals))
                    row.append({
                        'action':     best_a,
                        'label':      AC_LABELS[best_a],
                        'color':      AC_COLORS[best_a],
                        'q_values':   [round(float(v), 3) for v in q_vals],
                    })
                grid.append(row)
            result[q] = {
                'grid':          grid,
                'labels_densite':DISC_LABELS_D,
                'labels_risque': DISC_LABELS_R,
            }
        return jsonify({'politique': result, 'quartiers': quartiers,
                        'ac_labels': AC_LABELS, 'ac_colors': AC_COLORS})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@molka_bp.route('/scenarios/compare', methods=['POST'])
def scenarios_compare():
    """Compare 4 scénarios × N quartiers sur 26 semaines (Cell 16 part 2)."""
    err = _check_dataset()
    if err: return err

    data      = request.get_json(silent=True) or {}
    quartiers = data.get('quartiers', QUARTIERS)
    steps     = int(data.get('steps', HORIZON_RL))

    try:
        result = {}
        for q in quartiers:
            sarimax_res, rl_res = _get_or_train(q)
            result[q] = {}
            for mode in ['rien', 'vacc', 'cnvr', 'rl']:
                ds, rs, acts = simuler_twin(
                    q, mode=mode, steps=steps,
                    df=_DF, rl_result=rl_res,
                    preds_base=sarimax_res['preds_base'],
                )
                result[q][mode] = {
                    'densites': [round(float(v), 2) for v in ds],
                    'risques':  [round(float(v), 4) for v in rs],
                }
        return jsonify({'scenarios': result, 'quartiers': quartiers, 'steps': steps})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── LLM endpoint — DOGSIM-AI ─────────────────────────────────
GROQ_API_KEY = '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'
GROQ_URL     = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_MODEL   = 'llama-3.3-70b-versatile'

SYSTEM_PROMPT_LLM = """Tu es DOGSIM-AI, expert en gestion des chiens errants pour la Municipalité de Tunis.
RÔLE : Analyste décisionnel spécialisé en épidémiologie canine et intelligence artificielle urbaine.
TON : Professionnel, précis, orienté décision. Style rapport municipal.
RÈGLES :
1. Utilise UNIQUEMENT les données entre <data> et </data>.
2. Ne calcule JAMAIS toi-même des prédictions — interprète celles du système.
3. Cite toujours les valeurs numériques exactes du contexte.
4. Tes recommandations sont basées sur les résultats Q-Learning.
FORMAT (à suivre) :
📊 ANALYSE — [titre]
[2-3 phrases d'analyse]
📈 CHIFFRES CLÉS
- [indicateur] : [valeur]
🎯 INTERPRÉTATION
[1-2 phrases]
⚡ RECOMMANDATION
[Action concrète + justification RL]"""

QUARTIER_ALIASES = {
    'bab souika':'Bab Souika','bab':'Bab Souika',
    'medina':'La Medina','la medina':'La Medina',
    'bardo':'Le Bardo','le bardo':'Le Bardo',
    'ariana':'Ariana',
    'ben arous':'Ben Arous','arous':'Ben Arous',
    'marsa':'La Marsa','la marsa':'La Marsa',
}

def _parse_query_llm(query):
    import re
    q_lower = query.lower()
    quartier = next((nom for alias, nom in QUARTIER_ALIASES.items() if alias in q_lower), None)
    horizon  = 4
    m = re.search(r'(\d+)\s*(semaine|week)', q_lower)
    if m: horizon = int(m.group(1))
    else:
        m = re.search(r'(\d+)\s*(mois|month)', q_lower)
        if m: horizon = int(m.group(1)) * 4
    horizon = max(1, min(horizon, HORIZON_RL))
    intents = []
    for intent, kws in {
        'forecast' :['combien','densité dans','semaines','mois','prévoir'],
        'compare'  :['comparer','versus','vs','pire cas','meilleur'],
        'risk'     :['risque','maladie','rage','parvo','lepto','danger'],
        'recommend':['recommande','que faire','action','intervention','quel quartier'],
        'explain'  :['pourquoi','expliquer','comment','politique'],
    }.items():
        if any(kw in q_lower for kw in kws): intents.append(intent)
    return {'quartier': quartier, 'horizon': horizon, 'intents': intents or ['general'], 'raw': query}


def _build_context_llm(parsed):
    """Construit le contexte à injecter dans le prompt depuis le cache Molka."""
    import numpy as np
    ctx = {'query': parsed['raw'], 'intents': parsed['intents'], 'horizon': parsed['horizon']}

    classement = []
    with _CACHE_LOCK:
        trained = list(_CACHE.keys())

    for q in (trained or QUARTIERS[:1]):
        sub = _DF[_DF['quartier'] == q].sort_values('date').tail(8)
        d   = float(sub['densite'].mean())
        r   = float(sub['risque_global'].mean())
        prio = 'CRITIQUE' if d > 19 or r > 0.35 else 'ÉLEVÉ' if d > 13 or r > 0.22 else 'MODÉRÉ' if d > 9 or r > 0.15 else 'FAIBLE'
        classement.append({'quartier': q, 'densite': round(d,1), 'risque': round(r,3), 'priorite': prio})
    classement.sort(key=lambda x: -x['densite'])
    ctx['classement'] = classement

    q = parsed['quartier']
    if q and q in trained:
        with _CACHE_LOCK:
            sarimax_res = _CACHE[q]['sarimax']
            rl_res      = _CACHE[q]['rl']

        sub = _DF[_DF['quartier'] == q].sort_values('date').tail(8)
        d_now = float(sub['densite'].mean())
        r_now = float(sub['risque_global'].mean())
        prio  = 'CRITIQUE' if d_now > 19 or r_now > 0.35 else 'ÉLEVÉ' if d_now > 13 or r_now > 0.22 else 'MODÉRÉ' if d_now > 9 or r_now > 0.15 else 'FAIBLE'

        pred   = sarimax_res['preds_base'][q]
        idx    = min(parsed['horizon'] - 1, len(pred['mean']) - 1)
        d_fut  = round(float(pred['mean'][idx]), 1)
        ci_lo  = round(float(pred['ci_lo'][idx]), 1)
        ci_hi  = round(float(pred['ci_hi'][idx]), 1)
        delta  = d_fut - d_now
        tendance = f'HAUSSE (+{round(delta,1)})' if delta > 1.5 else f'BAISSE ({round(delta,1)})' if delta < -1.5 else 'STABLE'

        # RL scenario comparison
        scenarios = {}
        for mode in ['rien', 'cnvr', 'rl']:
            try:
                ds, rs, acts = simuler_twin(q, mode=mode, steps=parsed['horizon'],
                                             df=_DF, rl_result=rl_res,
                                             preds_base=sarimax_res['preds_base'])
                scenarios[mode] = {'densite': round(float(ds[-1]),1), 'risque': round(float(rs[-1]),3),
                                   'action_1': ACTIONS[acts[0]] if acts else 'N/A'}
            except: pass

        gain_rl = round((scenarios.get('rien',{}).get('densite', d_now) - scenarios.get('rl',{}).get('densite', d_now))
                        / max(scenarios.get('rien',{}).get('densite', d_now), 0.01) * 100, 1) if scenarios else 0

        ctx['quartier']   = q
        ctx['etat']       = {'densite': round(d_now,1), 'risque': round(r_now,3), 'priorite': prio,
                             'rage': round(float(sub['taux_rage'].mean()),3),
                             'mae': sarimax_res['mae_scores'].get(q,'N/A'),
                             'saison': str(sub['saison'].iloc[-1])}
        ctx['sarimax']    = {'densite_future': d_fut, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                             'tendance': tendance, 'horizon': parsed['horizon']}
        ctx['scenarios']  = scenarios
        ctx['gain_rl_pct']= gain_rl
    elif q:
        ctx['note'] = f"Quartier '{q}' pas encore entraîné — lance une simulation d'abord"

    return ctx


def _build_prompt_llm(ctx):
    data = '<data>\n'
    data += 'CLASSEMENT QUARTIERS :\n'
    for i, row in enumerate(ctx.get('classement', []), 1):
        data += f"  {i}. {row['quartier']:12s} | Densité={row['densite']} ch/km² | Risque={row['risque']} | Priorité={row['priorite']}\n"

    if ctx.get('quartier') and 'etat' in ctx:
        e = ctx['etat']; s = ctx['sarimax']; sc = ctx.get('scenarios', {})
        data += f"\nQUARTIER : {ctx['quartier']}\n"
        data += f"État actuel : densité={e['densite']} ch/km² | risque={e['risque']} | saison={e['saison']} | priorité={e['priorite']}\n"
        data += f"SARIMAX +{s['horizon']} sem : {s['densite_future']} ch/km² | IC95=[{s['ci_lo']}—{s['ci_hi']}] | tendance={s['tendance']}\n"
        if sc:
            data += f"Scénarios : Inaction={sc.get('rien',{}).get('densite','?')} | CNVR={sc.get('cnvr',{}).get('densite','?')} | RL={sc.get('rl',{}).get('densite','?')} ch/km²\n"
            data += f"Gain RL vs inaction : -{ctx.get('gain_rl_pct',0)}%\n"
            data += f"Action RL recommandée sem.1 : {sc.get('rl',{}).get('action_1','N/A')}\n"
    if ctx.get('note'):
        data += f"\nNote : {ctx['note']}\n"

    data += '</data>\n'
    return f"{data}\nQUESTION : {ctx['query']}\n\nRéponds en suivant ton format défini. Utilise UNIQUEMENT les données <data>."


@molka_bp.route('/llm', methods=['POST'])
def llm_chat():
    """Endpoint DOGSIM-AI — LLM Groq branché sur les données Molka."""
    import requests as req_lib
    data = request.get_json(silent=True) or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'question vide'}), 400

    try:
        parsed  = _parse_query_llm(question)
        ctx     = _build_context_llm(parsed)
        prompt  = _build_prompt_llm(ctx)

        resp = req_lib.post(
            GROQ_URL,
            headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
            json={
                'model':       GROQ_MODEL,
                'messages':    [{'role':'system','content':SYSTEM_PROMPT_LLM},
                                {'role':'user',  'content':prompt}],
                'temperature': 0.2,
                'max_tokens':  1024,
            },
            timeout=30,
        )
        answer = resp.json()['choices'][0]['message']['content']
        return jsonify({'answer': answer, 'quartier': parsed['quartier'], 'intents': parsed['intents']})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


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
