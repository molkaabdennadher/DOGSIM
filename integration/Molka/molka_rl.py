# =============================================================
# molka_rl.py — Q-Learning + Digital Twin DOGSIM-TN
# Molka Module | ESPRIT 3A IA
#
# Fonctions exportées :
#   - train_all_agents(df, preds_base)  : entraîne les 6 agents QL
#   - simuler_twin(quartier, mode, ...)  : simulation 26 semaines
#   - generer_strategie_budget(...)      : simulation avec contrainte budget
#   - valider_budget(budget)             : validation du budget
#   - ACTIONS, COUT_ACTION, BUDGET_*    : constantes
# =============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import itertools
from collections import defaultdict

from molka_data import (
    QUARTIERS, QUARTIERS_CFG, GEO,
    facteur_saison, label_saison,
)

# ── Constantes actions ────────────────────────────────────────
ACTIONS = {
    0: 'Rien faire',
    1: 'Vaccination',
    2: 'CNVR',
    3: 'Combo (Vacc+CNVR)',
}
N_ACTIONS = 4

# Coûts réels en DT tunisiens/semaine/quartier
# Sources : budget national 6M DT/an (Institut Pasteur 2024),
#           tarifs vétérinaires Tunis (Expat.com 2024)
COUT_ACTION = {0: 0, 1: 240, 2: 1000, 3: 1100}
BUDGET_MAX_SEMAINE = 1100

# Seuils budget
BUDGET_MIN_ABSOLU    =    240
BUDGET_MIN_CONSEILLE =  2_400
BUDGET_RECOMMANDE    =  6_240
BUDGET_OPTIMAL       = 26_000

# ── Impacts sur densité et risque ─────────────────────────────
IMPACT_D = {
    0: +0.007,   # Rien faire : croissance naturelle 35%/an ÷ 52 sem
    1: -0.005,   # Vaccination : effet indirect (abandon réduit)
    2: -0.005,   # CNVR : réduction directe, Bangkok benchmark
    3: -0.009,   # Combo : additif avec légère synergie
}
IMPACT_R = {
    0: +0.007,
    1: -0.016,   # OR=0.60 sur 6 mois → 40%/26sem ≈ 0.016/sem
    2: -0.008,   # réduction indirecte via densité
    3: -0.022,   # effet cumulé vaccination + CNVR
}

# ── Hyperparamètres Q-Learning ────────────────────────────────
LR         = 0.15
DISCOUNT   = 0.92
EPS_START  = 1.0
EPS_DECAY  = 0.997
EPS_MIN    = 0.05
N_EPISODES = 5000
HORIZON_RL = 26

# ── Densité de base par quartier ─────────────────────────────
D_BASE = {q: QUARTIERS_CFG[q]['base_d'] for q in QUARTIERS}


# ── Discrétisation états ─────────────────────────────────────
def _make_percentiles(df):
    """Calcule les percentiles de discrétisation sur le dataset."""
    p_d = np.percentile(df['densite'],       [20, 40, 60, 80])
    p_r = np.percentile(df['risque_global'], [20, 40, 60, 80])
    return p_d, p_r


def _disc_d(d, p_d):
    if d < p_d[0]: return 0
    if d < p_d[1]: return 1
    if d < p_d[2]: return 2
    if d < p_d[3]: return 3
    return 4

def _disc_r(r, p_r):
    if r < p_r[0]: return 0
    if r < p_r[1]: return 1
    if r < p_r[2]: return 2
    if r < p_r[3]: return 3
    return 4

def _disc_s(s):
    mapping = {'Hiver': 0, 'Ramadan': 1, 'Printemps': 2, 'Été': 3, 'Automne': 4}
    if s not in mapping:
        raise ValueError(f"Saison inconnue : '{s}'")
    return mapping[s]

def _disc_budget(budget_restant, budget_total):
    if budget_total <= 0: return 0
    ratio = budget_restant / budget_total
    if ratio > 0.75: return 3
    if ratio > 0.50: return 2
    if ratio > 0.25: return 1
    return 0

def encode(d, r, s, p_d, p_r, budget_restant=None, budget_total=None):
    """Encode l'état RL (3D ou 4D avec budget)."""
    base = (_disc_d(d, p_d), _disc_r(r, p_r), _disc_s(s))
    if budget_restant is not None and budget_total is not None:
        return base + (_disc_budget(budget_restant, budget_total),)
    return base


# ── Digital Twin ─────────────────────────────────────────────
def sarimax_natural(quartier, densite, date, preds_base):
    """Prédit la densité naturelle (sans intervention) via SARIMAX."""
    pred_dates = preds_base[quartier]['dates']
    pred_mean  = preds_base[quartier]['mean']
    diffs = np.abs((pred_dates - date).days)
    idx   = np.argmin(diffs)
    sarimax_pred = float(pred_mean[min(idx, len(pred_mean) - 1)])

    capacite_naturelle = D_BASE[quartier] * facteur_saison(date)
    retour = 0.10 * (capacite_naturelle - densite)

    return 0.5 * sarimax_pred + 0.4 * densite + 0.1 * capacite_naturelle + retour


def twin_step(quartier, densite, risque, saison, action, date, preds_base):
    """Simule un pas de temps (1 semaine) dans le Digital Twin."""
    d_nat = sarimax_natural(quartier, densite, date, preds_base)
    d_nat = max(D_BASE[quartier] * 0.4, d_nat)

    d_new = d_nat * (1 + IMPACT_D[action])
    r_new = risque * (1 + IMPACT_R[action])

    d_new = max(D_BASE[quartier] * 0.3, d_new)
    d_new = max(1.0, d_new + np.random.normal(0, abs(d_new) * 0.03))
    r_new = max(0.01, min(0.99, r_new + np.random.normal(0, 0.008)))

    next_date = date + pd.Timedelta(weeks=1)
    s_new = label_saison(next_date)
    return d_new, r_new, s_new


def _reward(d, r, action, D_MAX, R_MAX):
    """Fonction de récompense : sanitaire - budgétaire + comportemental."""
    d_norm    = min(d / D_MAX, 1.0)
    r_norm    = min(r / R_MAX, 1.0)
    cout_norm = COUT_ACTION[action] / BUDGET_MAX_SEMAINE

    base = -(1.0 * d_norm + 2.0 * r_norm)
    penalite_cout = 0.5 * cout_norm

    bonus = 0.0
    etat_critique = (d_norm > 0.60) or (r_norm > 0.60)
    if etat_critique and action == 0:
        bonus = -1.5
    elif not etat_critique and action == 3:
        bonus = -0.3

    return base - penalite_cout + bonus


# ── Entraînement Q-Learning ───────────────────────────────────
def train_all_agents(df, preds_base, verbose=True, quartiers=None):
    """
    Entraîne un agent Q-Learning par quartier.

    Args:
        df         : DataFrame issu de molka_data.generate_dataset()
        preds_base : dict SARIMAX issu de molka_sarimax.train_all()['preds_base']
        verbose    : affiche la progression
        quartiers  : liste de quartiers à entraîner (None = tous)

    Returns:
        dict {
            'agents':     {quartier: Q-table (defaultdict)},
            'histories':  {quartier: list[float]},
            'D_MAX':      float,
            'R_MAX':      float,
            'p_d':        np.array,
            'p_r':        np.array,
        }
    """
    if quartiers is None:
        quartiers = QUARTIERS

    p_d, p_r = _make_percentiles(df)
    D_MAX = float(np.percentile(df['densite'],       95))
    R_MAX = float(np.percentile(df['risque_global'], 95))

    if verbose:
        print(f"Seuils densité  (Q20/40/60/80) : {p_d.round(1)}")
        print(f"Seuils risque   (Q20/40/60/80) : {p_r.round(3)}")
        print(f"D_MAX (P95) = {D_MAX:.1f}  |  R_MAX (P95) = {R_MAX:.3f}")
        print(f"\nEntraînement Q-Learning ({len(quartiers)} quartier(s))\n")

    agents    = {}
    histories = {}

    for q in quartiers:
        Q    = defaultdict(lambda: np.zeros(N_ACTIONS))
        eps  = EPS_START
        hist = []

        dq = df[df['quartier'] == q].sort_values('date').reset_index(drop=True)

        for ep in range(N_EPISODES):
            row = dq.iloc[random.randint(0, len(dq) - 1)]
            d, r, s, date = (
                float(row['densite']), float(row['risque_global']),
                str(row['saison']),    row['date']
            )
            # Force états critiques 30% du temps
            if random.random() < 0.30:
                d = random.uniform(D_MAX * 0.75, D_MAX)
                r = random.uniform(R_MAX * 0.75, R_MAX)
                s = random.choice(['Été', 'Automne'])

            tot_r = 0.0
            for _ in range(HORIZON_RL):
                st = encode(d, r, s, p_d, p_r)
                a  = (random.randint(0, 3) if random.random() < eps
                      else int(np.argmax(Q[st])))
                d2, r2, s2 = twin_step(q, d, r, s, a, date, preds_base)
                rew = _reward(d2, r2, a, D_MAX, R_MAX)
                tot_r += rew
                st2 = encode(d2, r2, s2, p_d, p_r)
                Q[st][a] += LR * (rew + DISCOUNT * np.max(Q[st2]) - Q[st][a])
                d, r, s, date = d2, r2, s2, date + pd.Timedelta(weeks=1)

            hist.append(tot_r)
            eps = max(EPS_MIN, eps * EPS_DECAY)

        agents[q]    = Q
        histories[q] = hist

        if verbose:
            print(f"  ✅ {q:12s} | Récompense finale = {np.mean(hist[-200:]):.2f}")

    if verbose:
        print(f"\n✅ {len(quartiers)} agent(s) entraîné(s) — politiques optimales prêtes")

    return {
        'agents':    agents,
        'histories': histories,
        'D_MAX':     D_MAX,
        'R_MAX':     R_MAX,
        'p_d':       p_d,
        'p_r':       p_r,
    }


# ── Simulation Digital Twin ───────────────────────────────────
def simuler_twin(quartier, mode='rl', steps=HORIZON_RL,
                 budget_total=None, df=None, rl_result=None, preds_base=None):
    """
    Simule le Digital Twin pour un quartier sur `steps` semaines.

    Args:
        quartier     : nom du quartier
        mode         : 'rien' | 'vacc' | 'cnvr' | 'rl'
        steps        : nombre de semaines (défaut 26)
        budget_total : budget total en DT (None = sans contrainte)
        df           : DataFrame issu de generate_dataset()
        rl_result    : dict issu de train_all_agents()
        preds_base   : dict SARIMAX issu de train_all()['preds_base']

    Returns:
        (densites, risques, actions) — arrays numpy + liste d'entiers
    """
    row  = df[df['quartier'] == quartier].sort_values('date').tail(1).iloc[0]
    d, r, s, date = (
        float(row['densite']), float(row['risque_global']),
        str(row['saison']),    row['date']
    )

    Q      = rl_result['agents'][quartier]
    p_d    = rl_result['p_d']
    p_r    = rl_result['p_r']

    ds, rs, acts = [d], [r], []
    budget_restant = budget_total

    for _ in range(steps):
        st = encode(d, r, s, p_d, p_r, budget_restant, budget_total)

        if mode == 'rien':
            a = 0
        elif mode == 'vacc':
            a = 1
        elif mode == 'cnvr':
            a = 2
        else:  # rl
            if budget_restant is not None:
                faisables = [ac for ac in range(N_ACTIONS)
                             if COUT_ACTION[ac] <= budget_restant]
                if not faisables:
                    a = 0
                else:
                    q_vals = {ac: Q[st][ac] for ac in faisables}
                    a = max(q_vals, key=q_vals.get)
            else:
                a = int(np.argmax(Q[st]))

        if budget_restant is not None:
            budget_restant = max(0, budget_restant - COUT_ACTION[a])

        d, r, s = twin_step(quartier, d, r, s, a, date, preds_base)
        date += pd.Timedelta(weeks=1)
        ds.append(d); rs.append(r); acts.append(a)

    return np.array(ds), np.array(rs), acts


# ── Validation budget ─────────────────────────────────────────
def valider_budget(budget, nb_semaines=26):
    """
    Valide le budget saisi.
    Raise ValueError si invalide, retourne True sinon.
    """
    if budget <= 0:
        raise ValueError(f"Budget invalide : {budget} DT")
    if budget < BUDGET_MIN_ABSOLU:
        raise ValueError(
            f"Budget insuffisant : {budget:.0f} DT\n"
            f"Minimum absolu : {BUDGET_MIN_ABSOLU} DT (1 semaine vaccination)"
        )
    budget_min = COUT_ACTION[1] * max(1, nb_semaines // 4)
    if budget < budget_min:
        raise ValueError(
            f"Budget trop faible pour {nb_semaines} semaines : {budget:.0f} DT\n"
            f"Minimum conseillé : {budget_min} DT"
        )
    return True


# ── Stratégie avec budget ─────────────────────────────────────
def generer_strategie_budget(quartier, budget_total, df, rl_result, preds_base,
                              nb_semaines=HORIZON_RL):
    """
    Simule la politique RL en respectant le budget municipal.

    Returns:
        dict {
            densites, risques, actions, couts_sem, cout_total,
            budget_restant, action_counts
        }
    """
    row  = df[df['quartier'] == quartier].sort_values('date').tail(1).iloc[0]
    d, r, s, date = (
        float(row['densite']), float(row['risque_global']),
        str(row['saison']),    row['date']
    )

    Q   = rl_result['agents'][quartier]
    p_d = rl_result['p_d']
    p_r = rl_result['p_r']

    budget_restant = budget_total
    ds, rs, acts, couts_sem = [d], [r], [], []

    for _ in range(nb_semaines):
        st = encode(d, r, s, p_d, p_r)
        faisables = [a for a in range(N_ACTIONS)
                     if COUT_ACTION[a] <= budget_restant]
        if not faisables:
            a = 0
        else:
            q_vals = {a: Q[st][a] for a in faisables}
            a = max(q_vals, key=q_vals.get)

        budget_restant -= COUT_ACTION[a]
        d, r, s = twin_step(quartier, d, r, s, a, date, preds_base)
        date += pd.Timedelta(weeks=1)

        ds.append(d); rs.append(r)
        acts.append(a); couts_sem.append(COUT_ACTION[a])

    from collections import Counter
    action_counts = {ACTIONS[a]: cnt for a, cnt in Counter(acts).items()}

    return {
        'densites':       ds,
        'risques':        rs,
        'actions':        [ACTIONS[a] for a in acts],
        'actions_idx':    acts,
        'couts_sem':      couts_sem,
        'cout_total':     sum(couts_sem),
        'budget_restant': budget_total - sum(couts_sem),
        'action_counts':  action_counts,
    }


# ── Points GPS dans un quartier ───────────────────────────────
def points_dans_quartier(quartier, n_chiens, spread_factor=1.0):
    """Génère n_chiens points GPS dispersés dans le quartier."""
    cfg         = GEO[quartier]
    clat, clon  = cfg['center']
    r           = cfg['radius'] * spread_factor
    pts         = []
    attempts    = 0

    while len(pts) < n_chiens and attempts < n_chiens * 10:
        attempts += 1
        dlat = np.random.normal(0, r * 0.5)
        dlon = np.random.normal(0, r * 0.5)
        lat  = clat + dlat
        lon  = clon + dlon
        bmin, bmax = cfg['bounds']
        if bmin[0] <= lat <= bmax[0] and bmin[1] <= lon <= bmax[1]:
            pts.append((lat, lon))
    return pts


def color_risque(r):
    """Retourne la couleur hex correspondant au niveau de risque."""
    if r > 0.35: return '#E24B4A'
    if r > 0.22: return '#EF9F27'
    if r > 0.12: return '#f1c40f'
    return '#1D9E75'


# ── Test rapide ───────────────────────────────────────────────
if __name__ == '__main__':
    from molka_data    import generate_dataset
    from molka_sarimax import train_all

    df = generate_dataset()
    sarimax_res = train_all(df, verbose=False)
    rl_res = train_all_agents(df, sarimax_res['preds_base'], verbose=True)

    print("\nSimulation Bab Souika — mode RL:")
    ds, rs, acts = simuler_twin(
        'Bab Souika', mode='rl',
        df=df, rl_result=rl_res, preds_base=sarimax_res['preds_base']
    )
    for i in range(5):
        print(f"  Sem {i+1}: densité={ds[i+1]:.1f} risque={rs[i+1]:.3f} action={ACTIONS[acts[i]]}")
