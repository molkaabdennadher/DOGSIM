# =============================================================
# train_models.py
# Entraînement de tous les modèles du projet
#
# Modèles entraînés :
#   1. RandomForestRegressor  → prédiction nb_chiens
#   2. RandomForestClassifier → classification risque (0/1/2)
#   3. GradientBoostingRegressor → prédiction nb_chiens (comparaison)
#   4. Scaler + encodeurs
#
# Usage :
#   python train_models.py
#   python train_models.py --model rf     # seulement Random Forest
#   python train_models.py --model gb     # seulement Gradient Boosting
#   python train_models.py --eval         # avec évaluation détaillée
# =============================================================

import sys
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
)
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder, label_binarize
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold,
    learning_curve
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_SYN, MDL_DIR, VIZ_DIR, RF_PARAMS, RANDOM_SEED

# Features utilisées pour les modèles
FEATURE_COLS = [
    "population",
    "nb_menages",
    "taille_menage",
    "surface_km2",
    "densite_pop",
    "nb_food_poi",
    "nb_bennes_org",
    "dist_centre",
    "zone_coeff",
]
TARGET_REG  = "nb_chiens"
TARGET_CLF  = "risk_class"

CLASS_LABELS = {0: "Faible", 1: "Moyen", 2: "Élevé"}


# ── 1. Chargement des données ──────────────────────────────────

def load_data():
    path = DATA_SYN / "tunisia_sectors_synthetic.csv"
    if not path.exists():
        print("[TRAIN] Dataset introuvable → génération automatique…")
        sys.path.insert(0, str(path.parent.parent))
        from models.generate_synthetic_data import generate_full_dataset
        generate_full_dataset()

    df = pd.read_csv(path)

    # Encodage zone_type si pas déjà fait
    if "zone_type" in df.columns and df["zone_type"].dtype == object:
        le = LabelEncoder()
        df["zone_type_enc"] = le.fit_transform(df["zone_type"])
        joblib.dump(le, MDL_DIR / "label_encoder_zone.joblib")

    print(f"[TRAIN] Dataset chargé : {len(df):,} samples, {len(FEATURE_COLS)} features")
    return df


# ── 2. Préparation des données ─────────────────────────────────

def prepare_data(df):
    X = df[FEATURE_COLS].values
    y_reg = df[TARGET_REG].values
    y_clf = df[TARGET_CLF].values

    # Split train/test 80/20
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = \
        train_test_split(X, y_reg, y_clf, test_size=0.20,
                         random_state=RANDOM_SEED, stratify=y_clf)

    # Normalisation
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    MDL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MDL_DIR / "scaler.joblib")

    print(f"[TRAIN] Train : {len(X_train)} | Test : {len(X_test)}")
    return X_train_sc, X_test_sc, y_reg_train, y_reg_test, y_clf_train, y_clf_test, scaler


# ── 3. Random Forest Regressor ─────────────────────────────────

def train_rf_regressor(X_train, X_test, y_train, y_test, evaluate=True):
    print("\n[TRAIN] ── Random Forest Regressor ──")

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred) ** 0.5) if hasattr(mean_squared_error, "__wrapped__") else (mean_squared_error(y_test, y_pred) ** 0.5)
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE  : {mae:.1f} chiens")
    print(f"  RMSE : {rmse:.1f} chiens")
    print(f"  R²   : {r2:.4f}")

    if evaluate:
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv = cross_val_score(rf, X_train, y_train, cv=kf, scoring="r2")
        print(f"  CV R²: {cv.mean():.4f} ± {cv.std():.4f}")

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS)
    print(f"\n  Feature importance (top 5) :")
    for feat, imp in importances.nlargest(5).items():
        print(f"    {feat:<25} : {imp:.4f}")

    # Sauvegarde
    joblib.dump(rf, MDL_DIR / "rf_regressor.joblib")
    print(f"\n  ✅ Sauvegardé : models/trained/rf_regressor.joblib")

    return rf, importances


# ── 4. Random Forest Classifier ───────────────────────────────

def train_rf_classifier(X_train, X_test, y_train, y_test, evaluate=True):
    print("\n[TRAIN] ── Random Forest Classifier (risque) ──")

    params_clf = RF_PARAMS.copy()
    params_clf["class_weight"] = "balanced"
    rf_clf = RandomForestClassifier(**params_clf)
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"  Accuracy : {acc:.4f}")
    print(f"\n  Rapport de classification :")
    print(classification_report(y_test, y_pred,
                                 target_names=["Faible", "Moyen", "Élevé"],
                                 digits=3))

    if evaluate:
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv = cross_val_score(rf_clf, X_train, y_train, cv=kf, scoring="accuracy")
        print(f"  CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

    joblib.dump(rf_clf, MDL_DIR / "rf_classifier.joblib")
    print(f"\n  ✅ Sauvegardé : models/trained/rf_classifier.joblib")

    return rf_clf


# ── 5. Gradient Boosting Regressor ────────────────────────────

def train_gb_regressor(X_train, X_test, y_train, y_test):
    print("\n[TRAIN] ── Gradient Boosting Regressor ──")

    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )
    gb.fit(X_train, y_train)

    y_pred = gb.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = (mean_squared_error(y_test, y_pred) ** 0.5) if hasattr(mean_squared_error, "__wrapped__") else (mean_squared_error(y_test, y_pred) ** 0.5)
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE  : {mae:.1f} chiens")
    print(f"  RMSE : {rmse:.1f} chiens")
    print(f"  R²   : {r2:.4f}")

    joblib.dump(gb, MDL_DIR / "gb_regressor.joblib")
    print(f"\n  ✅ Sauvegardé : models/trained/gb_regressor.joblib")

    return gb


# ── 6. Visualisation des résultats ────────────────────────────

def plot_training_results(rf_reg, gb_reg, rf_clf,
                           X_train, X_test,
                           y_reg_train, y_reg_test,
                           y_clf_test, importances):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#0d1117")
    BG = "#161b22"; TC = "#c9d1d9"; AC = "#58a6ff"

    def style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.tick_params(colors=TC, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")

    # ── 1. RF : Prédiction vs Réalité ─────────────────────────
    ax1 = axes[0, 0]
    style(ax1, "RF Regressor — Pred vs Réel (test set)")
    y_pred_rf = rf_reg.predict(X_test)
    ax1.scatter(y_reg_test, y_pred_rf, alpha=0.4, s=15, color=AC)
    lims = [min(y_reg_test.min(), y_pred_rf.min()),
            max(y_reg_test.max(), y_pred_rf.max())]
    ax1.plot(lims, lims, "r--", lw=1.5, label="Parfait")
    ax1.set_xlabel("Valeur réelle", color=TC, fontsize=8)
    ax1.set_ylabel("Valeur prédite", color=TC, fontsize=8)
    r2 = r2_score(y_reg_test, y_pred_rf)
    ax1.text(0.05, 0.92, f"R² = {r2:.4f}", transform=ax1.transAxes,
             color=AC, fontsize=9, fontweight="bold")
    ax1.legend(facecolor=BG, edgecolor="#30363d", labelcolor=TC, fontsize=8)

    # ── 2. GB : Prédiction vs Réalité ─────────────────────────
    ax2 = axes[0, 1]
    style(ax2, "Gradient Boosting — Pred vs Réel (test set)")
    y_pred_gb = gb_reg.predict(X_test)
    ax2.scatter(y_reg_test, y_pred_gb, alpha=0.4, s=15, color="#f78166")
    ax2.plot(lims, lims, "r--", lw=1.5)
    ax2.set_xlabel("Valeur réelle", color=TC, fontsize=8)
    ax2.set_ylabel("Valeur prédite", color=TC, fontsize=8)
    r2_gb = r2_score(y_reg_test, y_pred_gb)
    ax2.text(0.05, 0.92, f"R² = {r2_gb:.4f}", transform=ax2.transAxes,
             color="#f78166", fontsize=9, fontweight="bold")

    # ── 3. Feature Importance ─────────────────────────────────
    ax3 = axes[0, 2]
    style(ax3, "Feature Importance (Random Forest)")
    imp_sorted = importances.sort_values()
    colors_fi  = ["#f78166" if v > imp_sorted.median() else AC for v in imp_sorted.values]
    ax3.barh(imp_sorted.index, imp_sorted.values, color=colors_fi, alpha=0.85)
    ax3.set_xlabel("Importance", color=TC, fontsize=8)
    for i, (name, val) in enumerate(zip(imp_sorted.index, imp_sorted.values)):
        ax3.text(val + 0.001, i, f"{val:.3f}", va="center", color=TC, fontsize=7)
    ax3.grid(True, color="#1a2040", alpha=0.3, axis="x")

    # ── 4. Matrice de confusion ────────────────────────────────
    ax4 = axes[1, 0]
    style(ax4, "Matrice de Confusion (RF Classifier)")
    y_pred_clf = rf_clf.predict(X_test)
    cm = confusion_matrix(y_clf_test, y_pred_clf)
    im = ax4.imshow(cm, cmap="Blues", alpha=0.9)
    labels = ["Faible", "Moyen", "Élevé"]
    ax4.set_xticks(range(3))
    ax4.set_yticks(range(3))
    ax4.set_xticklabels(labels, color=TC, fontsize=8)
    ax4.set_yticklabels(labels, color=TC, fontsize=8)
    ax4.set_xlabel("Prédit", color=TC, fontsize=8)
    ax4.set_ylabel("Réel", color=TC, fontsize=8)
    for i in range(3):
        for j in range(3):
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else TC,
                     fontsize=12, fontweight="bold")

    # ── 5. Distribution erreurs RF ─────────────────────────────
    ax5 = axes[1, 1]
    style(ax5, "Distribution des Erreurs (RF vs GB)")
    errors_rf = y_pred_rf - y_reg_test
    errors_gb = y_pred_gb - y_reg_test
    ax5.hist(errors_rf, bins=40, alpha=0.6, color=AC, label=f"RF (MAE={mean_absolute_error(y_reg_test, y_pred_rf):.0f})")
    ax5.hist(errors_gb, bins=40, alpha=0.6, color="#f78166", label=f"GB (MAE={mean_absolute_error(y_reg_test, y_pred_gb):.0f})")
    ax5.axvline(0, color="white", lw=1.5, linestyle="--")
    ax5.set_xlabel("Erreur (prédit - réel)", color=TC, fontsize=8)
    ax5.set_ylabel("Fréquence", color=TC, fontsize=8)
    ax5.legend(facecolor=BG, edgecolor="#30363d", labelcolor=TC, fontsize=8)
    ax5.grid(True, color="#1a2040", alpha=0.3)

    # ── 6. Learning curve RF ───────────────────────────────────
    ax6 = axes[1, 2]
    style(ax6, "Learning Curve (Random Forest Regressor)")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_reg_train, y_reg_test])
    train_sizes, train_sc, val_sc = learning_curve(
        rf_reg, X_all, y_all,
        cv=5, scoring="r2",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )
    ax6.plot(train_sizes, train_sc.mean(axis=1), "o-", color=AC,
             label="Train R²", linewidth=2)
    ax6.fill_between(train_sizes,
                     train_sc.mean(axis=1) - train_sc.std(axis=1),
                     train_sc.mean(axis=1) + train_sc.std(axis=1),
                     alpha=0.15, color=AC)
    ax6.plot(train_sizes, val_sc.mean(axis=1), "o-", color="#3fb950",
             label="Validation R²", linewidth=2)
    ax6.fill_between(train_sizes,
                     val_sc.mean(axis=1) - val_sc.std(axis=1),
                     val_sc.mean(axis=1) + val_sc.std(axis=1),
                     alpha=0.15, color="#3fb950")
    ax6.set_xlabel("Taille du dataset d'entraînement", color=TC, fontsize=8)
    ax6.set_ylabel("R² Score", color=TC, fontsize=8)
    ax6.set_ylim(0, 1.05)
    ax6.legend(facecolor=BG, edgecolor="#30363d", labelcolor=TC, fontsize=8)
    ax6.grid(True, color="#1a2040", alpha=0.3)

    plt.suptitle("Résultats d'Entraînement — Stray Dogs Tunisia | ESPRIT 3A IA",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    out = VIZ_DIR / "training_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n[TRAIN] Visualisation sauvegardée : {out}")


# ── 7. Sauvegarde des métadonnées ─────────────────────────────

def save_metadata(rf_reg, gb_reg, rf_clf, X_test,
                   y_reg_test, y_clf_test, scaler):
    """Sauvegarde les métriques d'entraînement pour le notebook."""
    y_pred_rf  = rf_reg.predict(X_test)
    y_pred_gb  = gb_reg.predict(X_test)
    y_pred_clf = rf_clf.predict(X_test)

    metadata = {
        "rf_regressor": {
            "mae":  round(mean_absolute_error(y_reg_test, y_pred_rf), 2),
            "rmse": round((mean_squared_error(y_reg_test, y_pred_rf) ** 0.5), 2),
            "r2":   round(r2_score(y_reg_test, y_pred_rf), 4),
        },
        "gb_regressor": {
            "mae":  round(mean_absolute_error(y_reg_test, y_pred_gb), 2),
            "rmse": round((mean_squared_error(y_reg_test, y_pred_gb) ** 0.5), 2),
            "r2":   round(r2_score(y_reg_test, y_pred_gb), 4),
        },
        "rf_classifier": {
            "accuracy": round(accuracy_score(y_clf_test, y_pred_clf), 4),
        },
        "feature_cols": FEATURE_COLS,
        "target_reg": TARGET_REG,
        "target_clf": TARGET_CLF,
    }

    import json
    out = MDL_DIR / "training_metadata.json"
    with open(out, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[TRAIN] Métadonnées sauvegardées : {out}")
    return metadata


# ── 8. Main ────────────────────────────────────────────────────

def run(train_rf=True, train_gb=True, evaluate=True):
    print("=" * 60)
    print("ENTRAÎNEMENT DES MODÈLES — Stray Dogs Tunisia")
    print("=" * 60)

    df = load_data()
    (X_train, X_test,
     y_reg_train, y_reg_test,
     y_clf_train, y_clf_test,
     scaler) = prepare_data(df)

    rf_reg = rf_clf = gb_reg = None

    if train_rf:
        rf_reg, importances = train_rf_regressor(
            X_train, X_test, y_reg_train, y_reg_test, evaluate=evaluate
        )
        rf_clf = train_rf_classifier(
            X_train, X_test, y_clf_train, y_clf_test, evaluate=evaluate
        )

    if train_gb:
        gb_reg = train_gb_regressor(
            X_train, X_test, y_reg_train, y_reg_test
        )

    if rf_reg and gb_reg and rf_clf:
        plot_training_results(
            rf_reg, gb_reg, rf_clf,
            X_train, X_test,
            y_reg_train, y_reg_test,
            y_clf_test, importances
        )
        meta = save_metadata(
            rf_reg, gb_reg, rf_clf,
            X_test, y_reg_test, y_clf_test, scaler
        )

        print(f"\n{'='*60}")
        print("RÉSUMÉ ENTRAÎNEMENT")
        print(f"{'='*60}")
        print(f"  RF Regressor  R²   : {meta['rf_regressor']['r2']}")
        print(f"  RF Regressor  MAE  : {meta['rf_regressor']['mae']} chiens")
        print(f"  GB Regressor  R²   : {meta['gb_regressor']['r2']}")
        print(f"  RF Classifier Acc  : {meta['rf_classifier']['accuracy']}")
        print(f"\n  Modèles dans : models/trained/")
        print(f"  Graphiques   : visualizations/training_results.png")
        print("=" * 60)

    return rf_reg, gb_reg, rf_clf, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "gb", "all"], default="all")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    run(
        train_rf=(args.model in ["rf", "all"]),
        train_gb=(args.model in ["gb", "all"]),
        evaluate=args.eval or args.model == "all",
    )
