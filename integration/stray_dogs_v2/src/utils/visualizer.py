# =============================================================
# visualizer.py
# Toutes les fonctions de visualisation du projet
# =============================================================

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import VIZ_DIR, AHP_WEIGHTS

BG      = "#0d1117"
BG2     = "#161b22"
TC      = "#c9d1d9"
WHITE   = "#ffffff"
BLUE    = "#58a6ff"
RED     = "#f78166"
GREEN   = "#3fb950"
ORANGE  = "#e6a23c"
PURPLE  = "#d2a8ff"
YELLOW  = "#f0c030"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG2)
    if title:
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TC, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=TC, fontsize=8)
    ax.tick_params(colors=TC, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363d")
    ax.grid(True, color="#1a2040", alpha=0.3, linewidth=0.5)


# ── 1. Dashboard principal ────────────────────────────────────

def plot_full_dashboard(
    district_name,
    bbox,
    s1_result,
    s2_result,
    s3_result,
    bins_df      = None,
    food_poi_df  = None,
    schools_df   = None,
    parks_df     = None,
    save         = True,
):
    """
    Dashboard complet avec 6 panneaux :
    1. Carte principale (bennes + POI + feeding zones)
    2. S1 : jauge risque + chiffres clés
    3. S1 : heatmap densité canine
    4. S2 : bennes avant / après
    5. S3 : heatmap score AHP
    6. S3 : bar chart scores feeding zones
    """
    fig = plt.figure(figsize=(22, 15))
    fig.patch.set_facecolor(BG)
    gs  = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.32)

    lat_c = (bbox["lat_min"] + bbox["lat_max"]) / 2
    lon_c = (bbox["lon_min"] + bbox["lon_max"]) / 2

    # ── Titre + KPIs ─────────────────────────────────────────
    fig.text(0.5, 0.98, f"STRAY DOGS — {district_name.upper()}",
             ha="center", color=WHITE, fontsize=16, fontweight="bold")
    fig.text(0.5, 0.955, "S1: Densité Canine  |  S2: Optimisation Bennes  |  S3: Feeding Zones",
             ha="center", color="#8b949e", fontsize=9)

    kpis = [
        (f"{s1_result['nb_chiens']:,}", "Chiens estimés",    s1_result["risk_color"]),
        (s1_result["risk_label"],       "Niveau de risque",   s1_result["risk_color"]),
        (str(s2_result["stats"].get("black_spots", "N/A")),     "Points noirs",  RED),
        (str(s2_result["stats"].get("bennes_optimisees", "N/A")), "Bennes optimisées", GREEN),
        (str(len(s3_result["feeding_zones"])) if not s3_result["feeding_zones"].empty else "0",
         "Feeding zones", PURPLE),
        (f"{s1_result['confidence']*100:.0f}%", "Confiance modèle", BLUE),
    ]
    for i, (val, label, color) in enumerate(kpis):
        x = 0.08 + i * 0.155
        fig.text(x, 0.935, val,   ha="center", color=color,  fontsize=14, fontweight="bold")
        fig.text(x, 0.915, label, ha="center", color="#8b949e", fontsize=7.5)

    # ── PANNEAU 1 : Carte principale ─────────────────────────
    ax_map = fig.add_subplot(gs[0:2, 0:2])
    ax_map.set_facecolor("#0d1b2a")
    _draw_main_map(ax_map, bbox, lat_c, lon_c,
                   s1_result, s2_result, s3_result,
                   bins_df, food_poi_df, schools_df, parks_df)
    ax_map.set_title(f"Carte — {district_name}", color=WHITE, fontsize=11, fontweight="bold")
    ax_map.set_xlabel("Longitude", color=TC, fontsize=8)
    ax_map.set_ylabel("Latitude", color=TC, fontsize=8)
    ax_map.tick_params(colors=TC, labelsize=7)
    for sp in ax_map.spines.values():
        sp.set_edgecolor("#30363d")

    # ── PANNEAU 2 : Jauge risque ─────────────────────────────
    ax_gauge = fig.add_subplot(gs[0, 2])
    _draw_risk_gauge(ax_gauge, s1_result)

    # ── PANNEAU 3 : Détails S1 ────────────────────────────────
    ax_s1 = fig.add_subplot(gs[0, 3])
    _draw_s1_details(ax_s1, s1_result, bins_df, food_poi_df)

    # ── PANNEAU 4 : Bennes avant/après ────────────────────────
    ax_bins = fig.add_subplot(gs[1, 2])
    _draw_bins_comparison(ax_bins, s2_result)

    # ── PANNEAU 5 : Clusters DBSCAN ───────────────────────────
    ax_dbscan = fig.add_subplot(gs[1, 3])
    _draw_dbscan(ax_dbscan, s2_result, bbox)

    # ── PANNEAU 6 : Heatmap score AHP ─────────────────────────
    ax_ahp = fig.add_subplot(gs[2, 0:2])
    _draw_ahp_heatmap(ax_ahp, s3_result, bbox)

    # ── PANNEAU 7 : Bar chart feeding zones ──────────────────
    ax_fz = fig.add_subplot(gs[2, 2])
    _draw_fz_scores(ax_fz, s3_result)

    # ── PANNEAU 8 : Poids AHP ─────────────────────────────────
    ax_poids = fig.add_subplot(gs[2, 3])
    _draw_ahp_weights(ax_poids)

    if save:
        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        slug = district_name.lower().replace(" ", "_")
        out  = VIZ_DIR / f"dashboard_{slug}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"[VIZ] Dashboard sauvegardé : {out}")
        return str(out)
    else:
        return fig


def _draw_main_map(ax, bbox, lat_c, lon_c, s1, s2, s3,
                   bins_df, food_poi_df, schools_df, parks_df):
    """Carte principale avec tous les éléments."""

    # Zone du quartier
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (bbox["lon_min"], bbox["lat_min"]),
        bbox["lon_max"] - bbox["lon_min"],
        bbox["lat_max"] - bbox["lat_min"],
        fill=True, facecolor="#0a1628", edgecolor="#3a4060",
        linewidth=1.5, zorder=0
    )
    ax.add_patch(rect)

    # Heatmap densité canine (fond)
    risk_colors_map = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    bg_color = risk_colors_map.get(s1["risk_class"], "#58a6ff")
    center_circle = plt.Circle((lon_c, lat_c),
                                max(bbox["lon_max"]-bbox["lon_min"],
                                    bbox["lat_max"]-bbox["lat_min"]) * 0.35,
                                color=bg_color, alpha=0.08, zorder=1)
    ax.add_patch(center_circle)

    handles_legend = []

    # Parcs
    if parks_df is not None and not parks_df.empty and "lat" in parks_df.columns:
        ax.scatter(parks_df["lon"], parks_df["lat"], c=GREEN, s=40,
                   marker="P", alpha=0.8, zorder=3, label="Parc/Espace vert")
        handles_legend.append(mpatches.Patch(color=GREEN, label="Parcs"))

    # Écoles
    if schools_df is not None and not schools_df.empty and "lat" in schools_df.columns:
        ax.scatter(schools_df["lon"], schools_df["lat"], c=YELLOW, s=55,
                   marker="^", alpha=0.9, zorder=4, label="Écoles")
        handles_legend.append(mpatches.Patch(color=YELLOW, label="Écoles"))

    # Restaurants
    if food_poi_df is not None and not food_poi_df.empty and "lat" in food_poi_df.columns:
        ax.scatter(food_poi_df["lon"], food_poi_df["lat"], c=ORANGE, s=18,
                   marker="D", alpha=0.6, zorder=3)
        handles_legend.append(mpatches.Patch(color=ORANGE, label="Restaurants/Cafés"))

    # Bennes actuelles
    if bins_df is not None and not bins_df.empty and "lat" in bins_df.columns:
        b_plas = bins_df[~bins_df.get("type", pd.Series()).str.lower().str.contains("organ", na=False)]
        b_org  = bins_df[bins_df.get("type", pd.Series()).str.lower().str.contains("organ", na=True)]
        if not b_plas.empty:
            ax.scatter(b_plas["lon"], b_plas["lat"], c=BLUE, s=18,
                       marker="s", alpha=0.6, zorder=4)
        if not b_org.empty:
            ax.scatter(b_org["lon"], b_org["lat"], c=GREEN, s=18,
                       marker="o", alpha=0.6, zorder=4)
        handles_legend.append(mpatches.Patch(color=BLUE, label="Bennes plastique"))
        handles_legend.append(mpatches.Patch(color=GREEN, label="Bennes organiques"))

    # Points noirs
    if not s2["black_spots"].empty and "lat" in s2["black_spots"].columns:
        pn = s2["black_spots"]
        ax.scatter(pn["lon"], pn["lat"], c=RED, s=60, marker="X",
                   edgecolors="white", linewidth=0.5, zorder=6, alpha=0.9)
        handles_legend.append(mpatches.Patch(color=RED, label=f"Points noirs ({len(pn)})"))

    # Bennes optimisées
    if not s2["optimized_bins"].empty and "lat" in s2["optimized_bins"].columns:
        opt = s2["optimized_bins"]
        ax.scatter(opt["lon"], opt["lat"], c="#00ff88", s=30, marker="o",
                   edgecolors="white", linewidth=0.5, zorder=5, alpha=0.85)
        handles_legend.append(mpatches.Patch(color="#00ff88", label="Bennes optimisées"))

    # Feeding zones
    fz = s3["feeding_zones"]
    if not fz.empty and "lat" in fz.columns:
        for _, row in fz.iterrows():
            circle = plt.Circle((row["lon"], row["lat"]), 0.003,
                                 color=PURPLE, alpha=0.2, zorder=7)
            ax.add_patch(circle)
            ax.scatter(row["lon"], row["lat"], c=PURPLE, s=150,
                       marker="*", edgecolors="white", linewidth=0.8, zorder=8)
            ax.annotate(row["zone_id"],
                        xy=(row["lon"], row["lat"] + 0.0022),
                        color=PURPLE, fontsize=7, ha="center", fontweight="bold")
        handles_legend.append(mpatches.Patch(color=PURPLE, label="Feeding Zones"))

    ax.set_xlim(bbox["lon_min"] - 0.002, bbox["lon_max"] + 0.002)
    ax.set_ylim(bbox["lat_min"] - 0.002, bbox["lat_max"] + 0.002)

    if handles_legend:
        ax.legend(handles=handles_legend, facecolor=BG, edgecolor="#30363d",
                  labelcolor=TC, fontsize=7, loc="lower right", ncol=2)


def _draw_risk_gauge(ax, s1):
    ax.set_facecolor(BG2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Niveau de Risque Canin", color=WHITE, fontsize=10, fontweight="bold", pad=8)

    color = s1["risk_color"]
    # Cercle de fond
    circle_bg = plt.Circle((0.5, 0.52), 0.35, color="#1a2040", zorder=1)
    circle_fg = plt.Circle((0.5, 0.52), 0.32, color=color, alpha=0.2, zorder=2)
    ax.add_patch(circle_bg)
    ax.add_patch(circle_fg)

    ax.text(0.5, 0.58, s1["risk_label"], ha="center", va="center",
            color=color, fontsize=18, fontweight="bold", zorder=3)
    ax.text(0.5, 0.45, f"{s1['nb_chiens']:,} chiens", ha="center",
            color=TC, fontsize=9, zorder=3)
    ax.text(0.5, 0.36, f"Confiance : {s1['confidence']*100:.0f}%",
            ha="center", color="#8b949e", fontsize=8, zorder=3)

    # Détails
    details = s1.get("details", {})
    y_pos = 0.20
    for k, v in list(details.items())[:3]:
        ax.text(0.5, y_pos, f"{k.replace('_', ' ')}: {v}",
                ha="center", color="#8b949e", fontsize=6.5)
        y_pos -= 0.08

    ax.text(0.5, 0.03, f"Source: {s1.get('source', 'RF')}", ha="center",
            color="#555", fontsize=6)


def _draw_s1_details(ax, s1, bins_df, food_poi_df):
    _style_ax(ax, "Composition du Score S1")
    ax.axis("off")

    labels = ["Population", "Zone coeff", "Food POI", "Bennes org"]
    values = [
        s1["details"].get("ratio_chien_habitant", "N/A"),
        f"× {s1['details'].get('zone_coeff', '1.0')}",
        f"+{s1['details'].get('contribution_food_poi', 0)} chiens",
        f"+{s1['details'].get('contribution_bennes', 0)} chiens",
    ]
    colors = [BLUE, ORANGE, RED, GREEN]

    for i, (lbl, val, col) in enumerate(zip(labels, values, colors)):
        y = 0.80 - i * 0.18
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.05), 0.96, 0.14,
            boxstyle="round,pad=0.02",
            facecolor=col, alpha=0.12, edgecolor=col, linewidth=1,
        ))
        ax.text(0.08, y + 0.02, lbl, color=col, fontsize=8, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.08, y - 0.02, str(val), color=TC, fontsize=7.5,
                transform=ax.transAxes)


def _draw_bins_comparison(ax, s2):
    _style_ax(ax, "Bennes : Avant / Après Optimisation")
    stats = s2.get("stats", {})
    labels  = ["Initiales", "Points noirs", "Clusters", "Optimisées"]
    vals    = [
        stats.get("bennes_initiales", 0),
        stats.get("black_spots", 0),
        stats.get("clusters_dbscan", 0),
        stats.get("bennes_optimisees", 0),
    ]
    colors_b = [BLUE, RED, ORANGE, GREEN]
    bars = ax.bar(labels, vals, color=colors_b, alpha=0.8, edgecolor="#30363d")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(v), ha="center", color=WHITE, fontsize=9, fontweight="bold")
    ax.set_ylabel("Nombre", color=TC, fontsize=8)
    ax.tick_params(axis="x", colors=TC, labelsize=8, rotation=10)


def _draw_dbscan(ax, s2, bbox):
    _style_ax(ax, "DBSCAN — Clusters Bennes Organiques")
    clusters = s2.get("clusters", pd.DataFrame())

    if clusters.empty or "lat" not in clusters.columns:
        ax.text(0.5, 0.5, "Aucune benne\nen entrée",
                ha="center", va="center", color=TC, fontsize=10,
                transform=ax.transAxes)
        return

    cmap_cl = plt.cm.tab10
    for cl_id in sorted(clusters["cluster"].unique()):
        sub = clusters[clusters["cluster"] == cl_id]
        col = "#666" if cl_id == -1 else cmap_cl(cl_id % 10)
        lbl = "Isolées" if cl_id == -1 else f"Cluster {cl_id}"
        ax.scatter(sub["lon"], sub["lat"], color=col, s=30, alpha=0.8,
                   label=lbl, edgecolors="white" if cl_id != -1 else "none",
                   linewidth=0.3)

    ax.set_xlim(bbox["lon_min"], bbox["lon_max"])
    ax.set_ylim(bbox["lat_min"], bbox["lat_max"])
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=TC,
              fontsize=6.5, loc="lower right", ncol=2)


def _draw_ahp_heatmap(ax, s3, bbox):
    _style_ax(ax, "Score AHP — Heatmap (Feeding Zones)")

    grid  = s3.get("grid", pd.DataFrame())
    fz_df = s3.get("feeding_zones", pd.DataFrame())

    if not grid.empty and "score" in grid.columns:
        try:
            pivot = grid.pivot_table(values="score", index="lat",
                                      columns="lon", aggfunc="mean")
            ax.imshow(
                pivot.values,
                extent=[pivot.columns.min(), pivot.columns.max(),
                        pivot.index.min(), pivot.index.max()],
                origin="lower", cmap="RdYlGn", alpha=0.80,
                aspect="auto", vmin=0, vmax=1,
            )
        except Exception:
            pass

    if not fz_df.empty and "lat" in fz_df.columns:
        for _, fz in fz_df.iterrows():
            ax.scatter(fz["lon"], fz["lat"], c=PURPLE, s=120,
                       marker="*", edgecolors="white", linewidth=0.8, zorder=3)
            ax.annotate(fz["zone_id"],
                        xy=(fz["lon"], fz["lat"] + 0.0018),
                        color="white", fontsize=6.5, ha="center", fontweight="bold")

    ax.set_xlim(bbox["lon_min"], bbox["lon_max"])
    ax.set_ylim(bbox["lat_min"], bbox["lat_max"])
    ax.set_xlabel("Longitude", color=TC, fontsize=8)
    ax.set_ylabel("Latitude", color=TC, fontsize=8)


def _draw_fz_scores(ax, s3):
    _style_ax(ax, "Scores des Feeding Zones")
    fz_df = s3.get("feeding_zones", pd.DataFrame())

    if fz_df.empty or "zone_id" not in fz_df.columns:
        ax.text(0.5, 0.5, "Aucune zone\nsélectionnée",
                ha="center", va="center", color=TC, fontsize=10,
                transform=ax.transAxes)
        return

    bars = ax.bar(fz_df["zone_id"], fz_df["score_pct"],
                  color=PURPLE, alpha=0.85, edgecolor="#30363d")
    ax.set_ylabel("Score (%)", color=TC, fontsize=8)
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", rotation=35, colors=TC, labelsize=7)
    for bar, v in zip(bars, fz_df["score_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", color=WHITE, fontsize=7)


def _draw_ahp_weights(ax):
    _style_ax(ax, "Poids AHP — Critères Feeding Zones")
    labels = {
        "dog_density":   "Densité canine",
        "dist_schools":  "Éloignement écoles",
        "near_green":    "Proximité parc",
        "dist_food_poi": "Éloignement restaus",
        "dist_org_bins": "Éloignement bennes",
        "road_access":   "Accessibilité route",
    }
    names  = list(labels.values())
    values = list(AHP_WEIGHTS.values())
    colors_pie = [RED, ORANGE, GREEN, BLUE, PURPLE, YELLOW]

    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors_pie,
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"linewidth": 1.5, "edgecolor": BG},
        textprops={"color": WHITE, "fontsize": 7},
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_fontweight("bold")

    legend_patches = [mpatches.Patch(color=c, label=n)
                      for c, n in zip(colors_pie, names)]
    ax.legend(handles=legend_patches, facecolor=BG, edgecolor="#30363d",
              labelcolor=TC, fontsize=6.5, loc="lower center",
              bbox_to_anchor=(0.5, -0.35), ncol=2)


# ── 2. Plot entraînement (appelé depuis train_models) ─────────

def plot_interactive_map_html(district_name, bbox, s1, s2, s3,
                               bins_df=None, food_poi_df=None,
                               schools_df=None, parks_df=None):
    """
    Génère une carte HTML interactive avec Folium.
    Nécessite : pip install folium
    """
    try:
        import folium

        lat_c = (bbox["lat_min"] + bbox["lat_max"]) / 2
        lon_c = (bbox["lon_min"] + bbox["lon_max"]) / 2

        m = folium.Map(location=[lat_c, lon_c], zoom_start=14,
                       tiles="CartoDB dark_matter")

        # Bennes organiques
        if bins_df is not None and not bins_df.empty:
            for _, row in bins_df.iterrows():
                color = "red" if row.get("type", "").lower().find("organ") >= 0 else "blue"
                folium.CircleMarker(
                    [row["lat"], row["lon"]], radius=5,
                    color=color, fill=True, fill_opacity=0.7,
                    popup=f"Benne {row.get('type', '')}",
                ).add_to(m)

        # Points noirs
        if not s2["black_spots"].empty:
            for _, row in s2["black_spots"].iterrows():
                folium.Marker(
                    [row["lat"], row["lon"]],
                    icon=folium.Icon(color="red", icon="exclamation-sign"),
                    popup="⚠️ Point Noir",
                ).add_to(m)

        # Feeding zones
        if not s3["feeding_zones"].empty:
            for _, row in s3["feeding_zones"].iterrows():
                folium.Marker(
                    [row["lat"], row["lon"]],
                    icon=folium.Icon(color="purple", icon="star"),
                    popup=f"{row['zone_id']} — Score: {row['score_pct']}%",
                ).add_to(m)
                folium.Circle(
                    [row["lat"], row["lon"]], radius=400,
                    color="purple", fill=True, fill_opacity=0.1,
                ).add_to(m)

        # Écoles
        if schools_df is not None and not schools_df.empty:
            for _, row in schools_df.iterrows():
                folium.Marker(
                    [row["lat"], row["lon"]],
                    icon=folium.Icon(color="orange", icon="education"),
                    popup=row.get("nom", "École"),
                ).add_to(m)

        VIZ_DIR.mkdir(parents=True, exist_ok=True)
        slug = district_name.lower().replace(" ", "_")
        out  = VIZ_DIR / f"map_{slug}.html"
        m.save(str(out))
        print(f"[VIZ] Carte interactive sauvegardée : {out}")
        return str(out)

    except ImportError:
        print("[VIZ] folium non installé → carte HTML non générée")
        return None
