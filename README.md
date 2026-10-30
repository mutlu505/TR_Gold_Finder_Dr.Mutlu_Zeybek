# TR_Gold_Finder_Dr.Mutlu_Zeybek

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from io import StringIO
import geopandas as gpd

# 1) Embed all data from Table 1 (all 87 rows)
raw = """deposit,province,type,rank,au_moz
Kisladag,Usak,Porphyry Au,1,17.481
Copler,Erzincan,Porphyry Au/epithermal IS,2,6.043
Halilaga,Canakkale,Porphyry Cu-Au,3,3.500
Mollakara,Agri,Carlin type?,4,3.372
Efemcukuru,Izmir,Epithermal LS,5,2.052
Sahinli,Canakkale,Epithermal IS,6,2.050
Ovacik,Izmir,Epithermal LS,7,1.955
Cukuralan,Izmir,Epithermal LS,8,1.801
Cevizlidere,Tunceli,Porphyry Cu-Au-Mo,9,1.576
Agi Dagi,Canakkale,Epithermal HS,10,1.444
Cerattepe,Artvin,VMS-Kuroko/Epithermal,11,1.195
Tac,Artvin,Epithermal HS,12,1.110
Kisacik,Canakkale,Epithermal LS,13,1.000
Yenipazar,Yozgat,VMS-Metamorphosed,14,0.956
Asikoy,Kastamonu,VMS-Cyprus,15,0.895
Corak,Artvin,Epithermal HS,16,0.760
Mastra,Gumushane,Epithermal LS,17,0.628
Ergani,Elazig,VMS-Cyprus,18,0.576
Altintepe,Ordu,Epithermal,19,0.574
Derekoy,Giresun,Epithermal,20,0.562
Kaymaz,Eskisehir,Orogenic Au-Listwanite Hosted,21,0.544
Kirazli,Canakkale,Epithermal HS,22,0.495
Muratdere,Bilecik,Porphyry Cu-Mo-Au,23,0.447
Himmetdede,Kayseri,Carlin type?,24,0.418
Karakartal,Erzincan,Porphyry Cu-Au,25,0.386
Akbaba,Canakkale,Epithermal LS,26,0.322
Oksut,Kayseri,Epithermal HS,27,0.317
Sisorta,Sivas,Epithermal HS,28,0.303
Gumushane,Artvin,Porphyry Cu-Au,29,0.289
Durusu,Artvin,Porphyry Au-Cu,30,0.289
Inlice,Konya,Epithermal HS,31,0.262
Cayeli,Rize,VMS-Kuroko,32,0.261
Murgul,Artvin,VMS-Kuroko/epithermal,33,0.257
Mayislar,Eskisehir,Porphyry Cu-Mo/Epithermal,34,0.244
Tavsan,Kutahya,Orogenic Au-Listwanite Hosted,35,0.204
Lahanos,Giresun,VMS-Kuroko,36,0.186
Kucukdere,Balikesir,Epithermal LS,37,0.169
Kiziltepe,Balikesir,Epithermal LS,38,0.168
Karaagac,Kutahya,Orogenic Au-Listwanite hosted,39,0.157
Gumuskoy,Kutahya,Epithermal,40,0.138
Kiratli,Izmir,Epithermal HS,41,0.124
Bakirtepe,Sivas,Iron oxide Cu-Au,42,0.115
Hazine Magara,Gumushane,Carbonate replacement,43,0.106
Korudanlik,Bilecik,Epithermal?,44,0.103
Arapdag,Izmir,Epithermal HS,45,0.097
Bolkardag I,Nigde,Carbonate replacement,46,0.086
Terziali,Kirsehir,Orogenic Au-Mesothermal?,47,0.080
Kaletas,Gumushane,Carlin type?,48,0.080
Sart,Manisa,Placer,49,0.074
Kirkpavli,Gumushane,Carbonate replacement,50,0.072
Akoluk,Ordu,Epithermal LS,51,0.071
Sobran,Gumushane,Epithermal,52,0.067
Beykoy,Balikesir,Carlin type?,53,0.065
Coraklik Tepe,Balikesir,Epithermal LS,54,0.059
Kisecikkoy,Hatay,VMS-Cyprus/epithermal?,55,0.058
Arapucandere,Canakkale,Epithermal IS,56,0.052
Narlica,Izmir,Epithermal LS,57,0.044
Darphane,Kars,Placer,58,0.040
Kuvarshan,Artvin,VMS-Kuroko,59,0.040
Papazlik,Balikesir,Carbonate replacement,60,0.039
Bozdag,Manisa,Orogenic Au-mesothermal,61,0.038
Agalik,Giresun,VMS-Kuroko,62,0.034
Akarsen,Artvin,VMS-Kuroko,63,0.032
Kubaslar,Balikesir,Epithermal LS,64,0.022
Seyitler,Artvin,VMS-Kuroko,65,0.017
Kayabasi,Trabzon,VMS-Kuroko,66,0.016
Bolkardag II,Nigde,Carbonate replacement,67,0.015
Suluklukoy,Bursa,Orogenic Au-Listwanite hosted,68,0.013
Bakibaba,Kastamonu,VMS-Cyprus,69,0.012
Savcilibeyit,Kirsehir,Orogenic au-mesothermal,70,0.011
Kartaldag,Canakkale,Epithermal HS,71,0.008
Kure,Izmir,Epithermal/orogenic gold,72,0.007
Emirli,Izmir,Detachment-fault related,73,0.007
Zeytindagi,Elazig,Skarn/porphyry Cu-Mo-Au,74,0.006
Isikdag,Ankara,Epithermal,75,0.006
Gelintepe,Izmir,Epithermal LS,76,0.006
Gicik,Ankara,Epithermal?,77,0.006
Karagoz,Nigde,Epithermal,78,0.005
Nazarusagi,Elazig,Epithermal,79,0.004
Madendag,Canakkale,Epithermal LS,80,0.003
Firat Batisi,Elazig,Porphyry Cu-Mo,81,0.002
Mert Golu,Kirklareli,Placer,82,0.002
Cobancesme,Bingol,Carbonate replacement,83,0.001
Akillicay,Hatay,Placer,84,0.001
Olucak,Gumushane,Epithermal LS,85,0.001
Sayaca,Ordu,Epithermal LS,86,0.001
Satilar,Aydin,Orogenic Au-mesothermal?,87,0.0002
"""
df = pd.read_csv(StringIO(raw))
df["au_moz"] = pd.to_numeric(df["au_moz"], errors="coerce").fillna(0)

# 2) Province centroids (lat, lon)
coords = {
    "Usak": (38.68, 29.41),
    "Erzincan": (39.75, 39.49),
    "Canakkale": (40.15, 26.41),
    "Agri": (39.72, 43.05),
    "Izmir": (38.42, 27.14),
    "Tunceli": (39.11, 39.55),
    "Artvin": (41.18, 41.82),
    "Yozgat": (39.82, 34.80),
    "Kastamonu": (41.38, 33.78),
    "Gumushane": (40.46, 39.48),
    "Elazig": (38.68, 39.22),
    "Ordu": (40.98, 37.88),
    "Giresun": (40.91, 38.39),
    "Eskisehir": (39.78, 30.52),
    "Bilecik": (40.14, 29.98),
    "Kayseri": (38.73, 35.49),
    "Sivas": (39.75, 37.02),
    "Konya": (37.87, 32.48),
    "Rize": (41.02, 40.52),
    "Kutahya": (39.42, 29.98),
    "Balikesir": (39.65, 27.89),
    "Nigde": (37.97, 34.68),
    "Kirsehir": (39.15, 34.16),
    "Manisa": (38.62, 27.43),
    "Hatay": (36.20, 36.16),
    "Kars": (40.60, 43.10),
    "Bursa": (40.19, 29.06),
    "Ankara": (39.93, 32.86),
    "Kirklareli": (41.73, 27.22),
    "Bingol": (38.88, 40.50),
    "Aydin": (37.85, 27.84),
    "Trabzon": (41.00, 39.72),
}
df["lat"] = df["province"].map(lambda p: coords.get(p, (np.nan, np.nan))[0])
df["lon"] = df["province"].map(lambda p: coords.get(p, (np.nan, np.nan))[1])
missing = df[df["lat"].isna()]["province"].unique()
if len(missing):
    print("Provinces without coordinates:", missing)
df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

# 3) Weights = sqrt(Au Moz), normalized (tempers Kisladag dominance)
w = np.sqrt(df["au_moz"].values)
w = w / w.sum()

# 4) KDE prospectivity surface
lat, lon = df["lat"].values, df["lon"].values
kde = gaussian_kde(np.vstack([lon, lat]), weights=w, bw_method=0.35)
gx, gy = np.meshgrid(np.linspace(25.5, 45.0, 300), np.linspace(35.5, 42.5, 200))
gz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)

# 5) Load Türkiye province boundaries (GeoJSON, WGS84)
PROV_URL = "https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json"
try:
    provinces = gpd.read_file(PROV_URL)
except Exception as e:
    print("Could not download province boundaries:", e)
    provinces = None

if provinces is not None:
    name_col = None
    for c in provinces.columns:
        if c.lower() in ("name", "il", "il_adi", "province"):
            name_col = c
            break
    if name_col is None:
        name_col = provinces.columns[0]
    provinces = provinces.rename(columns={name_col: "il"})
    provinces = provinces.to_crs(epsg=4326)

# 6) Plot
fig, ax = plt.subplots(figsize=(13, 7))

# 6a) KDE prospectivity surface
cs = ax.contourf(gx, gy, gz, levels=25, cmap="YlOrRd", alpha=0.85)
cb = plt.colorbar(cs, ax=ax, label="Relative Gold Prospectivity (KDE)", shrink=0.52)

# 6b) Province boundaries
if provinces is not None:
    provinces.boundary.plot(ax=ax, color="dimgray", linewidth=0.6, zorder=2)

# 6c) Grid (lat/lon graticule)
ax.set_xticks(np.arange(26, 46, 2))
ax.set_yticks(np.arange(36, 43, 1))
ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.4,
    color="gray",
    alpha=0.6,
    zorder=3,
)
ax.set_xlim(25.5, 45.0)
ax.set_ylim(35.5, 42.5)

# 6d) Deposit points
ax.scatter(
    lon,
    lat,
    s=20 + 120 * df["au_moz"] / df["au_moz"].max(),
    c="k",
    alpha=0.75,
    edgecolors="white",
    linewidths=0.5,
    zorder=4,
)

# 6e) Province labels
for prov, (plat, plon) in coords.items():
    ax.text(
        plon,
        plat,
        prov,
        fontsize=6.5,
        ha="center",
        va="center",
        color="navy",
        alpha=0.85,
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.55),
    )

ax.set_xlabel("Longitude (degrees East)")
ax.set_ylabel("Latitude (degrees North)")
ax.set_title("Türkiye Gold Prospectivity Map (all 87 deposits, Table 1)")
ax.set_aspect("equal")


# ------------------------------------------------------------------
# 7) MAP SCALE BAR
# ------------------------------------------------------------------
def add_scalebar(ax, length_km=100, location=(0.08, 1.50), fontsize=9):
    """
    Draw a horizontal scale bar on a lat/lon axes.
    length_km: total bar length in km
    location : (x, y) in axes fraction (0-1)
    """
    # Mean latitude of the map for the km-per-degree conversion
    lat_mid = np.mean(ax.get_ylim())
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat_mid))
    deg_len = length_km / km_per_deg_lon

    # Anchor in data coordinates
    x0 = ax.get_xlim()[0] + location[0] * (ax.get_xlim()[1] - ax.get_xlim()[0])
    y0 = ax.get_ylim()[0] + 0.07 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    # Draw a 3-segment alternating black/white bar
    n_seg = 3
    seg = deg_len / n_seg
    bar_h = 0.18  # degrees tall
    for i in range(n_seg):
        color = "black" if i % 2 == 0 else "white"
        ax.add_patch(
            plt.Rectangle(
                (x0 + i * seg, y0),
                seg,
                bar_h,
                facecolor=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=6,
            )
        )
    # Ticks & labels
    for i, lab in enumerate([0, length_km // 2, length_km]):
        xt = x0 + i * seg
        ax.plot(
            [xt, xt], [y0, y0 + bar_h + 0.06], color="black", linewidth=0.8, zorder=6
        )
        ax.text(
            xt,
            y0 + bar_h + 0.12,
            f"{lab}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            zorder=6,
        )
    ax.text(
        x0 + deg_len / 2,
        y0 - 0.18,
        f"{length_km} km",
        ha="center",
        va="top",
        fontsize=fontsize + 1,
        fontweight="bold",
        zorder=6,
    )


add_scalebar(ax, length_km=200, location=(0.06, 0.05), fontsize=8)


# ------------------------------------------------------------------
# 8) NORTH ARROW
# ------------------------------------------------------------------
def add_north_arrow(ax, location=(0.94, 0.90), size=0.07, fontsize=11):
    """
    Draw a north arrow in axes-fraction coordinates.
    location: (x, y) center of the arrow in axes fraction
    size    : arrow height in axes fraction
    """
    x = location[0]
    y = location[1]
    # Arrow shaft + head using annotate
    ax.annotate(
        "",
        xy=(x, y + size),
        xytext=(x, y - size * 0.6),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            facecolor="black", edgecolor="black", width=2.0, headwidth=9, headlength=12
        ),
        zorder=7,
    )
    # "N" label
    ax.text(
        x,
        y + size + 0.015,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        zorder=7,
    )


add_north_arrow(ax, location=(0.94, 0.86), size=0.06, fontsize=11)

plt.tight_layout()
plt.savefig("turkiye_gold_prospectivity.png", dpi=300)
plt.show()
print(f"Plotted {len(df)} deposits. Total Au resources: {df['au_moz'].sum():.2f} Moz")

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# ============================================================
# TÜRKİYE GOLD DEPOSITS – COMPLETE STATISTICAL ANALYSIS
# Self-contained: data entry + all models (frequentist + Bayesian)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# DATA (all 87 deposits from Yiğit, 2012a)
# ------------------------------------------------------------
data = [
    ("Kisladag", "Usak", "Porphyry Au", 1, 17.481),
    ("Copler", "Erzincan", "Porphyry Au/epithermal IS", 2, 6.043),
    ("Halilaga", "Canakkale", "Porphyry Cu-Au", 3, 3.500),
    ("Mollakara", "Agri", "Carlin type?", 4, 3.372),
    ("Efemcukuru", "Izmir", "Epithermal LS", 5, 2.052),
    ("Sahinli", "Canakkale", "Epithermal IS", 6, 2.050),
    ("Ovacik", "Izmir", "Epithermal LS", 7, 1.955),
    ("Cukuralan", "Izmir", "Epithermal LS", 8, 1.801),
    ("Cevizlidere", "Tunceli", "Porphyry Cu-Au-Mo", 9, 1.576),
    ("Agi Dagi", "Canakkale", "Epithermal HS", 10, 1.444),
    ("Cerattepe", "Artvin", "VMS-Kuroko/Epithermal", 11, 1.195),
    ("Tac", "Artvin", "Epithermal HS", 12, 1.110),
    ("Kisacik", "Canakkale", "Epithermal LS", 13, 1.000),
    ("Yenipazar", "Yozgat", "VMS-Metamorphosed", 14, 0.956),
    ("Asikoy", "Kastamonu", "VMS-Cyprus", 15, 0.895),
    ("Corak", "Artvin", "Epithermal HS", 16, 0.760),
    ("Mastra", "Gumushane", "Epithermal LS", 17, 0.628),
    ("Ergani", "Elazig", "VMS-Cyprus", 18, 0.576),
    ("Altintepe", "Ordu", "Epithermal", 19, 0.574),
    ("Derekoy", "Giresun", "Epithermal", 20, 0.562),
    ("Kaymaz", "Eskisehir", "Orogenic Au-Listwanite Hosted", 21, 0.544),
    ("Kirazli", "Canakkale", "Epithermal HS", 22, 0.495),
    ("Muratdere", "Bilecik", "Porphyry Cu-Mo-Au", 23, 0.447),
    ("Himmetdede", "Kayseri", "Carlin type?", 24, 0.418),
    ("Karakartal", "Erzincan", "Porphyry Cu-Au", 25, 0.386),
    ("Akbaba", "Canakkale", "Epithermal LS", 26, 0.322),
    ("Oksut", "Kayseri", "Epithermal HS", 27, 0.317),
    ("Sisorta", "Sivas", "Epithermal HS", 28, 0.303),
    ("Gumushane", "Artvin", "Porphyry Cu-Au", 29, 0.289),
    ("Durusu", "Artvin", "Porphyry Au-Cu", 30, 0.289),
    ("Inlice", "Konya", "Epithermal HS", 31, 0.262),
    ("Cayeli", "Rize", "VMS-Kuroko", 32, 0.261),
    ("Murgul", "Artvin", "VMS-Kuroko/epithermal", 33, 0.257),
    ("Mayislar", "Eskisehir", "Porphyry Cu-Mo/Epithermal", 34, 0.244),
    ("Tavsan", "Kutahya", "Orogenic Au-Listwanite Hosted", 35, 0.204),
    ("Lahanos", "Giresun", "VMS-Kuroko", 36, 0.186),
    ("Kucukdere", "Balikesir", "Epithermal LS", 37, 0.169),
    ("Kiziltepe", "Balikesir", "Epithermal LS", 38, 0.168),
    ("Karaagac", "Kutahya", "Orogenic Au-Listwanite hosted", 39, 0.157),
    ("Gumuskoy", "Kutahya", "Epithermal", 40, 0.138),
    ("Kiratli", "Izmir", "Epithermal HS", 41, 0.124),
    ("Bakirtepe", "Sivas", "Iron oxide Cu-Au", 42, 0.115),
    ("Hazine Magara", "Gumushane", "Carbonate replacement", 43, 0.106),
    ("Korudanlik", "Bilecik", "Epithermal?", 44, 0.103),
    ("Arapdag", "Izmir", "Epithermal HS", 45, 0.097),
    ("Bolkardag I", "Nigde", "Carbonate replacement", 46, 0.086),
    ("Terziali", "Kirsehir", "Orogenic Au-Mesothermal?", 47, 0.080),
    ("Kaletas", "Gumushane", "Carlin type?", 48, 0.080),
    ("Sart", "Manisa", "Placer", 49, 0.074),
    ("Kirkpavli", "Gumushane", "Carbonate replacement", 50, 0.072),
    ("Akoluk", "Ordu", "Epithermal LS", 51, 0.071),
    ("Sobran", "Gumushane", "Epithermal", 52, 0.067),
    ("Beykoy", "Balikesir", "Carlin type?", 53, 0.065),
    ("Coraklik Tepe", "Balikesir", "Epithermal LS", 54, 0.059),
    ("Kisecikkoy", "Hatay", "VMS-Cyprus/epithermal?", 55, 0.058),
    ("Arapucandere", "Canakkale", "Epithermal IS", 56, 0.052),
    ("Narlica", "Izmir", "Epithermal LS", 57, 0.044),
    ("Darphane", "Kars", "Placer", 58, 0.040),
    ("Kuvarshan", "Artvin", "VMS-Kuroko", 59, 0.040),
    ("Papazlik", "Balikesir", "Carbonate replacement", 60, 0.039),
    ("Bozdag", "Manisa", "Orogenic Au-mesothermal", 61, 0.038),
    ("Agalik", "Giresun", "VMS-Kuroko", 62, 0.034),
    ("Akarsen", "Artvin", "VMS-Kuroko", 63, 0.032),
    ("Kubaslar", "Balikesir", "Epithermal LS", 64, 0.022),
    ("Seyitler", "Artvin", "VMS-Kuroko", 65, 0.017),
    ("Kayabasi", "Trabzon", "VMS-Kuroko", 66, 0.016),
    ("Bolkardag II", "Nigde", "Carbonate replacement", 67, 0.015),
    ("Suluklukoy", "Bursa", "Orogenic Au-Listwanite hosted", 68, 0.013),
    ("Bakibaba", "Kastamonu", "VMS-Cyprus", 69, 0.012),
    ("Savcilibeyit", "Kirsehir", "Orogenic au-mesothermal", 70, 0.011),
    ("Kartaldag", "Canakkale", "Epithermal HS", 71, 0.008),
    ("Kure", "Izmir", "Epithermal/orogenic gold", 72, 0.007),
    ("Emirli", "Izmir", "Detachment-fault related", 73, 0.007),
    ("Zeytindagi", "Elazig", "Skarn/porphyry Cu-Mo-Au", 74, 0.006),
    ("Isikdag", "Ankara", "Epithermal", 75, 0.006),
    ("Gelintepe", "Izmir", "Epithermal LS", 76, 0.006),
    ("Gicik", "Ankara", "Epithermal?", 77, 0.006),
    ("Karagoz", "Nigde", "Epithermal", 78, 0.005),
    ("Nazarusagi", "Elazig", "Epithermal", 79, 0.004),
    ("Madendag", "Canakkale", "Epithermal LS", 80, 0.003),
    ("Firat Batisi", "Elazig", "Porphyry Cu-Mo", 81, 0.002),
    ("Mert Golu", "Kirklareli", "Placer", 82, 0.002),
    ("Cobancesme", "Bingol", "Carbonate replacement", 83, 0.001),
    ("Akillicay", "Hatay", "Placer", 84, 0.001),
    ("Olucak", "Gumushane", "Epithermal LS", 85, 0.001),
    ("Sayaca", "Ordu", "Epithermal LS", 86, 0.001),
    ("Satilar", "Aydin", "Orogenic Au-mesothermal?", 87, 0.0002),
]
df = pd.DataFrame(data, columns=["Deposit", "State", "Type", "Rank", "Au_Moz"])
df["Type_group"] = df["Type"].str.split(r"[/?]").str[0].str.strip()
df["log10_Au"] = np.log10(df["Au_Moz"])

# Region mapping
region_map = {
    "Izmir": "Aegean",
    "Usak": "Aegean",
    "Canakkale": "Aegean",
    "Balikesir": "Aegean",
    "Manisa": "Aegean",
    "Aydin": "Aegean",
    "Kutahya": "Aegean",
    "Artvin": "Black Sea",
    "Gumushane": "Black Sea",
    "Ordu": "Black Sea",
    "Giresun": "Black Sea",
    "Trabzon": "Black Sea",
    "Rize": "Black Sea",
    "Kastamonu": "Black Sea",
    "Ankara": "Central Anatolia",
    "Kayseri": "Central Anatolia",
    "Sivas": "Central Anatolia",
    "Yozgat": "Central Anatolia",
    "Kirsehir": "Central Anatolia",
    "Nigde": "Central Anatolia",
    "Eskisehir": "Central Anatolia",
    "Konya": "Central Anatolia",
    "Erzincan": "Eastern Anatolia",
    "Agri": "Eastern Anatolia",
    "Tunceli": "Eastern Anatolia",
    "Elazig": "Eastern Anatolia",
    "Kars": "Eastern Anatolia",
    "Bingol": "Eastern Anatolia",
    "Bilecik": "Marmara",
    "Bursa": "Marmara",
    "Kirklareli": "Marmara",
    "Hatay": "Mediterranean",
}
df["Region"] = df["State"].map(region_map).fillna("Other")

au = df["Au_Moz"].values
n = len(au)
log_au = np.log(au)
rng = np.random.default_rng(42)

print("=" * 65)
print(f"Loaded {n} deposits | total Au = {au.sum():.3f} Moz")
print("=" * 65)

# ============================================================
# 1) DESCRIPTIVES + LOG-NORMAL MLE
# ============================================================
print("\n[1] DESCRIPTIVES")
print(df["Au_Moz"].describe().round(4).to_string())
print(f"Skewness = {stats.skew(au):.3f} | Kurtosis = {stats.kurtosis(au):.3f}")
print(
    f"Shapiro-Wilk (raw)   : W={stats.shapiro(au).statistic:.3f}, p={stats.shapiro(au).pvalue:.2e}"
)
print(
    f"Shapiro-Wilk (log10) : W={stats.shapiro(np.log10(au)).statistic:.3f}, p={stats.shapiro(np.log10(au)).pvalue:.3f}"
)

mu_mle = log_au.mean()
sigma_mle = log_au.std(ddof=0)
print(f"\nLog-normal MLE: mu={mu_mle:.4f}, sigma={sigma_mle:.4f}")
print(
    f"Median = {np.exp(mu_mle):.4f} Moz | Mean = {np.exp(mu_mle + sigma_mle**2 / 2):.4f} Moz"
)

# ============================================================
# 2) BOOTSTRAP CIs
# ============================================================
print("\n[2] BOOTSTRAP CIs (B=5000)")
B = 5000
xmin80 = np.percentile(au, 80)
boot = {"mu": [], "sigma": [], "alpha": [], "median": [], "mean": []}
for _ in range(B):
    s = rng.choice(au, n, replace=True)
    ls = np.log(s)
    m, sg = ls.mean(), ls.std(ddof=0)
    boot["mu"].append(m)
    boot["sigma"].append(sg)
    boot["median"].append(np.exp(m))
    boot["mean"].append(np.exp(m + sg**2 / 2))
    tail = s[s >= xmin80]
    if len(tail) >= 3:
        boot["alpha"].append(len(tail) / np.sum(np.log(tail / xmin80)))
    else:
        boot["alpha"].append(np.nan)


def ci(arr):
    a = np.array(arr)
    a = a[~np.isnan(a)]
    return np.percentile(a, [2.5, 97.5])


for k in ["mu", "sigma", "alpha", "median", "mean"]:
    lo, hi = ci(boot[k])
    print(f"  {k:7s}: point={np.nanmean(boot[k]):7.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

# ============================================================
# 3) FREQUENTIST LNP MODEL (body log-normal + Pareto tail)
# ============================================================
print("\n[3] FREQUENTIST LNP (truncated LN body + Pareto tail)")


def lnp_nll(params, x):
    mu, sigma, alpha, u = params
    if sigma <= 0 or alpha <= 0 or u <= 0:
        return 1e10
    body = x[x < u]
    tail = x[x >= u]
    if len(body) < 2 or len(tail) < 2:
        return 1e10
    cdf_u = stats.lognorm.cdf(u, sigma, 0, np.exp(mu))
    cdf_u = np.clip(cdf_u, 1e-12, 1 - 1e-12)
    ll_b = np.sum(stats.lognorm.logpdf(body, sigma, 0, np.exp(mu)) - np.log(cdf_u))
    ll_t = np.sum(stats.pareto.logpdf(tail, alpha, scale=u))
    return -(ll_b + ll_t)


x0 = [mu_mle, sigma_mle, 1.5, np.percentile(au, 75)]
bounds = [(-5, 5), (0.1, 3), (0.5, 5), (np.percentile(au, 50), np.percentile(au, 95))]
res = optimize.minimize(lnp_nll, x0, args=(au,), method="L-BFGS-B", bounds=bounds)
mu_h, sig_h, alpha_h, u_h = res.x
nll = res.fun
k_par = 4
aic_lnp = 2 * k_par + 2 * nll
ll_pure_ln = np.sum(stats.lognorm.logpdf(au, sig_h, 0, np.exp(mu_h)))
aic_ln = 4 - 2 * ll_pure_ln
print(f"  mu={mu_h:.4f}  sigma={sig_h:.4f}  alpha={alpha_h:.4f}  u={u_h:.4f} Moz")
print(f"  AIC LNP={aic_lnp:.2f}  vs  AIC pure log-normal={aic_ln:.2f}")
print(f"  n_body={(au < u_h).sum()}, n_tail={(au >= u_h).sum()}")

# ============================================================
# 4) ZIPF / RANK-SIZE
# ============================================================
print("\n[4] RANK-SIZE (ZIPF) ANALYSIS")
ranked = df.sort_values("Au_Moz", ascending=False).reset_index(drop=True)
ranked["rank"] = np.arange(1, n + 1)
x = np.log(ranked["rank"].values)
y = np.log(ranked["Au_Moz"].values)
slope, intercept, r_val, p_val, se = stats.linregress(x, y)
tcrit = stats.t.ppf(0.975, n - 2)
print(f"  log(Au) = {intercept:.3f} {slope:+.3f}*log(rank)")
print(f"  R²={r_val**2:.3f}, p={p_val:.2e}, Zipf b'={-slope:.3f}")
print(f"  95% CI slope = [{slope - tcrit * se:.4f}, {slope + tcrit * se:.4f}]")
for k_top in [10, 20, 50, 87]:
    s_tmp = np.polyfit(
        np.log(np.arange(1, k_top + 1)), np.log(ranked["Au_Moz"].values[:k_top]), 1
    )[0]
    print(f"  top-{k_top:2d}: b'={-s_tmp:.3f} → α≈{-1 / s_tmp:.2f}")

# ============================================================
# 5) ANOVA BY TYPE AND REGION
# ============================================================
from statsmodels.formula.api import ols
import statsmodels.api as sm

print("\n[5] ANOVA by TYPE")
type_counts = df["Type_group"].value_counts()
df["Type_grp"] = df["Type_group"].where(
    df["Type_group"].isin(type_counts[type_counts >= 3].index), "Other"
)
m_t = ols("log10_Au ~ C(Type_grp)", data=df).fit()
a_t = sm.stats.anova_lm(m_t, typ=2)
ssb_t = a_t.loc["C(Type_grp)", "sum_sq"]
sst_t = a_t["sum_sq"].sum()
print(a_t.round(4).to_string())
print(f"  Eta²(Type) = {ssb_t / sst_t:.3f}")

print("\n[6] ANOVA by REGION")
m_r = ols("log10_Au ~ C(Region)", data=df).fit()
a_r = sm.stats.anova_lm(m_r, typ=2)
ssb_r = a_r.loc["C(Region)", "sum_sq"]
sst_r = a_r["sum_sq"].sum()
print(a_r.round(4).to_string())
print(f"  Eta²(Region) = {ssb_r / sst_r:.3f}")

reg = (
    df.groupby("Region")
    .agg(
        n=("Au_Moz", "size"),
        total_Moz=("Au_Moz", "sum"),
        median_Moz=("Au_Moz", "median"),
        max_Moz=("Au_Moz", "max"),
    )
    .sort_values("total_Moz", ascending=False)
)
reg["%_total"] = (reg["total_Moz"] / reg["total_Moz"].sum() * 100).round(1)
print("\nRegion summary:")
print(reg.round(3).to_string())

# ============================================================
# 7) PLOTS
# ============================================================
print("\n[7] Generating plots...")

# (a) distribution + log check
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(au, bins=30, color="steelblue", edgecolor="k")
axes[0].set_title("Au (Moz) – raw")
axes[0].set_xlabel("Au (Moz)")
axes[1].hist(np.log10(au), bins=30, color="darkorange", edgecolor="k")
axes[1].set_title("log10(Au) – log-normal check")
axes[1].set_xlabel("log10(Au)")
plt.tight_layout()
plt.savefig("plot_distribution.png", dpi=300)
plt.close()

# (b) LNP density
xs = np.logspace(np.log10(au.min() * 0.8), np.log10(au.max() * 1.5), 400)
cdf_u = stats.lognorm.cdf(u_h, sig_h, 0, np.exp(mu_h))
w_b = (au < u_h).mean()
w_t = 1 - w_b
pdf_b = w_b * stats.lognorm.pdf(xs, sig_h, 0, np.exp(mu_h)) / cdf_u
pdf_t = w_t * stats.pareto.pdf(xs, alpha_h, scale=u_h)
pdf_b[xs >= u_h] = 0
pdf_t[xs < u_h] = 0
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(
    au, bins=30, density=True, alpha=0.45, color="gray", edgecolor="k", label="data"
)
ax.plot(xs, pdf_b, "b-", lw=2, label=f"LN body (σ={sig_h:.2f})")
ax.plot(xs, pdf_t, "r-", lw=2, label=f"Pareto tail (α={alpha_h:.2f})")
ax.axvline(u_h, color="k", ls=":", label=f"u={u_h:.2f} Moz")
ax.set_xscale("log")
ax.set_xlabel("Au (Moz, log)")
ax.set_ylabel("density")
ax.set_title("LNP model")
ax.legend()
plt.tight_layout()
plt.savefig("plot_lnp.png", dpi=300)
plt.close()

# (c) Zipf
yhat = intercept + slope * x
resid = y - yhat
s_err = np.sqrt(np.sum(resid**2) / (n - 2))
x_mean = x.mean()
ci_band = tcrit * s_err * np.sqrt(1 / n + (x - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(ranked["rank"], ranked["Au_Moz"], s=28, alpha=0.7)
ordx = np.argsort(x)
ax.plot(
    np.exp(x[ordx]),
    np.exp(yhat[ordx]),
    "r-",
    lw=2,
    label=f"R²={r_val**2:.2f}, b'={-slope:.2f}",
)
ax.fill_between(
    np.exp(x[ordx]),
    np.exp(yhat[ordx] - ci_band[ordx]),
    np.exp(yhat[ordx] + ci_band[ordx]),
    color="red",
    alpha=0.15,
    label="95% CI",
)
for i in range(5):
    ax.annotate(
        ranked.loc[i, "Deposit"],
        (ranked.loc[i, "rank"], ranked.loc[i, "Au_Moz"]),
        fontsize=7,
        xytext=(4, 4),
        textcoords="offset points",
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Rank (log)")
ax.set_ylabel("Au (Moz, log)")
ax.set_title("Rank-size (Zipf)")
ax.legend()
plt.tight_layout()
plt.savefig("plot_zipf.png", dpi=300)
plt.close()

# (d) Region bar + boxplot
fig, ax = plt.subplots(figsize=(8, 4))
reg["total_Moz"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="k")
ax.set_ylabel("Total Au (Moz)")
ax.set_title("Au by region")
for i, v in enumerate(reg["total_Moz"]):
    ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("plot_region_total.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5))
df.boxplot(column="log10_Au", by="Region", ax=plt.gca())
plt.xticks(rotation=30, ha="right")
plt.suptitle("")
plt.title("log10(Au) by region")
plt.ylabel("log10(Au Moz)")
plt.tight_layout()
plt.savefig("plot_region_box.png", dpi=300)
plt.close()

# ============================================================
# 8) BAYESIAN (optional – PyMC)
# ============================================================
print("\n[8] BAYESIAN LNP (PyMC)")
try:
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt

    with pm.Model() as m:
        mu_b = pm.Normal("mu", mu=mu_mle, sigma=2)
        sigma_b = pm.HalfNormal("sigma", sigma=2)
        alpha_b = pm.HalfNormal("alpha", sigma=2)
        u_b = pm.Uniform("u", lower=0.05, upper=5.0)

        # Use scipy-based potential for stability
        def loglik(mu_v, sg_v, al_v, u_v):
            body = au[au < u_v]
            tail = au[au >= u_v]
            cdf_v = np.clip(
                stats.lognorm.cdf(u_v, sg_v, 0, np.exp(mu_v)), 1e-12, 1 - 1e-12
            )
            ll_b = np.sum(
                stats.lognorm.logpdf(body, sg_v, 0, np.exp(mu_v)) - np.log(cdf_v)
            )
            ll_t = np.sum(stats.pareto.logpdf(tail, al_v, scale=u_v))
            return ll_b + ll_t

        pm.Potential("lik", loglik(mu_b, sigma_b, alpha_b, u_b))

        trace = pm.sample(
            draws=1500,
            tune=1000,
            chains=2,
            cores=1,
            target_accept=0.95,
            random_seed=42,
            return_inferencedata=True,
            progressbar=False,
        )

    print(
        az.summary(
            trace, var_names=["mu", "sigma", "alpha", "u"], round_to=4
        ).to_string()
    )

    # Posterior predictive check
    post = trace.posterior
    mu_s = post["mu"].values.flatten()
    sigma_s = post["sigma"].values.flatten()
    alpha_s = post["alpha"].values.flatten()
    u_s = post["u"].values.flatten()

    obs_stats = {
        "mean": au.mean(),
        "median": np.median(au),
        "p90": np.percentile(au, 90),
        "p99": np.percentile(au, 99),
        "max": au.max(),
    }
    sim_stats = {k: [] for k in obs_stats}
    for i in rng.choice(len(mu_s), 300, replace=False):
        m_i, sg_i, al_i, u_i = mu_s[i], sigma_s[i], alpha_s[i], u_s[i]
        p_t = np.mean(au >= u_i)
        n_t = rng.binomial(n, p_t)
        n_b = n - n_t
        body = stats.lognorm.rvs(sg_i, 0, np.exp(m_i), size=n_b * 3)
        body = body[body < u_i][:n_b]
        tail = stats.pareto.rvs(al_i, scale=u_i, size=n_t)
        sim = np.concatenate([body, tail])
        for k, f in [
            ("mean", np.mean),
            ("median", np.median),
            ("p90", lambda a: np.percentile(a, 90)),
            ("p99", lambda a: np.percentile(a, 99)),
            ("max", np.max),
        ]:
            sim_stats[k].append(f(sim))

    print(
        f"\n{'stat':8s} {'obs':>8s} {'sim_mean':>9s} {'2.5%':>8s} {'97.5%':>8s} {'PPP':>5s}"
    )
    for k, v in obs_stats.items():
        lo, hi = np.percentile(sim_stats[k], [2.5, 97.5])
        ppp = np.mean(np.array(sim_stats[k]) >= v)
        print(
            f"{k:8s} {v:8.3f} {np.mean(sim_stats[k]):9.3f} {lo:8.3f} {hi:8.3f} {ppp:5.3f}"
        )

    # Posterior band plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(au, bins=30, density=True, alpha=0.4, color="gray", edgecolor="k")
    xs = np.logspace(np.log10(au.min() * 0.8), np.log10(au.max() * 1.5), 400)
    for i in rng.choice(len(mu_s), 150, replace=False):
        m_i, sg_i, al_i, u_i = mu_s[i], sigma_s[i], alpha_s[i], u_s[i]
        cdf_v = max(stats.lognorm.cdf(u_i, sg_i, 0, np.exp(m_i)), 1e-9)
        w_b = np.mean(au < u_i)
        w_t = 1 - w_b
        pb = w_b * stats.lognorm.pdf(xs, sg_i, 0, np.exp(m_i)) / cdf_v
        pt_ = w_t * stats.pareto.pdf(xs, al_i, scale=u_i)
        pb[xs >= u_i] = 0
        pt_[xs < u_i] = 0
        ax.plot(xs, pb + pt_, color="orange", alpha=0.04, lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Au (Moz, log)")
    ax.set_ylabel("density")
    ax.set_title("Bayesian LNP – 150 posterior density draws")
    plt.tight_layout()
    plt.savefig("plot_bayes_lnp.png", dpi=300)
    plt.close()

except ImportError:
    print("  PyMC/ArviZ not installed → skipping.  pip install pymc arviz")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 65)
print("DONE. Plots saved:")
print("  plot_distribution.png, plot_lnp.png, plot_zipf.png,")
print("  plot_region_total.png, plot_region_box.png, plot_bayes_lnp.png")
print("=" * 65)

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


