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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- 1) Data entry ---------------------------------------------------------
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

# --- 2) Summary statistics -------------------------------------------------
print("=== Summary statistics ===")
print(df["Au_Moz"].describe())
print(
    "\nSkewness:",
    round(stats.skew(df["Au_Moz"]), 3),
    "| Kurtosis:",
    round(stats.kurtosis(df["Au_Moz"]), 3),
)
print("Total Au (Moz):", round(df["Au_Moz"].sum(), 3))
print("Shapiro-Wilk (normality):", stats.shapiro(df["Au_Moz"]))

# --- 3) Distribution fits --------------------------------------------------
log_au = np.log10(df["Au_Moz"])
print("\n=== Log10(Au) statistics ===")
print(log_au.describe())
print("Shapiro-Wilk on log10(Au):", stats.shapiro(log_au))

# --- 4) Rank vs Au regression ----------------------------------------------
slope, intercept, r, p, se = stats.linregress(df["Rank"], log_au)
print(
    f"\nRank vs log10(Au): slope={slope:.4f}, intercept={intercept:.4f}, R²={r**2:.4f}, p={p:.2e}"
)

# --- 5) By deposit type ----------------------------------------------------
print("\n=== Top deposit types by total Au ===")
grouped = (
    df.groupby("Type_group")["Au_Moz"]
    .agg(["count", "sum", "mean"])
    .sort_values("sum", ascending=False)
)
print(grouped.head(10).round(3))

# --- 6) Plots --------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["Au_Moz"], bins=30, color="steelblue", edgecolor="k")
axes[0].set_title("Au resources (Moz) – raw")
axes[0].set_xlabel("Au (Moz)")
axes[1].hist(log_au, bins=30, color="darkorange", edgecolor="k")
axes[1].set_title("log10(Au) – log-normal check")
axes[1].set_xlabel("log10(Au Moz)")
plt.tight_layout()
plt.savefig("au_distribution.png", dpi=300)

plt.figure(figsize=(6, 4))
plt.scatter(df["Rank"], log_au, s=20)
plt.plot(df["Rank"], intercept + slope * df["Rank"], "r-", label=f"R²={r**2:.2f}")
plt.xlabel("Rank")
plt.ylabel("log10(Au Moz)")
plt.legend()
plt.title("Rank vs log10(Au)")
plt.tight_layout()
plt.savefig("au_vs_rank.png", dpi=300)

# --- 7) Top-10 Pareto check ------------------------------------------------
top10 = df.nlargest(10, "Au_Moz")["Au_Moz"].sum()
print(f"\nTop-10 deposits hold {top10 / df['Au_Moz'].sum() * 100:.1f}% of total Au")

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend (safe for scripts/servers)
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ============================================================
# 0) DATA ENTRY (all 87 deposits)
# ============================================================
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

au = df["Au_Moz"].values
log_au = np.log(au)

# ============================================================
# 1) LOG-NORMAL MLE FIT
# ============================================================
print("=" * 60)
print("1) LOG-NORMAL MLE FIT (Au in Moz)")
print("=" * 60)
mu_mle = log_au.mean()
sigma_mle = log_au.std(ddof=0)
print(f"mu_MLE    = {mu_mle:.4f}")
print(f"sigma_MLE = {sigma_mle:.4f}")
print(f"Median (exp(mu)) = {np.exp(mu_mle):.4f} Moz")
print(f"Mean   (exp(mu+σ²/2)) = {np.exp(mu_mle + sigma_mle**2 / 2):.4f} Moz")

ks_stat, ks_p = stats.kstest(au, "lognorm", args=(sigma_mle, 0, np.exp(mu_mle)))
print(f"KS test vs fitted log-normal: D={ks_stat:.4f}, p={ks_p:.4f}")

wb_params = stats.weibull_min.fit(au, floc=0)
ks_wb = stats.kstest(au, "weibull_min", args=wb_params)
print(f"Weibull fit params (c, loc, scale) = {tuple(round(p, 4) for p in wb_params)}")
print(f"KS test vs Weibull: D={ks_wb.statistic:.4f}, p={ks_wb.pvalue:.4f}")


def aic(loglik, k):
    return 2 * k - 2 * loglik


ll_ln = np.sum(stats.lognorm.logpdf(au, sigma_mle, 0, np.exp(mu_mle)))
ll_wb = np.sum(stats.weibull_min.logpdf(au, *wb_params))
print(f"AIC log-normal = {aic(ll_ln, 2):.2f}")
print(f"AIC Weibull    = {aic(ll_wb, 3):.2f}")

# ============================================================
# 2) PARETO TAIL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("2) PARETO TAIL ANALYSIS")
print("=" * 60)
threshold = np.percentile(au, 80)
tail = au[au >= threshold]
print(f"Tail threshold (80th pct) = {threshold:.4f} Moz, n_tail = {len(tail)}")
alpha_hat = len(tail) / np.sum(np.log(tail / threshold))
print(f"Pareto alpha (tail index) = {alpha_hat:.4f}")
print(
    "  alpha < 2 → infinite variance (heavy tail)"
    if alpha_hat < 2
    else "  alpha >= 2 → finite variance"
)

# Hill plot
sorted_au = np.sort(au)[::-1]
k_vals = np.arange(5, len(sorted_au))
hill = k_vals / np.sum(
    np.log(sorted_au[: len(k_vals)][:, None] / sorted_au[k_vals][:, None]), axis=1
)
plt.figure(figsize=(7, 4))
plt.plot(k_vals, hill, "b-")
plt.axhline(alpha_hat, color="r", ls="--", label=f"α={alpha_hat:.2f} (top 20%)")
plt.xlabel("k (number of top order statistics)")
plt.ylabel("Hill estimator α̂")
plt.title("Hill plot – tail index stability")
plt.legend()
plt.tight_layout()
plt.savefig("hill_plot.png", dpi=300)

# ============================================================
# 3) ANOVA ACROSS DEPOSIT TYPES
# ============================================================
print("\n" + "=" * 60)
print("3) ANOVA ACROSS DEPOSIT TYPES (log10 Au)")
print("=" * 60)
df["log10_Au"] = np.log10(df["Au_Moz"])
type_counts = df["Type_group"].value_counts()
df["Type_grp"] = df["Type_group"].where(
    df["Type_group"].isin(type_counts[type_counts >= 3].index), "Other"
)
print("Group sizes:\n", df["Type_grp"].value_counts(), "\n")

model = ols("log10_Au ~ C(Type_grp)", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table.round(4))

ss_between = anova_table.loc["C(Type_grp)", "sum_sq"]
ss_total = anova_table["sum_sq"].sum()
eta2 = ss_between / ss_total
print(f"\nEta² (effect size) = {eta2:.4f}  ({eta2 * 100:.1f}% of variance explained)")

tukey = pairwise_tukeyhsd(df["log10_Au"], df["Type_grp"], alpha=0.05)
print("\nTukey HSD post-hoc:\n", tukey.summary())

# ============================================================
# 4) Q-Q PLOTS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
stats.probplot(au, dist="lognorm", sparams=(sigma_mle, 0, np.exp(mu_mle)), plot=axes[0])
axes[0].set_title("Q-Q: log-normal fit")
stats.probplot(au, dist="weibull_min", sparams=(wb_params[0], 0), plot=axes[1])
axes[1].set_title("Q-Q: Weibull fit")
plt.tight_layout()
plt.savefig("qq_plots.png", dpi=300)

# ============================================================
# 5) BOXPLOT BY TYPE
# ============================================================
plt.figure(figsize=(10, 5))
order = df.groupby("Type_grp")["log10_Au"].median().sort_values(ascending=False).index
df.boxplot(column="log10_Au", by="Type_grp", ax=plt.gca(), positions=range(len(order)))
plt.title("log10(Au Moz) by deposit type")
plt.suptitle("")
plt.ylabel("log10(Au Moz)")
plt.xticks(range(len(order)), order, rotation=45, ha="right")
plt.tight_layout()
plt.savefig("anova_boxplot.png", dpi=300)

# ============================================================
# 6) SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY OF MODEL CHOICE")
print("=" * 60)
best = "log-normal" if aic(ll_ln, 2) < aic(ll_wb, 3) else "Weibull"
print(f"Best global fit  : {best}")
tail_msg = (
    "heavy tail, finite mean & variance"
    if alpha_hat >= 2
    else "heavy tail, finite mean, infinite variance"
)
print(f"Tail behaviour   : Pareto α = {alpha_hat:.3f} ({tail_msg})")
print(
    f"Type effect      : F-test p = {anova_table.loc['C(Type_grp)', 'PR(>F)']:.2e}, η² = {eta2:.3f}"
)

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from scipy import stats, optimize

# ============================================================
# 0) DATA ENTRY (all 87 deposits from Table 1)
# ============================================================
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

au = df["Au_Moz"].values
log_au = np.log(au)
n = len(au)
rng = np.random.default_rng(42)

# ============================================================
# 1) BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
print("=" * 65)
print("1) BOOTSTRAP CIs (10,000 resamples, percentile method)")
print("=" * 65)
B = 10_000
boot_mu, boot_sigma, boot_alpha = [], [], []
xmin_fixed = np.percentile(au, 80)

for _ in range(B):
    s = rng.choice(au, size=n, replace=True)
    ls = np.log(s)
    boot_mu.append(ls.mean())
    boot_sigma.append(ls.std(ddof=0))
    tail = s[s >= xmin_fixed]
    if len(tail) >= 3 and np.all(tail > 0):
        boot_alpha.append(len(tail) / np.sum(np.log(tail / xmin_fixed)))
    else:
        boot_alpha.append(np.nan)


def ci(arr, level=0.95):
    a = np.array(arr)
    a = a[~np.isnan(a)]
    lo, hi = np.percentile(a, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return lo, hi


for name, arr in [
    ("mu (log-mean)", boot_mu),
    ("sigma (log-sd)", boot_sigma),
    ("Pareto alpha", boot_alpha),
]:
    lo, hi = ci(arr)
    print(f"{name:15s}: point={np.nanmean(arr):.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

boot_median = np.exp(boot_mu)
boot_mean = np.exp(np.array(boot_mu) + np.array(boot_sigma) ** 2 / 2)
for name, arr in [("Median Au (Moz)", boot_median), ("Mean Au (Moz)", boot_mean)]:
    lo, hi = ci(arr)
    print(f"{name:15s}: point={np.nanmean(arr):.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

# ============================================================
# 2) COMBINED LOG-NORMAL + PARETO (LNP) TAIL MODEL
# ============================================================
print("\n" + "=" * 65)
print("2) COMBINED LOG-NORMAL BODY + PARETO TAIL (LNP)")
print("=" * 65)


def lnp_nll(params, x):
    mu, sigma, alpha, u = params
    if sigma <= 0 or alpha <= 0 or u <= 0:
        return 1e10
    body = x[x < u]
    tail = x[x >= u]
    if len(body) < 2 or len(tail) < 2:
        return 1e10
    cdf_u = stats.lognorm.cdf(u, sigma, 0, np.exp(mu))
    if cdf_u <= 0 or cdf_u >= 1:
        return 1e10
    ll_body = np.sum(stats.lognorm.logpdf(body, sigma, 0, np.exp(mu)) - np.log(cdf_u))
    ll_tail = np.sum(stats.pareto.logpdf(tail, alpha, scale=u))
    return -(ll_body + ll_tail)


x0 = [log_au.mean(), log_au.std(ddof=0), 1.5, np.percentile(au, 75)]
bounds = [
    (-5, 5),
    (0.1, 3),
    (0.5, 5),
    (float(np.percentile(au, 50)), float(np.percentile(au, 95))),
]
res = optimize.minimize(lnp_nll, x0, args=(au,), method="L-BFGS-B", bounds=bounds)
mu_h, sig_h, alpha_h, u_h = res.x
print(f"mu     = {mu_h:.4f}")
print(f"sigma  = {sig_h:.4f}")
print(f"alpha  = {alpha_h:.4f}   (tail index)")
print(f"u      = {u_h:.4f} Moz (body/tail threshold)")
print(f"NLL    = {res.fun:.2f}   |  AIC = {2 * 4 - 2 * (-res.fun):.2f}")
print(f"n_body = {(au < u_h).sum()},  n_tail = {(au >= u_h).sum()}")

ll_ln = np.sum(stats.lognorm.logpdf(au, sig_h, 0, np.exp(mu_h)))
aic_ln = 2 * 2 - 2 * ll_ln
print(f"AIC pure log-normal (full data) = {aic_ln:.2f}")
print(f"AIC LNP (body+tail, 4 params)   = {2 * 4 - 2 * (-res.fun):.2f}")

# ============================================================
# 3) LNP DENSITY PLOT
# ============================================================
xs = np.logspace(np.log10(au.min() * 0.8), np.log10(au.max() * 1.5), 400)
w_body = (au < u_h).mean()
cdf_u = stats.lognorm.cdf(u_h, sig_h, 0, np.exp(mu_h))
pdf_body = w_body * stats.lognorm.pdf(xs, sig_h, 0, np.exp(mu_h)) / cdf_u
w_tail = (au >= u_h).mean()
pdf_tail = w_tail * stats.pareto.pdf(xs, alpha_h, scale=u_h)
pdf_body[xs >= u_h] = 0
pdf_tail[xs < u_h] = 0

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(
    au, bins=30, density=True, alpha=0.45, color="gray", edgecolor="k", label="data"
)
ax.plot(xs, pdf_body, "b-", lw=2, label=f"LN body (μ={mu_h:.2f}, σ={sig_h:.2f})")
ax.plot(xs, pdf_tail, "r-", lw=2, label=f"Pareto tail (α={alpha_h:.2f})")
ax.axvline(u_h, color="k", ls=":", label=f"u={u_h:.2f} Moz")
ax.set_xscale("log")
ax.set_xlabel("Au (Moz, log scale)")
ax.set_ylabel("density")
ax.set_title("Combined log-normal + Pareto (LNP) model")
ax.legend()
plt.tight_layout()
plt.savefig("lnp_model.png", dpi=300)
plt.close()

# ============================================================
# 4) BOOTSTRAP CI FOR THE LNP PARAMETERS
# ============================================================
print("\n" + "=" * 65)
print("4) Bootstrap CIs for LNP parameters (B=500)")
print("=" * 65)
B2 = 500
boot_lnp = np.full((B2, 4), np.nan)
for b in range(B2):
    s = rng.choice(au, size=n, replace=True)
    try:
        r = optimize.minimize(
            lnp_nll, res.x, args=(s,), method="L-BFGS-B", bounds=bounds
        )
        if r.success:
            boot_lnp[b] = r.x
    except Exception:
        pass

param_names = ["mu", "sigma", "alpha", "u"]
for i, nm in enumerate(param_names):
    lo, hi = ci(boot_lnp[:, i])
    print(f"{nm:6s}: point={res.x[i]:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")

# ============================================================
# 5) TAIL PROBABILITY QUANTIFICATION
# ============================================================
print("\n" + "=" * 65)
print("5) TAIL QUANTITIES (from fitted LNP)")
print("=" * 65)
for x_thr in [1, 5, 10, 20]:
    if x_thr >= u_h:
        p_cond = (u_h / x_thr) ** alpha_h
        p_abs = w_tail * p_cond
        print(
            f"P(X > {x_thr:>2} Moz)  = {p_abs:.4f}  "
            f"(expected # deposits ≈ {p_abs * n:.1f})"
        )
x_k = 17.481
if x_k >= u_h:
    p_k = w_tail * (u_h / x_k) ** alpha_h
    print(f"\nP(X > Kisladag {x_k} Moz) = {p_k:.4f}  (~{p_k * n:.2f} deposits)")

print("\nDone. Figure saved: lnp_model.png")

XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


