"""Figuras para la presentación al jurado. Solo lee (leaderboard público, una consulta agregada y la caché local).

    python presentacion/make_figures.py        # desde la raíz del repo -> presentacion/*.png y presentacion/datos.json

Cada figura responde a una pregunta y usa un solo color con significado (azul = nosotros / lo que se destaca;
gris = contexto). Los números salen de logs/night_state.json (tramos entre despliegues) y del leaderboard.
"""
import datetime as dt
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import monitor  # noqa: E402

OUT = Path(__file__).resolve().parent
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_LIGHT, GRAY = "#2a78d6", "#86b6ef", "#b9b8b1"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "text.color": INK,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "xtick.major.size": 0, "ytick.major.size": 0})
es = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")
from matplotlib.ticker import FuncFormatter  # noqa: E402
COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
state = json.loads((OUT.parent / "logs" / "night_state.json").read_text())
datos = {}


def fig(title, subtitle, size=(8, 4.5)):
    f, ax = plt.subplots(figsize=size, dpi=200)
    f.subplots_adjust(top=0.80, left=0.09, right=0.95, bottom=0.14)
    f.text(0.09, 0.94, title, fontsize=15, fontweight="bold", color=INK, va="top")
    f.text(0.09, 0.865, subtitle, fontsize=10.5, color=INK2, va="top")
    return f, ax


def pair(a, b):
    """Diferencia de conversión a - b por tramos (solo los tramos en que corrieron las dos), su error y las requests de a."""
    ts = [dt.datetime.strptime(x, "%Y-%m-%d %H:%M").timestamp() for x in state["periods"]] + [time.time()]
    w = wd = n = 0.0
    for t0, t1 in zip(ts, ts[1:]):
        if t1 - t0 < 900:
            continue
        vs = CACHE.setdefault((t0, int(t1) // 300), monitor.since(t0, t1)[0])
        if a in vs and b in vs and min(vs[a]["n"], vs[b]["n"]) >= 200:
            var = monitor.NOISE_SD ** 2 * (1 / vs[a]["n"] + 1 / vs[b]["n"])
            w += 1 / var; wd += (vs[a]["conv"] - vs[b]["conv"]) / var; n += vs[a]["n"]
    return (wd / w, w ** -0.5, int(n)) if w else (float("nan"), float("nan"), 0)


CACHE: dict = {}

# ------------------------------------------------------------------ 1. progreso por horas
now = int(time.time()) // 300 * 300
d = monitor._fetch(now - 3600, now)
prog = d["progress"]
me = prog.get(monitor.TEAM, [])
hours = [dt.datetime.fromtimestamp(e) for e, _ in me]
last = {t: v[-2][1] if len(v) > 1 else 0 for t, v in prog.items()}
leader = max((t for t in prog if t != monitor.TEAM), key=lambda t: last[t])
f, ax = fig("Cuánto de la mejora posible capturamos, hora a hora",
            "100 = las 10 recomendaciones perfectas · 0 = enseñar siempre los más vendidos. Solo horas completas.")
x = np.arange(len(me) - 1)
ax.plot(x, [v for _, v in prog[leader]][:len(x)], color=GRAY, lw=2, solid_capstyle="round")
ax.plot(x, [v for _, v in me][:-1], color=BLUE, lw=2.4, solid_capstyle="round")
ax.scatter([x[-1]], [me[-2][1]], s=46, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
ax.annotate(f"nosotros  {es(me[-2][1], 0)}", (x[-1], me[-2][1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=10.5, color=INK)
ax.annotate(f"mejor equipo  {es(prog[leader][len(x) - 1][1], 0)}", (x[-1], prog[leader][len(x) - 1][1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=10.5, color=INK2)
ax.set_xticks(x[::2]); ax.set_xticklabels([h.strftime("%H:%M") for h in hours[:-1]][::2])
ax.set_xlim(-0.3, len(x) + 2.6); ax.set_ylim(0, 70); ax.set_ylabel("% de la mejora máxima posible")
ax.legend(handles=[plt.Line2D([], [], color=BLUE, lw=2.4, label="team_23 (nosotros)"), plt.Line2D([], [], color=GRAY, lw=2, label="mejor equipo en la última hora")],
          loc="lower right", bbox_to_anchor=(0.86, 0.02), frameon=False, fontsize=10)
f.savefig(OUT / "1_progreso.png"); plt.close(f)
datos["progreso"] = {"horas": [h.strftime("%H:%M") for h in hours], "nosotros": [v for _, v in me], "lider": leader}

# ------------------------------------------------------------------ 2. cadena de mejoras
steps = [("v8-new-products", "v14-cf-split", "parecido entre\nproductos"), ("v14-cf-split", "v23-pop-log", "reentrenar con\nlas respuestas\nde la noche"),
         ("v23-pop-log", "v24-rate", "carritos por\ncada visita\ndel producto")]
vals = [pair(b, a) for a, b, _ in steps]
f, ax = fig("Tres mejoras, cada una demostrada frente a la anterior",
            "Puntos de conversión ganados (compras esperadas por 100 requests), con su margen de error.")
xs = np.arange(len(steps))
ax.bar(xs, [v[0] for v in vals], width=0.11, color=BLUE, zorder=2)
ax.yaxis.set_major_formatter(COMMA)
for i, (dv, se, n) in enumerate(vals):
    ax.plot([i, i], [dv - 2 * se, dv + 2 * se], color=INK2, lw=1.2, zorder=3)
    ax.annotate(f"+{es(dv)}", (i, dv), xytext=(14, -2), textcoords="offset points", fontsize=13, fontweight="bold", color=INK, va="center")
    ax.annotate(f"z = {es(dv / se, 1)} · " + f"{n:,}".replace(",", ".") + " requests", (i, 0), xytext=(0, -46), textcoords="offset points", fontsize=9, color=MUTED, ha="center")
ax.axhline(0, color=AXIS, lw=1)
ax.set_xticks(xs); ax.set_xticklabels([s[2] for s in steps], fontsize=10.5, color=INK2)
ax.set_xlim(-0.6, len(steps) - 0.4); ax.set_ylim(0, max(v[0] + 2 * v[1] for v in vals) * 1.15); ax.set_ylabel("puntos de conversión")
f.subplots_adjust(bottom=0.26)
f.savefig(OUT / "2_mejoras.png"); plt.close(f)
datos["mejoras"] = [{"de": a, "a": b, "ganancia": round(v[0], 3), "error": round(v[1], 3), "requests": v[2]} for (a, b, _), v in zip(steps, vals)]

# ------------------------------------------------------------------ 3. curva de productos nuevos
curve = [("v25-rate-nonew", 0.6), ("v24-rate", 6.5), ("v26-rate-new2x", 13), ("v27-rate-new3x", 21), ("v28-rate-learned", 32)]
cv = [(0.0, 0.0, 0) if v == "v24-rate" else pair(v, "v24-rate") for v, _ in curve]
f, ax = fig("¿Cuántos productos nuevos enseñar? Lo decidió una prueba, no la intuición",
            "Mismo modelo, cinco niveles de empuje a los productos nuevos. Diferencia frente al nivel elegido.")
xs = [e for _, e in curve]
ax.axhline(0, color=AXIS, lw=1)
for xv, (dv, se, n), (name, _) in zip(xs, cv, curve):
    best = name == "v24-rate"
    if not best:
        ax.plot([xv, xv], [dv - 2 * se, dv + 2 * se], color=GRAY, lw=1.4, zorder=2)
    ax.scatter([xv], [dv], s=70 if best else 54, color=BLUE if best else GRAY, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate("elegido" if best else es(dv).replace("-", "−"), (xv, dv), xytext=(9, 7 if best else 0), textcoords="offset points", fontsize=10.5,
                color=INK if best else INK2, va="center", fontweight="bold" if best else "normal")
ax.set_xticks(xs); ax.set_xticklabels([f"{es(e, 1) if e < 1 else int(e)} %" for e in xs])
ax.set_xlabel("huecos de la lista ocupados por productos nuevos"); ax.set_ylabel("puntos de conversión")
ax.set_xlim(-3, 37); ax.set_ylim(min(v[0] - 2 * v[1] for v in cv) - 0.08, 0.25)
ax.yaxis.set_major_formatter(COMMA)
f.savefig(OUT / "3_productos_nuevos.png"); plt.close(f)
datos["productos_nuevos"] = [{"version": n, "huecos_pct": e, "dif": round(v[0], 3), "error": round(v[1], 3), "requests": v[2]} for (n, e), v in zip(curve, cv)]

# ------------------------------------------------------------------ 4. concentración de compras
try:
    m = np.load(OUT.parent / "cache" / "live_matrix.npz")
    pb = np.sort(np.bincount(m["prod"][m["buy"]], minlength=len(m["names"])))[::-1]
    cum = 100 * np.cumsum(pb) / pb.sum()
    f, ax = fig("Unos pocos productos se llevan casi todas las compras del carrusel",
                f"{int(pb.sum()):,} compras en nuestras listas; el catálogo tiene {len(m['names']):,} productos.".replace(",", "."))
    n_show = 300
    ax.plot(np.arange(1, n_show + 1), cum[:n_show], color=BLUE, lw=2.4, solid_capstyle="round")
    ax.fill_between(np.arange(1, n_show + 1), cum[:n_show], color=BLUE, alpha=0.10, lw=0)
    for k in (10, 50, 200):
        ax.scatter([k], [cum[k - 1]], s=46, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"los {k} más comprados: {es(cum[k - 1], 0)} %", (k, cum[k - 1]), xytext=(10, -12), textcoords="offset points", fontsize=10.5, color=INK)
    ax.set_xlim(0, n_show); ax.set_ylim(0, 100); ax.set_xlabel("productos, del más comprado al menos"); ax.set_ylabel("% de las compras (acumulado)")
    f.savefig(OUT / "4_estrellas.png"); plt.close(f)
    datos["estrellas"] = {"top10": round(float(cum[9]), 1), "top50": round(float(cum[49]), 1), "top200": round(float(cum[199]), 1), "compras": int(pb.sum())}
except Exception as e:
    print("figura 4 no generada:", e)

# ------------------------------------------------------------------ 5. latencia antes y después
q = monitor.query(f"""SELECT ts::DateTime64(3) >= toDateTime64('2026-11-14 20:06:00', 3) AS despues,
quantileIf(0.5)(latency_ms::Float64, served_by::String = 'team') AS p50, quantileIf(0.99)(latency_ms::Float64, served_by::String = 'team') AS p99
FROM {monitor.TEAM}.feedback WHERE request_id != '__init__' AND ((ts::DateTime64(3) >= toDateTime64('2026-11-14 17:00:00', 3) AND ts::DateTime64(3) < toDateTime64('2026-11-14 20:04:00', 3))
   OR ts::DateTime64(3) >= toDateTime64('2026-11-14 20:06:00', 3)) GROUP BY despues""")
lat = {str(r["despues"]).lower() in ("1", "true"): (float(r["p50"]) / 1000, float(r["p99"]) / 1000) for r in q}
f, ax = fig("Responder antes: de un bucle a tres tipos de hilo",
            "Segundos desde que nace la request hasta que nuestra respuesta está escrita. El límite es de 3 segundos.", size=(8, 3.6))
f.subplots_adjust(top=0.72, bottom=0.2, left=0.31)
ax.xaxis.set_major_formatter(COMMA)
rows = [("la mitad de las respuestas", 0), ("el 99 % de las respuestas", 1)]
for label, k in rows:
    a, b = lat[False][k], lat[True][k]
    ax.plot([b, a], [k, k], color=GRAY, lw=2, zorder=1, solid_capstyle="round")
    ax.scatter([a], [k], s=70, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.scatter([b], [k], s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"{es(a, 1)} s", (a, k), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10.5, color=INK2)
    ax.annotate(f"{es(b, 1)} s", (b, k), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10.5, color=INK, fontweight="bold")
ax.axvline(3, color=INK2, lw=1)
ax.annotate("límite: 3 s", (3, 1.45), xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=10, color=INK2)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows], color=INK2, fontsize=10.5)
ax.set_ylim(-0.6, 1.7); ax.set_xlim(0, 3.3); ax.set_xlabel("segundos"); ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE_LIGHT, markersize=8, label="antes (un solo bucle)"),
                   plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=8, label="después (lectores, cálculo y emisores)")],
          loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=10)
f.savefig(OUT / "5_latencia.png"); plt.close(f)
datos["latencia_s"] = {"antes": {"p50": round(lat[False][0], 2), "p99": round(lat[False][1], 2)}, "despues": {"p50": round(lat[True][0], 2), "p99": round(lat[True][1], 2)}}

(OUT / "datos.json").write_text(json.dumps(datos, indent=1, ensure_ascii=False))
print(json.dumps(datos, ensure_ascii=False)[:1500])
