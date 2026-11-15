"""Escribe el guion (guion.md) y las diapositivas (presentacion.html) para el jurado con los números actuales.

    python presentacion/make_figures.py && python presentacion/make_guion.py     # desde la raíz del repo

El jurado puntúa (30 puntos): método (10), aprender del stream (10), uso de RawTree (5) y comunicación (5).
Cada diapositiva lleva una idea, una figura como mucho y lo que hay que decir, en frases cortas.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
d = json.loads((OUT / "datos.json").read_text())
st = json.loads((OUT.parent / "logs" / "night_state.json").read_text())
es = lambda x, n=2: f"{x:.{n}f}".replace(".", ",")
champ = st["champion"]
g = d["mejoras"]
prog = d["progreso"]["nosotros"]
best_hour = max(prog[:-1])
lat = d["latencia_s"]
tot = d.get("mejora_total", {"ganancia": sum(x["ganancia"] for x in g), "error": 0.0})
total_gain = tot["ganancia"]
n_clear = sum(bool(x.get("clara")) for x in g)
ruido = d.get("ruido", {"total": 5.9})
hist = {x["grupo"]: x["tasa_pct"] for x in d.get("historial", [])}
h_buy, h_never = hist.get("lo compró desde un carrusel antiguo"), hist.get("nunca se lo habían enseñado")
uses_history = champ.startswith("v33")
fr = st.get("final_round")
seg = d.get("segmentos")
if seg and seg.get("delante", {}).get("n"):
    gap_txt = (f"Puesto {seg['puesto']} de {seg['equipos']}. Los {seg['delante']['n']} equipos de delante nos sacan unas décimas en los dos frentes: "
               f"cold {es(seg['nosotros']['cold'])} % frente a {es(seg['delante']['cold'][0])} %; conocidos {es(seg['nosotros']['conocidos'])} % frente a {es(seg['delante']['conocidos'][0])} %.")
    ref_txt = (f"A la referencia de los organizadores la ganamos en cold ({es(seg['nosotros']['cold'])} % frente a {es(seg['referencia']['cold'])} %) "
               f"y perdemos con conocidos ({es(seg['nosotros']['conocidos'])} % frente a {es(seg['referencia']['conocidos'])} %).")
else:
    gap_txt = ref_txt = None
NUM = {1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco"}
steps_txt = (f"{NUM.get(len(g), len(g)).capitalize()} pasos, +{es(total_gain)} puntos en total (±{es(2 * tot['error'])}). "
             + (f"{NUM.get(n_clear, n_clear).capitalize()} son claros por sí solos; " + ("el otro queda" if len(g) - n_clear == 1 else "los otros quedan") + " dentro del margen."
                if n_clear < len(g) else "Todos claros por sí solos."))
history_bullet = (f"Lo que cada comprador hizo en el sistema antiguo: si compró un producto desde un carrusel, lo compra {es(h_buy / h_never, 0)} veces más que uno que nunca vio."
                  if h_buy and h_never else "Lo que cada comprador hizo en los carruseles del sistema antiguo.")

slides = [
 {"title": "team_23 · Un recomendador que aprende mientras atiende",
  "bullets": [f"Capturamos el {es(best_hour, 0)} % de la mejora máxima posible (0 = los más vendidos, 100 = la lista perfecta).",
              "Respondemos en 1 segundo de mediana, con el límite en 3.",
              f"De la versión del sábado a la final: +{es(total_gain)} puntos de conversión (±{es(2 * tot['error'])}), medidos en vivo."],
  "say": "Somos cuatro y empezamos de cero el sábado por la mañana. Nuestra idea ha sido sencilla: un modelo que se pueda explicar, y que cada cambio se gane su sitio con datos en vivo y con su margen de error."},
 {"title": "Cómo funciona: una request, 10 productos, 1 milisegundo de cálculo",
  "figure": "5_latencia.png",
  "bullets": ["Memoria de cada comprador: visto, carrito y comprado (60 días + hoy).",
              "En cada request se puntúan los ~5.400 productos: señales × pesos.",
              f"Lectura, cálculo y envío en hilos separados: de {es(lat['antes']['p50'], 1)} a {es(lat['despues']['p50'], 1)} s de mediana.",
              "Cada despliegue pasa una prueba con requests reales y tiene vuelta atrás."],
  "say": "La fiabilidad va primero: si no respondes a tiempo, da igual lo bueno que sea el modelo. Lo ya comprado y lo agotado se quitan de la lista. Tuvimos tres paradas el sábado por desplegar a mano; de ahí salió el despliegue con comprobación previa y vuelta atrás con un comando."},
 {"title": "El modelo, en ideas que se pueden leer",
  **({"figure": "6_historial.png"} if uses_history and (OUT / "6_historial.png").exists() else {}),
  "bullets": ["Lo que el comprador ya hizo: su carrito sube; lo que ya compró, fuera (en vivo no se recompra).",
              "Lo que acaba de ver en esta sesión.",
              "Parecido entre productos: \"quien metió X en el carrito o lo compró, también Y\".",
              "Lo bueno que es el producto: carritos por cada visita, no ventas totales."]
             + ([history_bullet] if uses_history else [])
             + ["Pesos distintos para compradores nuevos y conocidos."],
  "say": "Es una suma de señales con pesos. Lo elegimos así a propósito: se entrena en segundos, responde en un milisegundo y cuando algo va mal podemos mirar qué peso lo causa. A los compradores nuevos les mueve lo que convence a todos; a los conocidos, su carrito y su historial."
         + (" La última señal salió de una tabla que casi nadie mira: los carruseles del sistema antiguo guardan qué enseñó, qué se clicó y qué se compró, comprador a comprador." if uses_history else "")},
 {"title": "Aprender del stream: reentrenar con nuestras propias respuestas",
  "figure": "2_mejoras.png",
  "bullets": ["Cada minuto: eventos nuevos a la memoria de cada comprador, tasas de cada producto y productos nuevos.",
              "Cada pocas horas: los pesos se reajustan con lo que de verdad se compró en nuestras listas.",
              "Cada versión nueva entra con parte del tráfico y compite con la campeona.",
              steps_txt],
  "say": "El histórico no se comporta igual que el vivo, así que sus pesos solo son el punto de partida. Para reentrenar reproducimos el día en orden, cada request con lo que se sabía en ese momento, para no hacer trampa. Lo que más nos ha dado es aprender de las listas que nosotros mismos enseñamos."
         + (f" El último paso parece pequeño porque solo cambia las listas de los compradores conocidos; con ellos la mejora es clara: +{es(fr['conocidos'])} puntos en casi cuatro horas de prueba, la más larga de la noche." if uses_history and fr else "")},
 {"title": "Decidir con datos: antes de medir, medir la regla",
  "figure": "3_productos_nuevos.png",
  "bullets": ["Comparamos versiones con la conversión esperada del leaderboard: mucho menos ruido que contar compras.",
              f"Medimos ese ruido: {es(ruido['total'], 1)} puntos por request. Dos versiones idénticas difieren 0,3 puntos con 600 requests.",
              "Regla: promocionar solo con 2 errores típicos y 3.000 requests.",
              "Nos corregimos: empezamos suponiendo un ruido de 4 y varias \"victorias\" no lo eran."],
  "say": "A las cinco de la mañana medimos bien el ruido, con 210 comparaciones entre versiones que corrieron a la vez, y salió un 50 % mayor del que usábamos. Releímos la noche entera: las mejoras grandes siguen en pie y otras decisiones, como cuántos productos nuevos enseñar, quedaron en \"no se puede saber con estos datos\". Preferimos contarlo así. También perdieron en vivo ideas que ganaban con datos guardados: poner todos los carritos primero, empujar las rebajas, un modelo de árboles."},
 {"title": "Lo que aprendimos de esta tienda",
  "figure": "4_estrellas.png",
  "bullets": [f"Los 50 productos más comprados se llevan el {es(d['estrellas']['top50'], 0)} % de las compras del carrusel; los 200, el {es(d['estrellas']['top200'], 0)} %.",
              "No son los que más se venden, sino los que más convencen: carritos por visita.",
              "Con compradores conocidos, 7 de cada 10 compras ya estaban en su carrito.",
              "El histórico engaña: allí el 64 % de las compras del carrusel eran recompras; en vivo no se recompra."],
  "say": "Esto cambió el modelo: dejamos de medir popularidad por volumen y pasamos a medirla por lo que convence un producto. Lo de las recompras lo vimos gracias a la columna propensity: el sistema antiguo elegía solo por popularidad, sin mirar al comprador, así que sus carruseles son un experimento limpio. También comprobamos lo que no hay: ni modas por horas, ni efecto de las rebajas, ni un nivel de precio propio de cada comprador. Y que toda compra pasa por un clic."},
 {"title": "Cómo usamos RawTree",
  "bullets": ["Requests: dos lectores en paralelo cada 0,2 s; respuestas por lotes en hilos aparte.",
              "Estado en SQL, sin ETL: stock cada 2 s; catálogo, popularidad y tasas por producto cada minuto.",
              "Memoria de compradores y parecido entre productos: agregaciones y un self-join sobre 12 millones de eventos.",
              "Historial de los 2 millones de carruseles antiguos: 10 millones de parejas comprador-producto agregadas en SQL.",
              "Entrenar y medir: feedback unido a requests y eventos en orden temporal; conversión esperada por versión."],
  "say": "Todo el cálculo pesado lo hace RawTree con SQL; en el portátil solo queda multiplicar señales por pesos. Aprendimos a no lanzar descargas grandes mientras producción corre: las bajamos en trozos pequeños, de uno en uno."},
 {"title": "Resultado: dónde estamos y por qué",
  "figure": "7_segmentos.png" if gap_txt and (OUT / "7_segmentos.png").exists() else "1_progreso.png",
  "bullets": [f"De {es(prog[7], 0)} % de la mejora posible a las 18:00 a {es(best_hour, 0)} % de madrugada."]
             + ([gap_txt, ref_txt] if gap_txt else [])
             + [f"Versión final: {champ}."],
  "say": "Cada décima que tenemos está medida frente a la versión anterior, con su margen, y sabemos de dónde viene. También sabemos lo que no sabemos: probamos una docena de señales más esta madrugada y ninguna movió la aguja, así que lo que les falta a nuestras listas no está en las señales que tenemos."},
 {"title": "Lo que salió mal y lo que haríamos distinto",
  "figure": "1_progreso.png",
  "bullets": ["Tres paradas por desplegar a mano y tres cortes de red: de ahí salieron el despliegue con vuelta atrás y la alarma de conexión.",
              "Medimos tarde el ruido de nuestra propia regla: varias decisiones de la noche se tomaron con menos certeza de la que creíamos.",
              "Con más tiempo: evaluar versiones sin gastar tráfico con la columna propensity, y un modelo en dos pasos (clic y compra)."],
  "say": "Los dos bajones de la curva son las paradas del sábado por la tarde y los cortes de red de medianoche. Si empezáramos hoy, lo primero sería la herramienta de medida, y lo segundo aprovechar que el sistema antiguo sorteaba por popularidad para probar ideas con el histórico antes de gastar tráfico real."},
]

md = ["# Guion de la presentación al jurado (team_23)", "",
      f"_Generado con los números actuales. {len(slides)} diapositivas, unos 6 minutos. En **negrita** lo que hay en pantalla; en cursiva, lo que se dice._", ""]
html = ["<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>team_23 · Presentación</title><style>",
        ":root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6}",
        "*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font-family:system-ui,-apple-system,'Segoe UI',sans-serif}",
        ".slide{background:var(--surface);border:1px solid var(--line);border-radius:10px;max-width:1120px;margin:24px auto;padding:44px 56px;min-height:630px;display:flex;flex-direction:column;page-break-after:always}",
        ".n{color:var(--muted);font-size:14px;margin-bottom:10px}h1{font-size:34px;line-height:1.15;margin:0 0 18px;font-weight:650;text-wrap:balance}",
        ".body{display:flex;gap:36px;flex:1;align-items:center}.body ul{flex:1;margin:0;padding-left:22px;font-size:22px;line-height:1.4}.body li{margin:0 0 16px}.body.fig ul{font-size:19px}",
        ".body li::marker{color:var(--accent)}.body img{flex:1.25;min-width:0;max-width:58%;height:auto;border-radius:6px}",
        ".say{margin-top:14px;padding-top:12px;border-top:1px solid var(--line);color:var(--ink2);font-size:15px;line-height:1.45}",
        "@media (max-width:820px){.slide{aspect-ratio:auto;padding:24px 16px;margin:12px}.body{flex-direction:column}.body img{max-width:100%}h1{font-size:25px}.body ul{font-size:17px}}",
        "@media print{body{background:#fff}.slide{border:0;margin:0;max-width:none;min-height:96vh}.say{display:none}}",
        "</style></head><body>"]
for i, s in enumerate(slides, 1):
    md += [f"## {i}. {s['title']}", ""] + ([f"![]({s['figure']})", ""] if s.get("figure") else []) + [f"- **{b}**" for b in s["bullets"]] + ["", f"_{s['say']}_", ""]
    img = f"<img src='{s['figure']}' alt='{s['title']}'>" if s.get("figure") else ""
    html.append(f"<section class='slide'><div class='n'>{i} / {len(slides)}</div><h1>{s['title']}</h1><div class='body{' fig' if s.get('figure') else ''}'><ul>"
                + "".join(f"<li>{b}</li>" for b in s["bullets"]) + f"</ul>{img}</div><div class='say'><b>Qué decir:</b> {s['say']}</div></section>")
html.append("</body></html>")
(OUT / "guion.md").write_text("\n".join(md))
(OUT / "presentacion.html").write_text("\n".join(html))
print(f"guion.md y presentacion.html escritos: {len(slides)} diapositivas | campeona: {champ} | mejor hora: {es(best_hour, 0)} %")
