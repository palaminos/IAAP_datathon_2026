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
total_gain = sum(x["ganancia"] for x in g)

slides = [
 {"title": "team_23 · Un recomendador que aprende mientras atiende",
  "bullets": [f"Capturamos el {es(best_hour, 0)} % de la mejora máxima posible (0 = los más vendidos, 100 = la lista perfecta).",
              "Respondemos en 1 segundo de mediana, con el límite en 3.",
              f"Tres mejoras seguidas demostradas en vivo durante la noche: +{es(total_gain)} puntos de conversión en total."],
  "say": "Somos cuatro y empezamos de cero el sábado por la mañana. Nuestra idea ha sido sencilla: un modelo que se pueda explicar, y que cada cambio se gane su sitio con datos en vivo."},
 {"title": "Cómo funciona: una request, 10 productos, 1 milisegundo de cálculo",
  "figure": "5_latencia.png",
  "bullets": ["Memoria de cada comprador: visto, carrito y comprado (60 días + hoy).",
              "En cada request se puntúan los ~5.400 productos: señales × pesos.",
              f"Lectura, cálculo y envío en hilos separados: de {es(lat['antes']['p50'], 1)} a {es(lat['despues']['p50'], 1)} s de mediana.",
              "Cada despliegue pasa una prueba con requests reales y tiene vuelta atrás."],
  "say": "La fiabilidad va primero: si no respondes a tiempo, da igual lo bueno que sea el modelo. Lo ya comprado y lo agotado se quitan de la lista. Tuvimos tres paradas el sábado por desplegar a mano; de ahí salió el despliegue con comprobación previa y vuelta atrás con un comando."},
 {"title": "El modelo, en cuatro ideas que se pueden leer",
  "bullets": ["Lo que el comprador ya hizo: lo que tiene en el carrito sube; lo que ya compró, fuera (en vivo no se recompra).",
              "Lo que acaba de ver en esta sesión.",
              "Parecido entre productos: \"quien metió X en el carrito o lo compró, también Y\".",
              "Lo bueno que es el producto: carritos por cada visita, no ventas totales.",
              "Pesos distintos para compradores nuevos y conocidos: a los nuevos les mueve lo que convence a todos; a los conocidos, su carrito y su historial."],
  "say": "Es una suma de señales con pesos. Lo elegimos así a propósito: se entrena en segundos, responde en un milisegundo y cuando algo va mal podemos mirar qué peso lo causa."},
 {"title": "Aprender del stream: reentrenar con nuestras propias respuestas",
  "figure": "2_mejoras.png",
  "bullets": ["Cada pocas horas se reajustan los pesos con lo que de verdad se compró.",
              "Cada versión nueva entra con parte del tráfico y compite con la campeona.",
              f"Tres pasos seguidos: +{es(g[0]['ganancia'])}, +{es(g[1]['ganancia'])} y +{es(g[2]['ganancia'])} puntos de conversión."],
  "say": "El histórico no se comporta igual que el vivo, así que sus pesos solo son el punto de partida. Para reentrenar reproducimos el día en orden, cada request con lo que se sabía en ese momento, para no hacer trampa. Lo que más nos ha dado es aprender de las listas que nosotros mismos enseñamos."},
 {"title": "Decidir con datos: lo que parecía buena idea y no lo era",
  "figure": "3_productos_nuevos.png",
  "bullets": ["Dos versiones idénticas pueden diferir 0,3 puntos con 600 requests.",
              "Regla: no decidir con menos de 3.000 requests ni de 2 errores típicos.",
              "Productos nuevos: ni sin empujón ni con más. Cinco niveles a la vez.",
              "Más de 25 versiones en un día; cada retirada, con su número."],
  "say": "Nos equivocamos varias veces, y lo interesante es cómo lo supimos: por la conversión esperada del leaderboard por versión, que tiene mucho menos ruido que contar compras. Perdieron en vivo ideas que ganaban en pruebas con datos guardados: poner todos los carritos primero, empujar las rebajas, un modelo de árboles. Llegamos a quitar el empujón a los productos nuevos por un análisis con pocas compras; la prueba en vivo dijo lo contrario en media hora."},
 {"title": "Lo que aprendimos de esta tienda",
  "figure": "4_estrellas.png",
  "bullets": [f"Los 50 productos más comprados se llevan el {es(d['estrellas']['top50'], 0)} % de las compras del carrusel; los 200, el {es(d['estrellas']['top200'], 0)} %.",
              "No son los que más se venden, sino los que más convencen: carritos por visita.",
              "Con compradores conocidos, 7 de cada 10 compras ya estaban en su carrito."],
  "say": "Esto cambió el modelo: dejamos de medir popularidad por volumen y pasamos a medirla por lo que convence un producto. Fue la última gran mejora de la noche. También comprobamos lo que no hay: ni modas por horas ni efecto de las rebajas en el carrusel."},
 {"title": "Cómo usamos RawTree",
  "bullets": ["Requests: dos lectores en paralelo cada 0,2 s; respuestas por lotes en hilos aparte.",
              "Estado en SQL, sin ETL: stock cada 2 s; catálogo, popularidad y tasas por producto cada minuto (una fila por producto).",
              "Memoria de compradores y tabla de parecidos entre productos con agregaciones y un self-join sobre 12 millones de eventos.",
              "Entrenamiento: feedback unido a requests y a eventos, reproducido en orden temporal.",
              "Medida: la conversión esperada por versión del leaderboard, por tramos entre despliegues."],
  "say": "Todo el cálculo pesado lo hace RawTree con SQL; en el portátil solo queda multiplicar señales por pesos. Aprendimos a no lanzar descargas grandes mientras producción corre."},
 {"title": "Resultado, y lo que haríamos distinto",
  "figure": "1_progreso.png",
  "bullets": [f"De {es(prog[7], 0)} % de la mejora posible a las 18:00 a {es(best_hour, 0)} % de madrugada.",
              "Lo que salió mal: paradas por desplegar a mano y cortes de red. De ahí salieron el despliegue con vuelta atrás y la alarma de conexión.",
              "Con más tiempo: compradores conocidos, y precio según dispositivo y región.",
              f"Versión final: {champ}."],
  "say": "Cada décima que tenemos está medida frente a la versión anterior y sabemos de dónde viene. Eso es lo que nos llevamos."},
]

md = ["# Guion de la presentación al jurado (team_23)", "",
      "_Generado con los números actuales. 8 diapositivas, unos 5 minutos. En **negrita** lo que hay en pantalla; en cursiva, lo que se dice._", ""]
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
