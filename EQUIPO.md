# IAAP · Datathon FME 2026 · Reto RawTree (team_23)

Equipo: **tarre · vidal · pau · iris**

Reglas completas del reto en [README.md](README.md) · Leaderboard: https://datathon.rawtree.dev/leaderboard

## Ficheros

| Fichero | Qué es |
|---|---|
| `recommender_v1.py` | **Baseline del equipo**: co-vistas + popularidad por subcategoría/categoría + tendencias de la última hora. Incluye test A/B (`VARIANTS`) |
| `monitor.py` | Diagnóstico (solo lee): `check`, `conv`, `latency`, `lag`, `all` |
| `starter/` | Kit oficial: baseline de los organizadores, `queries.sql`, notebook de evaluación offline |

## Arrancar

```bash
pip install -r starter/requirements.txt
cp .env.example .env            # TEAM_ID=team_23 y la clave (pedídsela al equipo; NUNCA se sube)
python monitor.py check         # ¿funciona la conexión?
./run_forever.sh                # producción: reinicia solo si se cae, logs en logs/production.log
```

Lanzadlo desde la app **Terminal** de macOS (no desde VS Code), con el Mac enchufado y la tapa abierta.

⚠️ **Solo un recomendador corriendo a la vez para todo el equipo.** Dos a la vez responden dos veces a la misma
request y solo cuenta una. Avisad en el grupo antes de parar o lanzar el que está en producción.

## Vigilar

```bash
python monitor.py conv 1        # conversión por versión del modelo, última hora
python monitor.py latency 10    # % a tiempo y motivos de fallback, últimos 10 min (objetivo >99 %)
python monitor.py lag           # cuánto tarda una request en aparecer
```

Referencia del sistema antiguo: **1,40 % conversión**, 3,75 % CTR, 0,49 €/request.

## Reparto del trabajo

Cuatro áreas, una persona responsable de cada una (todos pueden tocar de todo; el responsable es quien
se asegura de que esa parte sale). **Propuesta: cambiad los nombres como acordéis.**

| Área | Responsable | Qué hay que hacer | Puntos que afecta |
|---|---|---|---|
| **Infraestructura** | tarre (propuesta: producción corre en su Mac) | Que el sistema no falle: en qué máquina corre toda la noche, reinicio automático si se cae, latencia, vigilancia con `monitor.py` | Fiabilidad (10), uso de RawTree (5) |
| **Modelos** | _por asignar_ | Mejorar `recommend()`: compras juntas, historial de cada usuario, similitud por descripción para productos nuevos, tendencias | Conversión (50), cold start (10) |
| **Evaluación** | _por asignar_ | Evaluación offline con `starter/offline_eval.ipynb`: comparar cada idea contra el baseline antes de desplegarla; leer los tests A/B con `monitor.py conv` | Método, "validado offline" (10) |
| **EDA** | _por asignar_ | Explorar el histórico con preguntas concretas (abajo) y preparar los gráficos de la presentación | Aprendizaje del stream (10), comunicación (5) |

Preguntas para el EDA (corto y dirigido: cada hallazgo debe acabar en una señal del modelo o en una diapositiva):

- ¿En qué posición del carrusel se hace clic y se compra? (consulta 7 de `starter/queries.sql`)
- ¿Se compra el producto que se acaba de ver, u otro de la misma subcategoría?
- ¿Qué compran los usuarios nuevos (`is_new_user`)? ¿Y los que llegan sin `recent_views`?
- ¿Cambia lo que se compra según `page` (home/product/category), dispositivo, región o precio?
- ¿Cómo son los productos nuevos del evento (`new_product` en `shop.catalog_updates`) y a cuáles del histórico se parecen?

Dónde va cada cosa: notebooks de exploración en `notebooks/`, evaluación en `evaluation/`,
material de la presentación en `docs/`.

## Cómo trabajamos con git

- `main` = lo que está (o va a estar) en producción. Solo se cambia cuando una mejora está probada.
- Cada uno trabaja en su rama: `tarre`, `vidal`, `pau`, `iris`. Las áreas de arriba son reparto de trabajo, no ramas.

```bash
git checkout <tu-rama>
# ... cambios ...
git add -A && git commit -m "qué has cambiado" && git push
```

Para llevar una mejora a `main`: Pull Request en GitHub desde tu rama, y que otro miembro la revise.
Antes de proponerla:
1. Cambia `MODEL_VERSION` (o añade una variante en `VARIANTS` para un A/B).
2. Compárala offline (`starter/offline_eval.ipynb`) y/o en vivo con `python monitor.py conv`.

Para traer a tu rama lo último de `main`: `git pull origin main`.

## Fechas

- Práctica: sábado 10:00 → domingo 10:00 (no cuenta).
- **Final: domingo 10:00–12:00. La mejor versión tiene que estar corriendo antes de las 10:00.**
- Domingo tarde: presentación con demo (30 % de la nota).
