# team_23 · Recomendador en tiempo real (RawTree Datathon FME 2026)

Para cada visitante de la tienda elegimos, en menos de 3 segundos, los 10 productos que más probablemente
comprará. Miramos qué tiene en el carrito, qué acaba de ver, qué compraron otros compradores parecidos y lo que
hizo en el sistema antiguo; y cada cambio del modelo lo probamos en vivo con una parte del tráfico antes de
dárselo a todos. El enunciado del reto está en [`docs/RETO.md`](docs/RETO.md).

Versión de la final: **v33-user-history**. Presentación al jurado: [`presentacion/presentacion.pdf`](presentacion/presentacion.pdf)
(guion con lo que se dice en cada diapositiva: [`presentacion/guion.md`](presentacion/guion.md)).

## Cómo funciona

```
shop.requests ──► 2 lectores (cada 0,2 s) ──► cálculo: señales × pesos para los ~5.500 productos ──► 2 emisores ──► team_23.submissions
                                                   ▲
   RawTree (SQL, cada minuto): stock, catálogo, popularidad, carritos por visita, eventos nuevos de cada comprador
   team_23.feedback ──► reentrenamiento de los pesos con lo que de verdad se compró en nuestras listas
```

- **Modelo** (`model.py`): regresión logística sobre ~30 señales por producto; se enseñan los 10 de mayor
  puntuación. Reglas fijas: nada ya comprado (en vivo no se recompra) y nada agotado. Pesos distintos para
  compradores nuevos y conocidos.
- **Señales principales**: carrito y vistas del comprador (60 días de historial + el evento), lo visto en la
  sesión, parecido entre productos por carritos y compras (`train_cf.py`), gustos de compradores parecidos
  (factorización de matrices, `train_mf.py`), carritos por visita de cada producto, y lo que cada comprador
  hizo en los carruseles del sistema antiguo (`shop.history_sessions`).
- **Productos y compradores nuevos (cold start)**: un producto nuevo recibe el parecido medio de su
  subcategoría y un empujón calibrado; los compradores nuevos tienen sus propios pesos.
- **Producción** (`recommender.py`): lectura, cálculo y envío en hilos separados (mediana ~1 s, límite 3 s).
  `deploy.sh` prueba cada versión con 100 requests reales antes de desplegarla y permite volver atrás.

## Cómo validamos

1. **Antes del evento, con el histórico** (`train.py`): todo lo que el modelo sabe se calcula con datos
   anteriores a una fecha de corte; se aprende con una semana y se mide en la siguiente.
2. **Con nuestro feedback** (`train_live.py`, `analysis/live_matrix.py`): se reproduce el día en orden, cada
   request con lo que se sabía en ese momento; validación cruzada en 5 bloques de tiempo para elegir señales.
3. **En vivo**: cada versión nueva compite con la campeona con parte del tráfico; decide la conversión esperada
   del leaderboard con su margen de error (ruido medido: 5,9 puntos por request, `analysis/ab_noise.py`).

## Versiones que marcan el camino

| Versión | Qué añade | Frente a la anterior (conversión esperada) |
|---|---|---|
| v8 | pesos corregidos con el feedback en vivo + productos nuevos | — |
| v14 | parecido entre productos (carritos y compras) y pesos para nuevos/conocidos | +0,31 ± 0,26 |
| v24 | lo que convence cada producto: carritos por visita | +0,47 ± 0,40 (dos pasos) |
| v33 | historial de cada comprador en el sistema antiguo | +0,05 ± 0,11 en total; +0,14 con conocidos |

Por el camino se probaron y descartaron con datos, entre otras: poner todos los carritos primero, empujar las
rebajas, un modelo de árboles, pesos por página y la `propensity` del sistema antiguo (es solo popularidad).

## Estructura

| Fichero | Qué hace |
|---|---|
| `recommender.py` | El sistema en producción (lectura, cálculo, envío, preflight) |
| `model.py` | Señales, memoria de compradores y puntuación |
| `train.py`, `train_live.py`, `train_cf.py`, `train_mf.py`, `train_organic.py` | Entrenamiento con el histórico, con el feedback y de las tablas de parecido |
| `analysis/` | Análisis que justificaron las decisiones clave |
| `monitor.py`, `night_check.py` | Medida en vivo (leaderboard por versión, salud) |
| `deploy.sh`, `run_forever.sh`, `set_final.sh`, `stop.sh` | Despliegue seguro y relanzado automático |
| `tests/` | Tests (`python -m unittest discover -s tests`) |
| `starter/` | Kit de inicio de los organizadores |

## Cómo ejecutarlo

```bash
cp .env.example .env          # TEAM_ID y RAWTREE_API_KEY
pip install -r starter/requirements.txt
python -m unittest discover -s tests
python recommender.py --preflight      # prueba con 100 requests reales sin enviar nada
./run_forever.sh                       # producción (se relanza sola si se para)
```

Equipo: ver [`EQUIPO.md`](EQUIPO.md).
