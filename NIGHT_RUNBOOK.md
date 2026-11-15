# Guardia nocturna v6 · sábado 14 → domingo 15 de noviembre

Mandato de tarre (sábado 20:15): *"Quiero que sigas probando modelos durante la noche. Te doy libertad a
tocar todos los modelos siempre y cuando dejes constancia de lo que estás tocando. Y me hagas un resumen
bastante resumido de cuál será el mejor modelo y qué tengo que presentar mañana a las 10."*

**Objetivo: llegar a las 09:15 del domingo con (1) producción sin una sola parada más, (2) la mejor
versión posible al 100 % del tráfico y (3) un resumen muy corto para tarre en `logs/night_summary.md`.**
Solo puntúa el domingo de 10:00 a 12:00; esta noche el tráfico se usa para aprender.

Carpetas: trabajo en `~/Desktop/Q5/Datathon/rawtree` (rama `tarre`); producción en
`~/Desktop/Q5/Datathon/rawtree-prod` (nunca se edita a mano). Python: `/Applications/anaconda3/bin/python3`.
Diario: `logs/night_report.md`. Estado: `logs/night_state.json` (campeona, retadoras, inicio de la ronda).

## Qué salió mal el sábado (para no repetirlo)

1. **Tres paradas de producción** (17:56, 19:19 y 19:46). Las tres por parar el lanzador a mano antes de
   desplegar. Corregido: `deploy.sh` arranca el lanzador si no lo hay y producción corre en segundo plano.
2. **Versiones elegidas con pruebas offline que en vivo perdieron** (v5). Las pruebas con datos guardados
   sirven para descartar ideas y fijar pesos, pero **solo decide la conversión esperada del leaderboard**.
3. **Decisiones con muy pocos datos.** El leaderboard tiene un ruido de ~4 puntos por request: con 600
   requests el margen es ±0,16 puntos y con 5.000, ±0,06. Dos versiones idénticas (v6 y v10 sin rebajas)
   llegaron a marcar 1,51 % y 1,79 %. Diferencias de 0,2-0,3 con ~1.000 requests no significan nada.

## Reglas duras

1. **Producción solo se toca con `./deploy.sh`** desde `rawtree-prod`, y antes deben pasar
   `python -m unittest discover -s tests` y `python recommender.py --preflight`. Nunca `kill`, `pkill` ni
   Ctrl+C. Si no hay recomendador corriendo: `./deploy.sh` (lo arranca solo).
2. **Reparto.** La campeona con un tercio del tráfico o más; hasta 4 retadoras serias (nada de sorteos).
3. **Tras cada despliegue, comprobar a los 2-3 minutos** con `night_check.py --quiet`: responden las
   variantes esperadas, 0 duplicadas, sin `recommend falló` en el log. Si algo falla:
   `./deploy.sh --rollback`.
4. **Las decisiones se toman con `python monitor.py lbs HH:MM`** (conversión esperada desde el inicio de la
   ronda, con margen de error y z frente a la campeona):
   - **Promocionar** una retadora: z ≥ +2 frente a la campeona con 3.000 requests o más.
   - **Retirar** una retadora: z ≤ -2, o z ≤ -1 con más de 8.000 requests (no va a ganar).
   - Con |z| < 2 no hay veredicto: se deja correr o se sustituye por una idea mejor, apuntándolo.
   - `v10-sale` fuera de las ventanas de rebajas es el gemelo de v6: su z es el ruido real del momento.
5. **Trabajo pesado de uno en uno**, con `nice -n 19` y `OMP_NUM_THREADS=1`, nunca a la vez que un
   despliegue. El Mac tiene 8 GB y va justo.
6. **Horario.** Rondas de ~3-4 horas. Experimentos hasta las 08:30. A las 08:30 se decide la campeona final.
   **Antes de las 09:15, campeona al 100 % y verificada. Después, ningún cambio**, salvo `--rollback` si
   falla la salud. De 10:00 a 12:00 solo se vigila.
7. **Nada de descargas grandes en paralelo contra RawTree con producción en marcha** (`train.py`, consultas de
   millones de filas con varias conexiones): a las 23:51 una así coincidió con 555 respuestas perdidas. Los
   análisis del histórico usan la caché local (`cache/exp_*.pkl`). Desde las 09:15 no se lanza nada contra
   RawTree salvo el recomendador y `night_check.py`.
8. **Todo se apunta** en `logs/night_report.md` (qué se tocó, con qué evidencia, qué pasó) y cada cambio
   de modelo va en un commit con su explicación, subido a `origin/tarre` y verificado con `git ls-remote`.

## En cada despertar

1. `cd ~/Desktop/Q5/Datathon/rawtree && /Applications/anaconda3/bin/python3 night_check.py`
2. Si la salud falla, arreglarlo primero (reglas 1 y 3).
3. Leer la comparación de la ronda (la imprime `night_check.py`) y aplicar la regla 4.
4. Avanzar la cola de ideas si hay hueco y no son más de las 08:30.
5. Responder en dos o tres líneas: salud, cómo van las variantes y qué se ha hecho.

## Estado de la noche (actualizar al cambiar algo)

| Tramo | Reparto | Commit |
|---|---|---|
| 20:05-20:35 | v6 60 · v8 20 · v10 20 | 35ffd24 |
| 20:35-21:05 | v6 32 · v8 17 · v12 17 · v10 17 · v9 17 (sin internet de 20:46 a 20:56) | c60d19f |
| 21:05-21:20 | v8 32 (campeona, decisión de tarre) · v12 19 · v13 19 · v9 18 · v6 12; bucle con hilos | fce3bb0 |
| 21:20-21:55 | v8 28 · v12 20 · v13 20 · v14 20 · v6 12 | 86b9f96 |
| 22:00-22:28 | v8 30 (control) · v14 19 · v17 19 · v18 19 · v12 13 | b41e021 |
| 22:30-23:19 | v14 30 (CAMPEONA) · v17 25 · v13 25 · v8 10 · v18 10 | 36b8a6b |
| 23:20-00:23 | v14 25 · v13 20 · v20 20 · v21 20 · v17 15 (sin red 23:51-23:58 y 00:07-00:14) | 9fe5af5 |
| 00:25-01:12 | v14 30 · v22-cart-top 25 · v23-pop-log 25 · v20 20 (desde las 00:32 con alarma de conexión) | ad27e8e, 2d22000 |
| 01:15-01:38 | v23 35 (CAMPEONA) · v24-rate 35 · v20 20 · v14 10 | 4cc2f7f |
| 01:40-02:13 | v23 30 · v24-rate 30 · v25-rate-nonew 30 · v20 10 | 1c8a377 |
| 02:15-02:33 | v24 30 (CAMPEONA) · v26-new2x 25 · v27-new3x 25 · v25 10 · v23 10 | 71af842 |
| 02:35- | v24 25 · v26 25 · v27 25 · v28-rate-learned 25 (curva de productos nuevos: 7 % · 13 % · 21 % · 32 %) | f9a0f69 |

Qué es cada versión:
- **v8** (campeona nominal por decisión de tarre, y control): v6 reentrenado + señal de producto nuevo. Empatada con v6.
- **v12-cf**: v8 + parecido entre productos por carritos y compras ("quien compró X también Y"). +0,20 (z = +1,9).
- **v14-cf-split**: v12 + parecido con lo visto en la sesión y otros días + pesos distintos para compradores
  nuevos y conocidos + parecido estimado para productos nuevos. +0,30 (z = +2,0) en su primera media hora. **Es la
  mejor candidata a campeona.**
- **v17-ease**: v14 con la tabla de parecidos EASE (`cf_ease.npz`).
- **v18-live**: v14 + lo que el comprador ha visto o metido en el carrito durante el evento.
- Retiradas con datos: v13 (+0,18; v14 la incluye), v6 (= v8), v9 (dos veces z ≈ -2), v10, v5, v4, v7.

Las decisiones se leen con `python monitor.py ab` (usa `periods` y `champion` de `logs/night_state.json`).
**Tras cada despliegue hay que añadir a `periods` la hora de inicio del tramo nuevo** (redondeada a 5 minutos
hacia arriba) y, si cambia la campeona, `champion`.

## Receta para reentrenar una versión (≈ 10 minutos, de uno en uno)

```bash
cd ~/Desktop/Q5/Datathon/rawtree; PY=/Applications/anaconda3/bin/python3; D=cart_no_buy,viewed_no_cart,organic,on_sale
OMP_NUM_THREADS=1 nice -n 19 $PY -u analysis/live_matrix.py --cf-fill          # reproduce el día: tabla de señales (4-6 min)
OMP_NUM_THREADS=1 nice -n 19 $PY -u train_live.py --matrix --cf-fill --segment known --out vNN --drop $D --cap is_new_prod=1.0
OMP_NUM_THREADS=1 nice -n 19 $PY -u train_live.py --matrix --cf-fill --segment new --no-base --out vNNn --drop $D --cap is_new_prod=1.0
$PY analysis/calibrate_new.py vNN                  # productos nuevos: misma exposición que v8 (~6 % de los huecos)
$PY analysis/compare_lists.py v8 v14 vNN           # ¿enseña algo raro? (nuevos, popularidad, ms por request)
# después: VARIANTS en recommender.py, tests, --preflight, commit, push, git ls-remote, ./deploy.sh, night_state.json, diario
```

Lo aprendido esta noche sobre las pruebas offline: con ~150-200 compras de prueba no distinguen modelos
parecidos (árboles, logit condicional, tabla de parecidos con datos de hoy: todo dentro del ruido). Sirven
para detectar disparates, no para elegir. **Elige el leaderboard.**

Otros datos útiles:
- El leaderboard trae un campo `progress` por equipo y hora = 100 × (conversión − baseline) / (óptimo − baseline):
  qué parte de la mejora máxima posible captura cada equipo. El líder ronda el 60 %, la referencia el 53 % y
  nosotros el 38-41 % cuando no hay paradas. Solo depende de la conversión total (las respuestas perdidas
  cuentan como baseline). Cold start y fiabilidad se puntúan aparte.
- Rebajas: empiezan a las 10:30, 13:30, 16:30, 19:30... UTC y duran 2 h. La señal "rebajado" no aporta (peso
  aprendido -0,15).
- Productos nuevos: 100 a las 10:00, 14:00 y 18:00 UTC del sábado (¿cada 4 h?). Si sigue así, entran 100 a las
  11:00 hora local del domingo, en plena final. En los eventos orgánicos se venden como los antiguos.

## Cola de ideas (en este orden)

| # | Idea | Estado |
|---|---|---|
| A | Ronda 10: curva de exposición a productos nuevos con los pesos de v24: v24 7 % de los huecos · v26 13 % · v27 21 % · v28 32 % (peso aprendido sin recortar). Sin empujón (v25) pierde 0,2-0,4. Promocionar el mejor nivel con z ≥ +2 y 3.000 requests; retirar con z ≤ -2. Si v28 gana a v27, el techo está aún más arriba: probar factor mayor | En marcha |
| B | Reentreno final con la receta de v24 (`--drop cart_no_buy,viewed_no_cart,organic,on_sale,cart_live,view_live,pop_log,rate_buy`), y `is_new_prod` calibrado con `analysis/calibrate_new.py vNN --factor F` al nivel de exposición que gane la curva (F = 1 es v24, 2 es v26, 3.5 es v27) | Pendiente |
| C | Último reentreno hacia las 05:00 con la receta ganadora (`analysis/live_matrix.py --cf-fill` + `train_live.py --segment known/new`), como retadora con nombre nuevo. Reentrenar SÍ ayuda cuando hay feedback nuevo de las versiones buenas (v20: +0,20; z = +2,2) | Pendiente |
| D | Ronda final (desde ~06:30): las 2 mejores al 50 % cada una | Pendiente |
| E | Ya contestadas: pesos separados o un solo juego dan lo mismo (v13 = v14); EASE da lo mismo que el coseno en vivo (v17 = v14); las señales del evento no ayudan (v18); poner el mejor carrito primero no se nota (v22); cruces carrito × señales, árboles, logit condicional: sin mejora offline | Cerradas |
| F | Descartadas esta noche tras probarlas offline: modelo de árboles, logit condicional, base del histórico sin recompras (sale igual), tabla de parecidos con eventos de hoy (sin diferencia medible), subir el peso de productos nuevos, empujar rebajas | Cerradas |

## Último despertar antes de la final (08:30-09:15)

1. Decidir la campeona final con la regla 4. En caso de duda, la que más requests lleve validada.
2. Dejarla al 100 % (`VARIANTS = {"<campeona>": 100}`), tests, preflight, commit, push verificado con
   `git ls-remote`, `./deploy.sh`, y comprobar a los 3 minutos.
3. Escribir `logs/night_summary.md`, muy corto: qué versión queda y por qué (una tabla), y qué debe hacer
   tarre a las 10:00: portátil enchufado y con la tapa abierta, cerrar las aplicaciones que no hagan falta,
   no ejecutar nada en la carpeta de producción y no pulsar Ctrl+C en ninguna terminal.
