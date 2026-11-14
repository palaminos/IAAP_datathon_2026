# Guardia nocturna v4 · sábado 14 → domingo 15 de noviembre

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
7. **Todo se apunta** en `logs/night_report.md` (qué se tocó, con qué evidencia, qué pasó) y cada cambio
   de modelo va en un commit con su explicación, subido a `origin/tarre` y verificado con `git ls-remote`.

## En cada despertar

1. `cd ~/Desktop/Q5/Datathon/rawtree && /Applications/anaconda3/bin/python3 night_check.py`
2. Si la salud falla, arreglarlo primero (reglas 1 y 3).
3. Leer la comparación de la ronda (la imprime `night_check.py`) y aplicar la regla 4.
4. Avanzar la cola de ideas si hay hueco y no son más de las 08:30.
5. Responder en dos o tres líneas: salud, cómo van las variantes y qué se ha hecho.

## Rondas

| Ronda | Horario | Reparto | Estado |
|---|---|---|---|
| 1 | 20:40 → ~00:00 | v6 32 % (campeona) · v8 17 % · v12-cf 17 % · v10-sale 17 % · v9-organic 17 % | En marcha (commit c60d19f) |
| 2 | ~00:00 → ~04:00 | ganadora de la ronda 1 + nuevas candidatas (ver cola) | Pendiente |
| 3 | ~04:00 → 08:30 | las 2-3 finalistas con mucho tráfico cada una, para decidir con margen pequeño | Pendiente |

## Cola de ideas (en este orden)

| # | Idea | Estado |
|---|---|---|
| A | `v12-cf`: parecido entre productos por carritos y compras ("quien compró X también Y") aplicado a lo que el comprador ya metió en el carrito o compró. En el histórico, sin ese parecido casi no hay compras | En la ronda 1 |
| B | `v13`: el mismo parecido aplicado a lo que el comprador acaba de VER en la sesión y a lo que vio otros días. Es lo único que sabemos de un comprador nuevo; en el histórico multiplica la compra hasta por 30 | Construyéndose (señales `cf_sess`, `cf_view`) |
| C | Rebajas (`v10-sale`): solo se juzga dentro de las ventanas (20:30-22:30, 23:30-01:30, 02:30-04:30, 05:30-07:30, 08:30-10:30 y probablemente 11:30-13:30) | En la ronda 1 |
| D | Productos nuevos: v8 los enseña en el 5,5 % de los huecos. Probar más exposición (el peso aprendido pide mucha más) | Pendiente, ronda 2 |
| E | Latencia: quitar 0,2 s de espera del bucle (`POLL_S` 0,3 → 0,1) y enviar las respuestas en un hilo aparte | Con el siguiente despliegue |
| F | Reentrenar la finalista con el feedback de toda la noche y validarla como retadora en la ronda 3 | Pendiente |

## Último despertar antes de la final (08:30-09:15)

1. Decidir la campeona final con la regla 4. En caso de duda, la que más requests lleve validada.
2. Dejarla al 100 % (`VARIANTS = {"<campeona>": 100}`), tests, preflight, commit, push verificado con
   `git ls-remote`, `./deploy.sh`, y comprobar a los 3 minutos.
3. Escribir `logs/night_summary.md`, muy corto: qué versión queda y por qué (una tabla), y qué debe hacer
   tarre a las 10:00: portátil enchufado y con la tapa abierta, cerrar las aplicaciones que no hagan falta,
   no ejecutar nada en la carpeta de producción y no pulsar Ctrl+C en ninguna terminal.
