# Guardia nocturna v3 · sábado 14 → domingo 15 de noviembre

Mandato de tarre (sábado 20:00): *"Te dejo la noche para que lo hagas bien. Arréglame los errores. Quiero
volver al top 3. Quiero ganar esto."* A partir de ahora el bucle opera producción de forma autónoma, con
las reglas de este documento. Sustituye a las versiones anteriores.

**Objetivo: llegar a las 09:15 del domingo con (1) producción sin una sola parada más, (2) la mejor
versión posible al 100 % del tráfico y (3) un resumen claro para tarre.** Solo puntúa el domingo de 10:00
a 12:00.

Carpetas: trabajo en `~/Desktop/Q5/Datathon/rawtree` (rama `tarre`); producción en
`~/Desktop/Q5/Datathon/rawtree-prod`. Python: `/Applications/anaconda3/bin/python3`.
Diario: `logs/night_report.md`. Estado: `logs/night_state.json`.

## Qué salió mal el sábado (para no repetirlo)

1. **Tres paradas de producción** (17:56, 19:19 y 19:46: 2,5 + 1,5 + 15 minutos). Las tres por parar el
   lanzador a mano antes de desplegar. Ya corregido: `deploy.sh` arranca el lanzador si no lo hay, y
   producción corre en segundo plano, sin depender de ninguna terminal.
2. **Versiones recomendadas por pruebas offline que en vivo perdieron**: v5 (-29 %) y v9 (-18 %). Las
   pruebas con datos guardados NO deciden; solo decide la conversión esperada del leaderboard.
3. **Demasiadas variantes a la vez** (cinco, con dos sorteos): bajaban la conversión del equipo.

## Reglas duras

1. **Producción solo se toca con `./deploy.sh`** desde `rawtree-prod`, y antes deben pasar
   `python -m unittest discover -s tests` y `python recommender.py --preflight`. Nunca `kill`, `pkill` ni
   Ctrl+C. Si no hay recomendador corriendo: `./deploy.sh` (lo arranca solo).
2. **La campeona siempre tiene al menos el 60 % del tráfico.** Como mucho 2 retadoras, con un 20 % cada
   una como máximo. Sin variantes de sorteo. Campeona actual: `v6-similar-users`.
3. **Tras cada despliegue, comprobar a los 2-3 minutos** con `night_check.py --quiet`: responden las
   variantes esperadas, 0 duplicadas, sin `recommend falló` en el log. Si algo falla:
   `./deploy.sh --rollback`.
4. **Las decisiones se toman solo con `python monitor.py lb <minutos>`** (conversión esperada, la métrica
   que puntúa), comparando campeona y retadora en las MISMAS ventanas:
   - Se **promociona** una retadora si lleva al menos 5.000 requests, supera a la campeona en 0,08 puntos
     o más, y va por delante en al menos 2 de las 3 últimas horas.
   - Se **retira** si con 3.000 requests o más va 0,10 puntos o más por debajo.
   - El ruido se calibra con el par gemelo `v10-sale` / `v6` fuera de las ventanas de rebajas (dan
     exactamente las mismas listas): la diferencia que marquen es puro ruido.
5. **Al promocionar**, la nueva campeona pasa al 60 % y la anterior se queda una ronda al 20 % como control.
6. **Trabajo pesado (entrenar, reproducir el día) de uno en uno**, con `nice -n 19` y
   `OMP_NUM_THREADS=1`, nunca a la vez que un despliegue. El Mac tiene 8 GB y va justo.
7. **Horario.** Experimentos hasta las 08:30. A las 08:30 se decide la campeona final. **Antes de las
   09:15, campeona al 100 % y verificada. Después, ningún cambio**, salvo `--rollback` si falla la salud.
   De 10:00 a 12:00 solo se vigila.
8. **Todo se apunta** en `logs/night_report.md`: qué se hizo, con qué evidencia y qué pasó.

## En cada despertar

1. `cd ~/Desktop/Q5/Datathon/rawtree && /Applications/anaconda3/bin/python3 night_check.py`
2. Si la salud falla, arreglarlo primero (reglas 1 y 3).
3. Mirar el leaderboard de las variantes activas (`monitor.py lb 60` y `lb 180`) y aplicar la regla 4.
4. Avanzar la cola de hipótesis si hay hueco y no son más de las 08:30.
5. Responder en dos o tres líneas: salud, cómo van las variantes y qué se ha hecho.

## Cola de hipótesis (en este orden)

| # | Hipótesis | Estado |
|---|---|---|
| A | `v8-new-products`: v6 reentrenado con más feedback + señal de producto nuevo (tope 0,4) | En marcha al 20 % desde las 20:05 |
| B | `v10-sale`: v6 + empujón a los productos rebajados. Solo se juzga DENTRO de las ventanas de rebajas (previstas 20:30-22:30, 23:30-01:30, 02:30-04:30, 05:30-07:30). Fuera de ellas es el gemelo de v6 para medir ruido | En marcha al 20 % desde las 20:05 |
| C | `v11-ctx-price`: v6 + sensibilidad al precio según dispositivo, región y página. En el histórico del carrusel (20 M de impresiones), en Android lo caro se compra un 0,32 de lo barato, frente a 0,77 en iOS y web; por regiones va de 0,37 (Andalucía) a 0,88 (Galicia). Confirmado en los eventos orgánicos de hoy (Android 0,43 frente a 0,70). Nuestro modelo no usa ni dispositivo ni región | Por construir. Entra como retadora en cuanto A o B tengan veredicto, o sustituyendo a v8 a las 22:30 si sigue en empate |
| D | Latencia: el p99 ronda 2,5-2,9 s con el límite en 3 s. Bajar la espera del bucle de 0,3 a 0,15 s | Desplegar junto con el siguiente cambio y vigilar el p99 |
| E | Reentrenar la campeona con el feedback de toda la noche (hacia las 05:30) y probarla como retadora hasta las 08:30. Solo se queda si cumple la regla 4 | Pendiente |
| F | Combinar lo que gane de A, B y C en una sola versión y validarla como retadora antes de las 08:30 | Pendiente |

## Último despertar antes de la final (08:30-09:15)

1. Decidir la campeona final con la regla 4. En caso de duda, la que más horas lleve validada.
2. Dejarla al 100 % (`VARIANTS = {"<campeona>": 100}`), tests, preflight, commit, push verificado con
   `git ls-remote`, `./deploy.sh`, y comprobar a los 3 minutos.
3. Escribir `logs/night_summary.md`: salud de la noche, tabla de cada comparación con sus requests, qué
   se promocionó o retiró y por qué, qué versión queda para la final, y qué debe hacer tarre:
   portátil enchufado y con la tapa abierta, cerrar las aplicaciones que no hagan falta, no ejecutar
   nada en la carpeta de producción y no pulsar Ctrl+C en ninguna terminal.
