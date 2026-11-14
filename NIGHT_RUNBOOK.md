# Guardia nocturna · sábado 14 → domingo 15 de noviembre

Instrucciones para el bucle automático que vigila producción durante la noche. Acordado con tarre el
sábado por la tarde. **Objetivo: llegar a las 8:30 con producción sana, un A/B de toda la noche y los
pesos reentrenados, para que el equipo elija la versión de la final.** La noche de práctica no puntúa:
ante la duda, no tocar.

Carpetas: trabajo en `~/Desktop/Q5/Datathon/rawtree` (rama `tarre`); producción en
`~/Desktop/Q5/Datathon/rawtree-prod` (no se edita). Python: `/Applications/anaconda3/bin/python3`.
Estado del bucle: `logs/night_state.json`. Informe acumulado: `logs/night_report.md`.

## En cada despertar

1. `cd ~/Desktop/Q5/Datathon/rawtree && /Applications/anaconda3/bin/python3 night_check.py`
   (añade salud + A/B a `logs/night_report.md`). Código de salida 2 = problema de salud.
2. Leer `logs/night_state.json` y actuar según la hora y las reglas de abajo.
3. Apuntar en `logs/night_report.md` cualquier acción tomada (qué, por qué, resultado).
4. Responder en una o dos líneas: salud, A/B y si se ha hecho algo.

## Si la salud falla

- **No hay recomendador ni lanzador corriendo:** lanzarlo desde `rawtree-prod` con `./run_forever.sh`
  en segundo plano, comprobar a los 60 s con `night_check.py --quiet` y apuntarlo.
- **Hay lanzador pero el % a tiempo < 97 %** (sin un despliegue en los últimos 15 min que lo explique):
  mirar las últimas líneas de `rawtree-prod/logs/production.log`. Si el fallo empezó tras un despliegue
  del bucle, `cd ~/Desktop/Q5/Datathon/rawtree-prod && ./deploy.sh --rollback`. Si no, solo apuntarlo.
- **Duplicadas > 5:** hay otro recomendador enviando. No se puede arreglar desde aquí: apuntarlo bien visible.
- Nunca usar `kill`/`pkill` a mano; solo `./deploy.sh`, `./deploy.sh --rollback` y `./run_forever.sh`.

## El único cambio permitido (como máximo UNO en toda la noche)

Solo entre la **01:00 y las 04:00**, solo si `changes_done` es 0 en el estado, y solo sobre la variante
retadora. **La variante `v4-live-weights` y su 50 % no se tocan nunca.**

1. Reentrenar con todo el feedback acumulado, sin las dos señales redundantes:
   `nice -n 15 /Applications/anaconda3/bin/python3 train_live.py --out v7 --drop cart_no_buy,viewed_no_cart`
   → `model_weights_v7.json`. Mirar su tabla de PRUEBA.
2. Decidir con el A/B de v6 contra v4 que da `night_check.py` (llevará 7 h o más):
   - **v6 gana con claridad** (z de conversión ≥ +2): no cambiar nada. Apuntar `changes_done = 1`
     con motivo "v6 gana, se mantiene".
   - **En cualquier otro caso** (v6 pierde o empata): sustituir v6 por v7, siempre que en la tabla de
     PRUEBA el AUC de compra de v7 no sea peor que el de "v4 en producción" y el fichero se haya guardado.
     Si v7 no cumple, no cambiar nada.
3. Para sustituir: en `recommender.py`, dejar `VARIANTS` como
   `{"v7-retrained": 50, "v4-live-weights": 50}` (v6 pasa a comentario con su resultado);
   `python -m unittest discover -s tests` y `python recommender.py --preflight` deben pasar;
   commit, `git push origin tarre`, **verificar** con `git ls-remote origin tarre` que coincide con el
   commit local; `cd ../rawtree-prod && ./deploy.sh`; comprobar con `night_check.py --quiet` a los 2-3 min
   que responden v7 y v4 y que la salud es OK. Apuntar `changes_done = 1`.
4. Si tras el cambio la salud falla en dos chequeos seguidos o el log muestra `recommend falló`:
   `./deploy.sh --rollback` y apuntarlo.

## Límites

- Ningún despliegue después de las **07:00**.
- No añadir variantes distintas de la descrita, no tocar `main`, no borrar nada, no usar `--force`.
- Tarreneitor no entra en producción esta noche.
- No desplegar la versión final: eso lo decide el equipo por la mañana.

## Último despertar (a partir de las 07:30)

Escribir `logs/night_summary.md`: cómo fue la salud durante la noche (caídas, huecos), la tabla final de
cada A/B con su margen de error (v6 contra v4, y v7 contra v4 si hubo cambio), qué se cambió y por qué, y
una recomendación razonada de qué versión llevar a la final. Recordar los pasos de la mañana:
reentrenar con todo el feedback, poner la elegida al 100 % y desplegar antes de las 9:30.
