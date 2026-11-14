# Guardia nocturna · sábado 14 → domingo 15 de noviembre

Instrucciones para el bucle automático que vigila producción durante la noche.
**Objetivo: llegar a las 8:30 con producción sana y el máximo de datos útiles para reentrenar el modelo
de la final.** La noche de práctica no puntúa: ante la duda, no tocar.

Carpetas: trabajo en `~/Desktop/Q5/Datathon/rawtree` (rama `tarre`); producción en
`~/Desktop/Q5/Datathon/rawtree-prod` (no se edita). Python: `/Applications/anaconda3/bin/python3`.
Informe acumulado: `logs/night_report.md`.

## Plan de la noche (actualizado el sábado a las 18:45)

- Reparto previsto: `v6-similar-users` 50 % + `v7-explore` 50 %. La variante de exploración sortea 10
  productos entre los 40 mejores de v6 y guarda en `team_23.scratch` (`feature_set = 'explore_v1'`) las
  probabilidades con que salió cada uno. Convierte menos a propósito: sirve para aprender qué pasa con lo
  que el modelo normalmente no enseña. **No es para la final.**
- Lo despliega tarre (o el bucle solo si tarre lo pide expresamente) después de ver la prueba previa.
- **El bucle NO cambia el modelo ni el reparto esta noche.** Queda anulado el "único cambio permitido" de
  la versión anterior de este documento.

## En cada despertar

1. `cd ~/Desktop/Q5/Datathon/rawtree && /Applications/anaconda3/bin/python3 night_check.py`
   (añade salud + A/B a `logs/night_report.md`). Código de salida 2 = problema de salud.
2. Si hay un problema de salud, aplicar la sección siguiente. Si no, no hacer nada más.
3. Responder en una o dos líneas: salud, variantes activas y si se ha hecho algo.

No lanzar entrenamientos ni análisis pesados desde el bucle: el Mac tiene 8 GB y va justo de memoria; un
proceso grande puede frenar producción.

## Si la salud falla

- **No hay recomendador ni lanzador corriendo:** lanzarlo desde `rawtree-prod` con `./run_forever.sh`
  en segundo plano, comprobar a los 60 s con `night_check.py --quiet` y apuntarlo.
- **Hay lanzador pero el % a tiempo < 97 %** en dos chequeos seguidos: mirar las últimas líneas de
  `rawtree-prod/logs/production.log`. Si el fallo empezó tras un despliegue reciente,
  `cd ~/Desktop/Q5/Datathon/rawtree-prod && ./deploy.sh --rollback`. Si no, solo apuntarlo.
- **El log muestra `recommend falló` de forma repetida:** `./deploy.sh --rollback` y apuntarlo.
- **Duplicadas > 5:** hay otro recomendador enviando. No se puede arreglar desde aquí: apuntarlo bien visible.
- `registro de exploración fallido` en el log NO es un problema de salud: las respuestas no dependen de eso.
- Nunca usar `kill`/`pkill` a mano; solo `./deploy.sh --rollback` y `./run_forever.sh`.

## Límites

- Ningún despliegue de versiones nuevas por parte del bucle. Solo rollback si la salud falla.
- No tocar `main`, no borrar nada, no usar `--force`.

## Último despertar (a partir de las 07:30)

Escribir `logs/night_summary.md`: cómo fue la salud durante la noche (caídas, huecos), cuántas respuestas
y compras ha reunido cada variante, la conversión de v6 y la de la exploración con su margen de error, y
los pasos de la mañana:

1. Reentrenar con todo el feedback (incluye la exploración):
   `nice -n 15 /Applications/anaconda3/bin/python3 train_live.py --out v8 --drop cart_no_buy,viewed_no_cart`
   y mirar la tabla de PRUEBA, incluida la fila "solo exploración".
2. Dejar en `VARIANTS` solo la versión elegida al 100 % (**sin** `v7-explore`), pasar tests y prueba previa.
3. Desplegar con `./deploy.sh` antes de las 9:30 y no tocar nada más hasta las 12:00.
4. Antes de las 10:00: cerrar aplicaciones que no hagan falta (el Mac va justo de memoria), portátil
   enchufado, tapa abierta.
