# EDA 01 · Hallazgos (sáb 14-nov, 16:10)

Script: `notebooks/eda_01.py` (agrega todo en RawTree, ~30 s). Entre corchetes, las tablas de las que sale cada dato.

## Calidad
- Sin nulos críticos, sin duplicados, sin fuga temporal en `recent_views` (0 vistas posteriores a la request). Precios de eventos = catálogo. [history_sessions, history_events, catalog]
- `propensity`: mínimo 2e-6 y 164 valores fuera de (0,1] → IPS solo con clipping; varianza alta. [history_sessions]
- Sin deriva: conversión semanal estable (1,38-1,44 %); mezcla page/device/region/categoría histórico ≈ live. [history_sessions, requests, history_events, events]

## Target [history_sessions, 2 M]
- CTR 3,75 %, conversión 1,40 %, 0,49 €/request, 35 €/compra. Compra siempre = clic.
- Posición: conversión slot 1 = 0,45 % vs slot 10 = 0,05 % (9×). Confundido con popularidad (el sistema viejo ponía best sellers arriba): el efecto puro de posición es menor. Pendiente en EDA 02.
- page=home (20 %): sin recent_views, conversión 0,14 % (10× menos).

## Señales
- 100 % de las compras con vistas están en la **misma categoría que la última vista**; 33 % misma subcategoría, 23 % misma marca, 16 % un producto de recent_views, 8 % el último visto. [history_sessions + catalog]
- Historial del usuario, solo eventos **anteriores** a la sesión: 63 % de las compras del carrusel son productos que el usuario había visto antes (base: 25 % de lo mostrado); 47 % ya los había comprado. Recompra orgánica 11 % de pares usuario-producto. [history_sessions + history_events]
- Categoría top del usuario: 78 % de las compras (56 % en home). Ojo: calculado con todos los eventos, también posteriores; recalcular sin fuga en EDA 02. [history_sessions + history_events + catalog]
- Conversión orgánica en U por precio (<10 € y >250 € más altas). [history_events + catalog]
- Compras del carrusel concentradas en 564 productos (sesgo del sistema viejo); orgánicas cubren 4.809.

## Live
- 3 req/s. 30 % usuarios nuevos (sin historial); 33 % cold start. 200 productos nuevos (subcat y marca conocidas). [requests, catalog_updates]
- 1.895 cambios de precio, la mayoría −20 %. Efecto en conversión aún sin medir. [catalog_updates + catalog]
- Nuevos convierten 1,40 % vs conocidos 0,58 % con nuestros modelos (mezcla de versiones: orientativo, no causal). [team_23.feedback + requests]
- Versiones live: v3-no-rebuy 1,56 % (n=2,4k), v2-user-lr 1,12 % (n=9,6k), v1 0,72 %, baseline 0,73 %. [feedback + submissions]
- Fiabilidad: ~19 % fallback acumulado (7.208 missing a las 11h, 313 a las 14h). [feedback]

## Validación decidida
- Offline: split temporal últimos 7 días; métrica principal = compras del carrusel en top-10 ponderadas por posición; secundaria = siguiente compra orgánica. IPS solo como control.
- Online: A/B por user_id (VARIANTS), ≥5k requests por variante, separar nuevos/conocidos.

# EDA 02 · Diseño del modelo (sáb 14-nov, 16:35) — `notebooks/eda_02.py`, sin fuga temporal

## Cobertura de candidatos [history_sessions + history_events, solo eventos ANTERIORES a la sesión]
- De las compras del carrusel: 63 % ya las había visto el usuario (36 % en los últimos 7 días), 47 % ya las había comprado (43 % en los últimos 30 días). Igual en home.
- 74 % caen en la categoría top previa del usuario (sus eventos están un 55 % en esa categoría → hay lift). En home: 55 %.

## Posición pura (observado / esperado por la tasa del producto)
- Slot 1 = 1,00 · 2 = 0,63 · 3 = 0,55 · 5 = 0,44 · 10 = 0,37. Efecto real ≈ 2,7× (no 9×; el resto era popularidad).

## Afinidad usuario-producto: P(compra por impresión) [muestra 25 % de sesiones, 5 M impresiones]
| Relación previa | fuera de recent_views | en recent_views |
|---|---|---|
| nunca visto | 0,055 % | 0,70 % |
| visto <1 d / <7 d / <30 d / más | 0,29 / 0,22 / 0,16 / 0,14 % | 2,4 / 1,9 / 1,4 / 0,9 % |
| comprado <1 h | 4,4 % | 19 % |
| comprado <1 d / <7 d / <30 d / más | 2,9 / 2,0 / 1,2 / 0,74 % | 16 / 11 / 6,3 / 3,7 % |
- Lo comprado recientemente es lo que MÁS se vuelve a comprar (×50 frente a nunca visto) y decae con el tiempo. En el histórico no hay rastro de "no recompra". Contradice v3-no-rebuy → hay que testear en vivo.
- Recompra orgánica: mediana 245 h entre compras; 11 % en <1 día, 39 % en <7 días.

## Rebajas [shop.events, 1.250 productos con y sin descuento]
- Compras por vista +17 % con descuento; las vistas no cambian.

## Cold start [shop.events + catalog_updates]
- Productos nuevos convierten un 18 % más que su subcategoría y reciben ~44 vistas/h desde el lanzamiento (igual que los existentes). Se lanzan en lotes de 100 por hora.
- Usuarios nuevos compran igual que los conocidos (5,3 % compras/vista; 25 % en el top-100 histórico).
