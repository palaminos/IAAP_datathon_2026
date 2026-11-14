#!/bin/bash
# Lanza el recomendador en producción y lo vuelve a arrancar si se cae.
#
#   ./run_forever.sh                      # recommender_v1.py
#   ./run_forever.sh otro_recommender.py  # otro script
#
# - Para cualquier recomendador que ya estuviera corriendo en ESTE Mac (para no responder dos veces).
# - Impide que el Mac se duerma mientras corre (caffeinate). Cerrar la tapa sí lo duerme: dejadla abierta.
# - Guarda la salida en logs/production.log (con fecha de cada arranque).
# - Para pararlo: Ctrl+C en esta terminal.

cd "$(dirname "$0")" || exit 1
SCRIPT="${1:-recommender_v1.py}"
mkdir -p logs
LOG=logs/production.log

if pgrep -f "recommender.*\.py" >/dev/null; then
    echo "Parando el recomendador que ya corría en este Mac..."
    pkill -f "recommender.*\.py"
    sleep 1
fi

trap 'echo; echo "Parado a mano: $(date "+%F %T")" | tee -a "$LOG"; exit 0' INT TERM

while true; do
    echo "=== $(date "+%F %T") arrancando $SCRIPT ===" | tee -a "$LOG"
    caffeinate -is python3 -u "$SCRIPT" 2>&1 | tee -a "$LOG"
    echo "=== $(date "+%F %T") $SCRIPT se ha parado; reinicio en 5 s ===" | tee -a "$LOG"
    sleep 5
done
