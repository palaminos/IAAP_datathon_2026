#!/bin/bash
# Lanza el recomendador en producción y lo vuelve a arrancar si se cae.
#
#   ./run_forever.sh                      # recommender.py
#   ./run_forever.sh otro_recommender.py  # otro script
#
# - Para cualquier recomendador que ya estuviera corriendo en ESTE Mac (para no responder dos veces).
# - Impide que el Mac se duerma mientras corre (caffeinate). Cerrar la tapa sí lo duerme: dejadla abierta.
# - Guarda la salida en logs/production.log (con fecha de cada arranque).
# - Para pararlo: Ctrl+C en esta terminal.

cd "$(dirname "$0")" || exit 1
SCRIPT="${1:-recommender.py}"
mkdir -p logs
LOG=logs/production.log

# Elegir un Python que tenga las librerías (en cada Mac el "python3" por defecto puede ser otro).
# Se puede forzar con:  PYTHON=/ruta/a/python3 ./run_forever.sh
if [ -z "$PYTHON" ]; then
    for p in /Applications/anaconda3/bin/python3 /opt/anaconda3/bin/python3 "$HOME/anaconda3/bin/python3" \
             /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /opt/homebrew/bin/python3 \
             /usr/local/bin/python3 /usr/bin/python3; do
        if [ -x "$p" ] && "$p" -c "import requests, numpy" 2>/dev/null; then PYTHON="$p"; break; fi
    done
fi
if [ -z "$PYTHON" ]; then
    echo "No encuentro un python3 con 'requests' y 'numpy'. Instálalo:  python3 -m pip install -r starter/requirements.txt"
    exit 1
fi
echo "Usando $PYTHON"

if pgrep -f "python3 -u .*recommender" >/dev/null; then
    echo "Parando el recomendador que ya corría en este Mac..."
    pkill -f "python3 -u .*recommender"
    sleep 1
fi

trap 'echo; echo "Parado a mano: $(date "+%F %T")" | tee -a "$LOG"; exit 0' INT TERM

while true; do
    echo "=== $(date "+%F %T") arrancando $SCRIPT ===" | tee -a "$LOG"
    caffeinate -is "$PYTHON" -u "$SCRIPT" 2>&1 | tee -a "$LOG"
    echo "=== $(date "+%F %T") $SCRIPT se ha parado; reinicio en 5 s ===" | tee -a "$LOG"
    sleep 5
done
