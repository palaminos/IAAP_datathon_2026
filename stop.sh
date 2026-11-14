#!/bin/bash
# Para producción del todo (lanzador y recomendador). Úsalo solo si de verdad quieres dejar de responder:
# para cambiar de versión NO hace falta parar nada, basta con ./deploy.sh
cd "$(dirname "$0")" || exit 1
pkill -f "^(/bin/)?bash [^ ]*run_forever\.sh"
pkill -f "python3 -u .*recommender"
sleep 1
echo "Parado a mano con stop.sh: $(date "+%F %T")" | tee -a logs/production.log
echo "Producción está PARADA. Para volver a arrancarla:  ./deploy.sh   (o ./run_forever.sh)"
