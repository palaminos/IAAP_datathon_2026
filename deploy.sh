#!/bin/bash
# Despliega una versión en producción SIN parar antes lo que está corriendo.
#
#   cd ~/Desktop/Q5/Datathon/rawtree-prod
#   ./deploy.sh              # despliega origin/tarre
#   ./deploy.sh origin/main  # u otra rama o commit
#   ./deploy.sh --rollback   # vuelve a la versión anterior
#
# Se ejecuta en una terminal DISTINTA de la que tiene ./run_forever.sh. Pasos:
#   1. Comprueba que estás en la carpeta de producción y que no hay cambios hechos a mano.
#   2. Descarga la versión y la PRUEBA aparte (100 requests reales por variante, sin enviar nada).
#   3. Solo si la prueba pasa: cambia el código y reinicia el recomendador (run_forever.sh lo relanza
#      solo en ~2 s). Si algo falla antes, producción sigue como estaba.
#   4. Apunta la versión anterior en logs/previous_version para poder volver.

set -u
DIR="$(pwd)"
fail() { echo "✗ $1"; echo "  Producción NO se ha tocado."; exit 1; }

git rev-parse --git-dir >/dev/null 2>&1 || fail "Esta carpeta no es el repositorio. Haz: cd ~/Desktop/Q5/Datathon/rawtree-prod"
if BRANCH=$(git symbolic-ref -q --short HEAD); then
    fail "Esta carpeta está en la rama '$BRANCH': es la de TRABAJO, no la de producción. Haz: cd ~/Desktop/Q5/Datathon/rawtree-prod"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    git status --short --untracked-files=no
    fail "Hay ficheros editados a mano en producción (arriba). Descártalos con:  git checkout -- .   y vuelve a lanzar ./deploy.sh"
fi

mkdir -p logs
if [ "${1:-}" = "--rollback" ]; then
    [ -f logs/previous_version ] || fail "No hay versión anterior apuntada (logs/previous_version)."
    REF=$(cat logs/previous_version)
else
    REF="${1:-origin/tarre}"
    git fetch -q || fail "No se pudo descargar de GitHub (¿hay internet?)."
fi
NEW=$(git rev-parse --verify -q "$REF^{commit}") || fail "No existe la versión '$REF'."
OLD=$(git rev-parse HEAD)
if [ "$NEW" = "$OLD" ]; then
    echo "El código ya está en esa versión: $(git log --oneline -1)"
    if pgrep -f "python3 -u .*recommender" >/dev/null; then
        echo "✓ Y el recomendador está corriendo."
    else
        echo "✗ PERO NO HAY NINGÚN RECOMENDADOR CORRIENDO: producción está PARADA."
        echo "  Lánzalo ya en la terminal de producción:  ./run_forever.sh"
        exit 1
    fi
    exit 0
fi
echo "Actual:    $(git log --oneline -1 "$OLD")"
echo "Candidata: $(git log --oneline -1 "$NEW")"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for p in /Applications/anaconda3/bin/python3 /opt/anaconda3/bin/python3 "$HOME/anaconda3/bin/python3" \
             /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 /opt/homebrew/bin/python3 \
             /usr/local/bin/python3 /usr/bin/python3; do
        if [ -x "$p" ] && "$p" -c "import requests, numpy" 2>/dev/null; then PYTHON="$p"; break; fi
    done
fi
[ -n "$PYTHON" ] || fail "No encuentro un python3 con 'requests' y 'numpy'."

# Probar la candidata en una carpeta aparte
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git archive "$NEW" | tar -x -C "$TMP" || fail "No se pudo extraer la versión."
cp .env "$TMP/.env" 2>/dev/null || fail "Falta el fichero .env en esta carpeta."
mkdir -p cache && ln -s "$DIR/cache" "$TMP/cache"
if grep -q -- "--preflight" "$TMP/recommender.py" 2>/dev/null; then
    echo "Probando la candidata (sin enviar nada)..."
    (cd "$TMP" && "$PYTHON" recommender.py --preflight) || fail "La candidata NO ha pasado la prueba."
else
    echo "AVISO: esta versión es anterior a la prueba automática; se despliega sin probar."
fi

# Cambiar el código y reiniciar
echo "$OLD" > logs/previous_version
git checkout -q --detach "$NEW" || fail "git no pudo cambiar de versión."
echo "✓ Código cambiado a: $(git log --oneline -1)"
if [ "${NO_RESTART:-}" = 1 ]; then
    echo "NO_RESTART=1: código cambiado, recomendador sin reiniciar."
elif pgrep -f "run_forever.sh" >/dev/null; then
    pkill -f "python3 -u .*recommender"
    echo "Reiniciando el recomendador (run_forever.sh lo relanza solo)..."
    for _ in $(seq 1 40); do
        sleep 1
        if tail -5 logs/production.log 2>/dev/null | grep -q "listo:"; then break; fi
    done
    tail -5 logs/production.log | grep "listo:" || echo "AVISO: aún no veo 'listo:' en logs/production.log; mira la terminal de producción."
    echo "✓ Desplegado. Para volver atrás: ./deploy.sh --rollback"
else
    echo "No hay ningún run_forever.sh corriendo. Lánzalo:  ./run_forever.sh"
fi
