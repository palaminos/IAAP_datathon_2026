#!/bin/bash
# Deja UNA versión al 100 % del tráfico (para la final) y la despliega con todas las comprobaciones.
#
#   cd ~/Desktop/Q5/Datathon/rawtree && ./set_final.sh v12-cf
#
# Hace, en orden: cambia VARIANTS en recommender.py, pasa los tests y el preflight (si algo falla NO
# despliega), hace commit y push a la rama tarre, y lanza ../rawtree-prod/deploy.sh (que relanza producción
# sola). Para volver atrás: cd ../rawtree-prod && ./deploy.sh --rollback
set -euo pipefail
cd "$(dirname "$0")"
V="${1:-}"
[ -n "$V" ] || { echo "Uso: ./set_final.sh <versión>   (por ejemplo: ./set_final.sh v12-cf)"; exit 1; }
PY=/Applications/anaconda3/bin/python3
[ "$(git rev-parse --abbrev-ref HEAD)" = "tarre" ] || { echo "Esta carpeta no está en la rama tarre: no toco nada."; exit 1; }
"$PY" - "$V" <<'PYEOF'
import sys
from pathlib import Path
v = sys.argv[1]
p = Path("recommender.py")
s = p.read_text()
a, b = s.index("VARIANTS: dict[str, int] = {"), s.index("# Exploración: variante -> versión")
p.write_text(s[:a] + 'VARIANTS: dict[str, int] = {\n    "%s": 100,   # FINAL: una sola versión con todo el tráfico\n}\n\n' % v + s[b:])
PYEOF
"$PY" -m unittest discover -s tests
"$PY" recommender.py --preflight
git add -A .
git commit -q -m "FINAL: $V al 100 % del tráfico" -m "Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>"
git push -q origin tarre
echo "En GitHub: $(git ls-remote origin tarre | cut -c1-12) | aquí: $(git log --oneline | head -1)"
cd ../rawtree-prod && ./deploy.sh
