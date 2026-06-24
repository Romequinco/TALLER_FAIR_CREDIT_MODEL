# -*- coding: utf-8 -*-
"""Ensambla las secciones HTML + CSS en un único informe.html."""
import pathlib

BASE = pathlib.Path(__file__).parent
CSS = (BASE / "report.css").read_text(encoding="utf-8")

ORDER = [
    "00_portada.html",
    "02_contexto.html",
    "03_metodologia.html",
    "04_tarea1.html",
    "05_tarea2.html",
    "06_tarea3.html",
    "07_tarea4.html",
    "08_resultados.html",
    "09_conclusiones.html",
]

# Sin saltos de página manuales: el contenido fluye y las figuras/callouts evitan
# cortarse por dentro (page-break-inside:avoid en el CSS).
PAGEBREAK_BEFORE = set()

parts = []
for name in ORDER:
    html = (BASE / "sections" / name).read_text(encoding="utf-8").strip()
    if name in PAGEBREAK_BEFORE:
        parts.append('<div class="pagebreak"></div>')
    parts.append(f"<!-- {name} -->\n{html}")

body = "\n\n".join(parts)

doc = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Taller B4-T1 · Crédito Justo y Confiable</title>
<style>
{CSS}
</style>
</head>
<body>
{body}
</body>
</html>
"""

out = BASE / "informe.html"
out.write_text(doc, encoding="utf-8")
print("escrito:", out, len(doc), "bytes")
