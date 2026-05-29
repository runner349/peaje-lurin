# ============================================================
# SISTEMA PEAJE LURÍN — Backend FastAPI
# ============================================================
# Despliega en: Render.com / Railway / Fly.io (gratis)
# Comando: uvicorn main:app --host 0.0.0.0 --port 8000
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import easyocr
import numpy as np
import requests
import json
import time
import re
import io
from datetime import datetime

# ── Configuración ─────────────────────────────────────────────
SUPABASE_URL = "https://kdhwjqakljlgkmrtacoe.supabase.co"
SUPABASE_KEY = "sb_publishable_jmvQZdNYBCo1Wa6wWp9uDQ_A5VpJ_c6"
TABLA_URL    = f"{SUPABASE_URL}/rest/v1/registro_placas"

HEADERS_SB = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=representation",
}

UMBRAL_CONFIANZA  = 0.50
PATRON_PLACA      = re.compile(r'[A-Z]{3}[-\s]?[0-9]{3}', re.IGNORECASE)

# ── App FastAPI ───────────────────────────────────────────────
app = FastAPI(
    title="Sistema Peaje Lurín",
    description="Microservicio de reconocimiento de placas vehiculares",
    version="1.0.0"
)

# CORS — permite que el sitio web llame al backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # En producción: pon la URL exacta de tu web
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Carga del modelo EasyOCR (una sola vez al arrancar) ───────
print("Cargando EasyOCR...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("✅ EasyOCR listo.")


# ── Funciones de procesamiento ────────────────────────────────

def bytes_a_imagen(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def preprocesar(img: np.ndarray) -> tuple:
    gris      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (5, 5), 0)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mejorado  = clahe.apply(suavizado)
    _, umbral = cv2.threshold(mejorado, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    nitidez   = cv2.Laplacian(gris, cv2.CV_64F).var()
    return umbral, nitidez


def limpiar_placa(texto: str) -> str:
    texto = texto.upper().strip()
    texto = re.sub(r'\s+', '-', texto)
    texto = re.sub(r'[^A-Z0-9\-]', '', texto)
    if re.match(r'^[A-Z]{3}[0-9]{3}$', texto):
        texto = f"{texto[:3]}-{texto[3:]}"
    return texto


def reconocer_ocr(img_proc: np.ndarray) -> tuple:
    resultados = reader.readtext(img_proc, detail=1, paragraph=False,
                                  width_ths=0.7, height_ths=0.7)
    if not resultados:
        return "", 0.0, []

    texto_total  = " ".join([r[1] for r in resultados])
    coincidencia = PATRON_PLACA.search(texto_total)

    if coincidencia:
        raw       = coincidencia.group(0)
        confianza = max(
            (r[2] for r in resultados if raw[:3].upper() in r[1].upper()),
            default=0.0
        )
    else:
        mejor     = max(resultados, key=lambda r: r[2])
        raw       = mejor[1]
        confianza = mejor[2]

    return limpiar_placa(raw), confianza, resultados


def guardar_supabase(placa: str, confianza: float, automatico: bool) -> dict:
    nivel = "ÓPTIMO" if confianza > 75 else ("ACEPTABLE" if confianza > 50 else "BAJO")
    payload = {
        "numero_placa":    placa,
        "confianza":       round(confianza, 2),
        "nivel_confianza": nivel,
        "es_automatico":   automatico,
    }
    try:
        r = requests.post(TABLA_URL, headers=HEADERS_SB,
                          data=json.dumps(payload), timeout=6)
        r.raise_for_status()
        datos = r.json()
        id_reg = datos[0]["id"] if isinstance(datos, list) and datos else None
        return {"ok": True, "id": id_reg, "nivel": nivel}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── Endpoints ─────────────────────────────────────────────────

@app.get("/")
def raiz():
    return {"servicio": "Sistema Peaje Lurín", "estado": "activo",
            "version": "1.0.0"}


@app.get("/health")
def health():
    return {"ok": True, "timestamp": datetime.now().isoformat()}


@app.post("/procesar-placa")
async def procesar_placa(imagen: UploadFile = File(...)):
    """
    Recibe una imagen JPG/PNG del vehículo, ejecuta el pipeline
    OpenCV + EasyOCR y registra en Supabase si la confianza > 50%.

    Retorna JSON con:
      - placa         : texto reconocido
      - confianza     : porcentaje 0-100
      - estado        : APROBADO | PENDIENTE_MANUAL | ERROR
      - id_registro   : id en Supabase (solo si APROBADO)
      - tiempo_proceso: segundos
      - motivo        : razón si fue rechazado
    """
    # Validar tipo de archivo
    if imagen.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400,
                            detail="Solo se aceptan imágenes JPG o PNG.")

    inicio = time.time()

    # Leer bytes y convertir a OpenCV
    data     = await imagen.read()
    img_orig = bytes_a_imagen(data)

    if img_orig is None:
        raise HTTPException(status_code=422, detail="No se pudo decodificar la imagen.")

    # Preprocesamiento
    img_proc, nitidez = preprocesar(img_orig)
    imagen_borrosa    = nitidez < 100

    # OCR
    placa, confianza, _ = reconocer_ocr(img_proc)
    confianza_pct       = round(confianza * 100, 2)
    tiempo              = round(time.time() - inicio, 3)

    # Decisión
    if confianza > UMBRAL_CONFIANZA and placa and not imagen_borrosa:
        db = guardar_supabase(placa, confianza_pct, automatico=True)
        if db["ok"]:
            return JSONResponse({
                "estado":        "APROBADO",
                "placa":         placa,
                "confianza":     confianza_pct,
                "nivel":         db["nivel"],
                "id_registro":   db["id"],
                "tiempo_proceso": tiempo,
                "mensaje":       "Barrera levantada. Registro automático exitoso.",
            })
        else:
            return JSONResponse({
                "estado":  "ERROR",
                "placa":   placa,
                "confianza": confianza_pct,
                "tiempo_proceso": tiempo,
                "error":   db["error"],
            }, status_code=500)
    else:
        if imagen_borrosa:
            motivo = f"Imagen borrosa (nitidez={nitidez:.0f}). Recapturar."
        elif not placa:
            motivo = "No se detectó patrón de placa válido."
        else:
            motivo = f"Confianza insuficiente: {confianza_pct}% (mínimo: 50%)"

        return JSONResponse({
            "estado":        "PENDIENTE_MANUAL",
            "placa":         placa,
            "confianza":     confianza_pct,
            "tiempo_proceso": tiempo,
            "motivo":        motivo,
            "mensaje":       "Validación manual requerida. Barrera retenida.",
        })


@app.get("/registros")
def obtener_registros(limite: int = 20):
    """Consulta los últimos registros de Supabase."""
    try:
        r = requests.get(TABLA_URL, headers=HEADERS_SB,
                         params={"order": "fecha_hora.desc", "limit": limite},
                         timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
