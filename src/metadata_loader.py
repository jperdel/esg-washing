"""
metadata_loader.py
------------------
Carga el fichero muestras_informes.xlsx y construye un diccionario de lookup
    (país_excel, empresa_excel_normalizada, año) → tipo_documento

Los nombres de carpeta y los nombres del Excel difieren bastante
(ej. "VINCI S.A" vs "VINCI"), por lo que se aplica:
  1. Normalización textual (mayúsculas, sin tildes, sin puntuación)
  2. Tabla de mapeo manual cuyas CLAVES son la forma ya normalizada
     del nombre de carpeta (para evitar dobles normalizaciones)
"""

from __future__ import annotations
import unicodedata
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Tablas de normalización
# ---------------------------------------------------------------------------

_COUNTRY_MAP: dict[str, str] = {
    "ALEMANIA":     "GERMANY",
    "BELGICA":      "BELGIUM",
    "ESPAÑA":       "SPAIN",
    "FINLANDIA":    "FINLAND",
    "FRANCIA":      "FRANCE",
    "ITALIA":       "ITALY",
    "PAISES BAJOS": "NETHERLANDS",
}

# CLAVES = _normalize(nombre_carpeta)  |  VALORES = nombre tal como aparece en el Excel
# (los valores se normalizan también al hacer el lookup, así que pueden escribirse
#  en cualquier capitalización/puntuación natural)
_COMPANY_MAP: dict[str, str] = {
    # España
    "BBVA":                                          "BCO BILBAO VIZCAYA ARGENTARIA",
    "BANCO SANTANDER S A":                           "BCO SANTANDER",
    "IBERDROLA S A":                                 "IBERDROLA",
    "INDITEX":                                       "INDUSTRIA DE DISENO TEXTIL SA",
    # Alemania
    "BMW GROUP":                                     "BMW",
    "DEUTSCHE BORSE":                                "DEUTSCHE BOERSE",
    "DEUTSCHE POST AG DHL":                          "DEUTSCHE POST",
    "DEUTSCHE TELEKOM AG":                           "DEUTSCHE TELEKOM",
    "INFINEON TECHNOLOGIES AG":                      "INFINEON TECHNOLOGIES",
    "MERCEDES BENZ GROUP AG ANTIGUO DAIMLER":        "MERCEDES BENZ GROUP",
    "MUNICH RE AG":                                  "MUENCHENER RUECK",
    "MUNICH RE AG":                                  "MUENCHENER RUECK",
    "SAP SE":                                        "SAP",
    "SIEMENS AG":                                    "SIEMENS",
    "VOLKSWAGEN AG VZ":                              "VOLKSWAGEN PREF",
    # Francia
    "AXA S A":                                       "AXA",
    "AIRBUS SE":                                     "AIRBUS",
    "BNP PARIBAS S A":                               "BNP PARIBAS",
    "COMPAGNIE DE SAINT GOBAIN S A":                 "SAINT GOBAIN",
    "HERMES INTERNATIONAL S A":                      "HERMES INTERNATIONAL",
    "KERING S A":                                    "KERING",
    "L OREAL S A":                                   "L OREAL",
    "LVMH SA":                                       "LVMH MOET HENNESSY",
    "PERNOD RICARD S A":                             "PERNOD RICARD",
    "SANOFI AVENTIS S A":                            "SANOFI",
    "SANOFI AVENTIS SA":                             "SANOFI",
    "TOTAL S A":                                     "TOTALENERGIES",
    "VINCI S A":                                     "VINCI",
    # Italia
    "ENI SPA":                                       "ENI",
    "FERRARI NV":                                    "FERRARI",
    "INTESA SANPAOLO S P A":                         "INTESA SANPAOLO",
    "INTESA SANPAOLO SPA":                           "INTESA SANPAOLO",
    "UNICREDIT SPA O UNICREDIT GROUP":               "UNICREDIT",
    "UNICREDIT GROUP":                               "UNICREDIT",
    # Países Bajos
    "AHOLD DELHAIZE KON EO 01":                      "AHOLD DELHAIZE",
    "ASML HOLDING NV":                               "ASML HLDG",
    "GRUPO ING NV":                                  "ING GRP",
    "STELLANTIS NV":                                 "STELLANTIS",
    "WOLTERS KLUWER NV":                             "WOLTERS KLUWER",
}


# ---------------------------------------------------------------------------
# Helpers de normalización
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Convierte a mayúsculas, elimina diacríticos y sustituye todo lo que no
    sea letra o dígito por espacios (colapsa espacios múltiples al final).
    """
    text = str(text).upper().strip()
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _resolve_country(folder_name: str) -> str:
    """Traduce el nombre de carpeta de país al código del Excel."""
    return _COUNTRY_MAP.get(folder_name.upper().strip(), folder_name.upper().strip())


def _resolve_company(folder_name: str) -> str:
    """
    Devuelve la clave normalizada de empresa para el lookup.
    Primero intenta el mapa manual (con clave = forma normalizada del nombre
    de carpeta); si no hay match, devuelve la normalización directa.
    El valor del mapa se normaliza también para garantizar consistencia
    con las claves generadas desde el Excel.
    """
    norm = _normalize(folder_name)
    mapped = _COMPANY_MAP.get(norm, norm)
    # Normalizar el valor del mapa para que coincida con _normalize(excel_company)
    return _normalize(mapped)


def _parse_year(year_raw) -> Optional[int]:
    """
    Convierte el año a entero de forma robusta:
    acepta strings ("2018"), enteros (2018) y floats (2018.0).
    """
    try:
        return int(float(str(year_raw)))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Cargador principal
# ---------------------------------------------------------------------------

def load_document_metadata(xlsx_path: Path) -> dict:
    """
    Lee la Hoja2 de muestras_informes.xlsx y devuelve:
        {(pais_excel_upper, empresa_normalizada, año_int): doc_type_str}
    """
    logger.info(f"Cargando metadatos de documentos desde: {xlsx_path}")

    df = pd.read_excel(xlsx_path, sheet_name=1, header=None)

    # Fila 1 (índice 1) es la cabecera real
    header_row = df.iloc[1].tolist()
    years = [int(y) for y in header_row[3:]
             if isinstance(y, (int, float)) and 2000 <= int(y) <= 2100]

    lookup: dict = {}

    for _, row in df.iloc[2:].iterrows():
        country_raw = row.iloc[0]
        company_raw = row.iloc[1]

        if pd.isna(country_raw) or pd.isna(company_raw):
            continue

        country_excel = str(country_raw).strip().upper()
        company_norm  = _normalize(str(company_raw))

        for i, year in enumerate(years):
            doc_type_raw = row.iloc[3 + i]
            if pd.isna(doc_type_raw) or doc_type_raw == 0:
                doc_type = None
            else:
                doc_type = str(doc_type_raw).strip()

            lookup[(country_excel, company_norm, year)] = doc_type

    logger.success(f"Metadatos cargados: {len(lookup)} entradas (empresa x ano)")
    return lookup


def get_doc_type(
    lookup: dict,
    folder_country: str,
    folder_company: str,
    year,
) -> str:
    """
    Consulta el tipo de documento para una combinación
    (carpeta_país, carpeta_empresa, año). Devuelve "Unknown" si no hay match.
    """
    country_key = _resolve_country(folder_country)
    company_key = _resolve_company(folder_company)
    year_int    = _parse_year(year)

    if year_int is None:
        logger.debug(f"Año no parseable: '{year}' para {folder_company}")
        return "Unknown"

    result = lookup.get((country_key, company_key, year_int))

    if result is None:
        logger.debug(
            f"Sin match: country='{country_key}' "
            f"company='{company_key}' year={year_int}"
        )
        return "Unknown"

    return result
