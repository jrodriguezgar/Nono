from pathlib import Path

import sqlite3

# About this module:
# It is used to analize files in a directory and store their paths in a SQLite database.

import logging
import traceback
import inspect
import os
import sys

# Define el entorno directamente aquí
MODULE_PATH = os.path.dirname(os.path.abspath(__file__)) + os.sep
print("Module Directory:", MODULE_PATH)
print("Module Name:", __name__)

global Environment
Environment = "DEVELOPMENT"  # Cambia a "PRODUCTION" para activar logging
#Environment = "PRE"

def set_Environment(p_environment):
    global Environment
    Environment = Environment

# Configurar logging solo si es producción
if Environment == "PRODUCTION" or Environment == "PRO":
    logging.basicConfig(
        filename=MODULE_PATH + 'app.log',
        level=logging.ERROR,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def stack_trace(e):
    # stack_frames = traceback.extract_stack()
    # print("Call Stack (usando extract_stack()):")
    # for frame in stack_frames:
    #     # Puedes filtrar para evitar mostrar el propio traceback.extract_stack()
    #     if frame.filename != __file__ or frame.name != 'extract_stack':
    #         print(f"  Archivo: {frame.filename}, Línea: {frame.lineno}, Función: {frame.name}, Código: {frame.line}")

    """
    Función personalizada para trazar errores.

    Args:
        exc_type: El tipo de excepción (e.g., ValueError, ZeroDivisionError).
        exc_value: La instancia de la excepción (el mensaje de error).
        exc_traceback: Un objeto traceback con la pila de llamadas.
    """
    exc_type = type(e)
    exc_value = e
    exc_traceback = e.__traceback__
        
    print("--- ¡Se ha detectado un error! ---")
    print(f"Tipo de error: {exc_type.__name__}")
    print(f"Mensaje del error: {exc_value}")

    print("\n--- Pila de llamadas (traceback) ---")
    # traceback.print_tb(exc_traceback) # Una opción para imprimir directamente
    
    # Para obtener la información como una lista de cadenas y poder manipularla:
    formato_traceback = traceback.format_exception(exc_type, exc_value, exc_traceback)
    for linea in formato_traceback:
        sys.stderr.write(linea) # Escribir en stderr es una buena práctica para errores
    print("------------------------------------")

def handle_error(e, p_environment):
    stack_trace(e)
    if Environment == "PRODUCTION":
        logging.error("Unhandled exception: %s", traceback.format_exc())
    else:
        raise e  # En desarrollo, relanzar para depurar

def print_inspect(msg):
    frame_info = inspect.stack()[1]  # Frame de la función que llamó a print_inspect
    args_info = inspect.getargvalues(frame_info.frame)
    fx_name = frame_info.function
    arguments = args_info.locals  # Diccionario de variables locales, incluyendo los argumentos

    print(f"Funcion: [{fx_name}] -> {msg} -> Argumentos: {arguments}")


# importing my libraries
#C:\app\OneDrive - Atresmedia Corporacion De Medios De Comunicacion SA\dev\code\vs_code
#sys.path.append('./python/libraries')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'libraries')))
import FastFormula as ffx


global files_ignore
global files_include
files_ignore = [
    '.tmp', '.log', '.bak','.thumb', '.lnk', '.sys', '.dll', '.exe',
    '.obj', '.so', '.bin', '.lock', '.cache'
]
# Lista de extensiones de archivos a incluir (comunes de documentos, imágenes, código, etc.)
files_include = [
    '.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tif', '.tiff',
    '.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mov', 
    '.zip', '.rar', '.7z', '.tar', '.gz'
]

def get_files(root_dir, db_path):
    """
    Recorre recursivamente el directorio root_dir y guarda en una base de datos SQLite
    los archivos encontrados, almacenando su ruta completa. Cada vez que se escanea un
    directorio, se guardan los resultados antes de pasar al siguiente.
    """
    # Crear/conectar a la base de datos
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Crear tabla si no existe
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL
        )
    """)
    conn.commit()

    # Recorrer directorios recursivamente
    for current_dir, _, files in os.walk(root_dir):
        file_paths = []
        for filename in files:
            full_path = os.path.join(current_dir, filename)

            ext = Path(full_path).suffix.lower()
            # Filtrado por extensiones a ignorar
            if files_ignore and ext in files_ignore:
                continue
            # Filtrado por extensiones a incluir
            if files_include and ext not in files_include:
                continue
            file_paths.append((full_path,))
        
        if file_paths:
            cursor.executemany("INSERT INTO files (path) VALUES (?)", file_paths)
            conn.commit()  # Guardar después de cada directorio

    conn.close()


def scan_directory(p_directory_path,p_pattern,p_recursive):
    ScannedFiles = []
    for root, _, files in os.walk(p_directory_path):
        for filename in files:
            full_path = os.path.join(root, filename)
            ScannedFiles.append(full_path)
    return ScannedFiles


def analize_propose_file(v_name,'name'):
    ffx.get_filename_from_path(p_path)


if __name__ == "__main__":
    files_path = r"C:\apps\OneDrive - Atresmedia Corporacion De Medios De Comunicacion SA"
    db_path="file_Analizer.db"

    #get_files(files_path, db_path)
    

    analize_propose_file(v_name,'name')



    print('organize_myfiles: The End')
