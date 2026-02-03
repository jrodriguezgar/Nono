# SSL Configuration Guide

Este módulo proporciona tres opciones para configurar la verificación de certificados SSL cuando se conecta a servicios de IA generativa.

## Opciones Disponibles

### Opción 1: Modo INSECURE (Por Defecto)

**⚠️ ADVERTENCIA: Solo para desarrollo/testing**

Este modo deshabilita completamente la verificación SSL. Es útil para desarrollo rápido pero **NO DEBE USARSE EN PRODUCCIÓN**.

```python
from fastetl.connectors.connector_genai import configure_ssl_verification, SSLVerificationMode

# Ya está configurado por defecto, pero puedes llamarlo explícitamente:
configure_ssl_verification(SSLVerificationMode.INSECURE)
```

### Opción 2: Modo CERTIFI (Recomendado para Producción)

**✓ RECOMENDADO para entornos de producción**

Este modo usa el paquete `certifi` que proporciona certificados raíz actualizados de Mozilla.

```python
from fastetl.connectors.connector_genai import configure_ssl_verification, SSLVerificationMode

# Configurar antes de usar cualquier servicio de IA
configure_ssl_verification(SSLVerificationMode.CERTIFI)
```

Si `certifi` no está instalado, el módulo intentará instalarlo automáticamente.

### Opción 3: Modo CUSTOM (Certificados Corporativos)

**✓ Para entornos corporativos con certificados propios**

Este modo te permite especificar un archivo de certificado personalizado (por ejemplo, el certificado raíz de tu empresa).

```python
from fastetl.connectors.connector_genai import configure_ssl_verification, SSLVerificationMode

# Especificar la ruta al certificado corporativo
configure_ssl_verification(
    SSLVerificationMode.CUSTOM,
    custom_cert_path=r'C:\certificados\atresmedia-root-ca.crt'
)
```

## Uso en modulo python (main.py)

Para cambiar la configuración SSL en el archivo principal, añade estas líneas **ANTES** de importar o usar cualquier servicio de IA:

```python
# Al inicio del archivo, después de los imports básicos
from fastetl.connectors.connector_genai import configure_ssl_verification, SSLVerificationMode

# Opción 1: Desarrollo (inseguro)
configure_ssl_verification(SSLVerificationMode.INSECURE)

# Opción 2: Producción con certifi (recomendado)
# configure_ssl_verification(SSLVerificationMode.CERTIFI)

# Opción 3: Certificado corporativo
# configure_ssl_verification(
#     SSLVerificationMode.CUSTOM,
#     custom_cert_path=r'C:\certificados\atresmedia-root-ca.crt'
# )
```

## Variables de Entorno

Cada modo configura diferentes variables de entorno:

### Modo INSECURE:

- `PYTHONHTTPSVERIFY=0`
- `CURL_CA_BUNDLE=''`
- `REQUESTS_CA_BUNDLE=''`

### Modo CERTIFI:

- `REQUESTS_CA_BUNDLE=<ruta a certifi>`
- `SSL_CERT_FILE=<ruta a certifi>`
- `CURL_CA_BUNDLE=<ruta a certifi>`

### Modo CUSTOM:

- `REQUESTS_CA_BUNDLE=<ruta personalizada>`
- `SSL_CERT_FILE=<ruta personalizada>`
- `CURL_CA_BUNDLE=<ruta personalizada>`

## Solución de Problemas

### Error: `SSL_ERROR_SSL: CERTIFICATE_VERIFY_FAILED`

Este error ocurre cuando Python no puede verificar el certificado SSL del servidor.

**Soluciones:**

1. **Desarrollo rápido:** Usa modo INSECURE (ya configurado por defecto)
2. **Producción:** Usa modo CERTIFI
3. **Detrás de un proxy corporativo:** Usa modo CUSTOM con el certificado de tu empresa

### ¿Cómo obtener el certificado corporativo?

En Windows, puedes exportar el certificado raíz:

1. Abre el Administrador de Certificados: `certmgr.msc`
2. Navega a: Entidades de certificación raíz de confianza > Certificados
3. Busca el certificado de tu empresa (ej: "Atresmedia Root CA")
4. Clic derecho > Todas las tareas > Exportar
5. Selecciona formato "X.509 codificado base 64 (.CER)"
6. Guarda el archivo y usa su ruta en el modo CUSTOM

### Verificar qué modo está activo

El módulo imprime un mensaje en la consola al configurarse:

- `⚠️ SSL verification DISABLED` → Modo INSECURE
- `✓ SSL verification enabled using certifi` → Modo CERTIFI
- `✓ SSL verification enabled using custom certificate` → Modo CUSTOM

## Migración a Producción

**IMPORTANTE:** Antes de pasar a producción, cambia el modo de INSECURE a CERTIFI:

```python
# En main_guardianZ.py, cambiar de:
configure_ssl_verification(SSLVerificationMode.INSECURE)

# A:
configure_ssl_verification(SSLVerificationMode.CERTIFI)
```

Esto asegura que las conexiones SSL sean seguras y verificadas.
