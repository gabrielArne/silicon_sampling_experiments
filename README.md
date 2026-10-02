# Silicon sampling · Q121

Un notebook independiente por cada modelo de `notebooks/llms.ipynb`, en formato
nativo de marimo (`.py`). Cada archivo incluye sus dependencias y todo el código
necesario para ejecutarlo en molab, sin importar módulos locales del repositorio.

| Modelo | Notebook |
| --- | --- |
| Qwen3-14B | [q121_qwen3_14b.py](notebooks/q121_qwen3_14b.py) |
| Qwen3-32B | [q121_qwen3_32b.py](notebooks/q121_qwen3_32b.py) |
| Qwen3.5-4B | [q121_qwen3_5_4b.py](notebooks/q121_qwen3_5_4b.py) |
| GPT-OSS-20B | [q121_gpt_oss_20b.py](notebooks/q121_gpt_oss_20b.py) |
| GPT-OSS-120B | [q121_gpt_oss_120b.py](notebooks/q121_gpt_oss_120b.py) |

El análisis conjunto se ejecuta con
[analysis_q121_models.py](notebooks/analysis_q121_models.py). Lee de Drive las
corridas más completas de cada modelo, calcula las métricas comparables con el
tutorial y guarda tablas, figuras e informe en `silicon_sampling/q121_analysis`.

## Ejecutar en molab

1. Abrí uno de los archivos `.py` en molab y seleccioná una máquina con GPU
   y memoria suficiente para ese modelo. Las dependencias están declaradas al
   comienzo del archivo. GPT-OSS incluye `kernels` y `triton` para sus pesos MXFP4;
   Qwen3.5 usa su procesador multimodal, con mensajes de texto solamente.
2. Subí `data/input/WVS_wave7_migracion_prompts.csv` al directorio de trabajo de
   molab, o guardalo en la carpeta `silicon_sampling` de Google Drive. También
   podés indicar `INPUT_DRIVE_FILE_ID`; si lo completás, tiene prioridad sobre
   el archivo local. `DATA_PATH` permite usar otra ruta.
3. La configuración inicial usa `N_SAMPLE = 1`: genera un perfil por país, tres
   respuestas por modelo. Cuando la prueba funcione, `N_SAMPLE = None` usa todos
   los perfiles válidos. Con el CSV
   actual hay 764 perfiles de Argentina, 1479 de Estados Unidos y 786 de Uruguay:
   **3029 perfiles por modelo**.
4. Pulsá **Iniciar / reanudar** y seguí el flujo de autorización de Google Drive,
   basado en `conexion_drive_ejemplo.ipynb`. Las credenciales se leen desde
   `~/.config/pydata/pydata_google_credentials.json`, configurable en la celda.
5. Los resultados aparecen en `data/output/` y en la carpeta `silicon_sampling`
   de Drive. Si la carpeta no existe, se crea. Si hay varias con ese nombre,
   indicá `DRIVE_FOLDER_ID` para elegir la correcta.

Para abrirlos localmente con las dependencias del propio notebook:

```bash
uvx marimo edit --sandbox notebooks/q121_qwen3_14b.py
```

Los notebooks cargan los pesos de Hugging Face en la máquina de ejecución.
GPT-OSS-120B requiere una GPU de la clase de 80 GB para el despliegue MXFP4
descrito por sus autores; los modelos Qwen grandes también necesitan abundante
memoria. Consultá las fichas enlazadas más abajo al elegir la máquina.

## CSV y reanudación

Se mantienen el filtro de países, las cinco opciones y el prompt de
`experiment_q121.ipynb`. Se envía un perfil por generación; la respuesta empírica
se conserva para comparar, pero no se incluye en el prompt.

Cada fila es un **intento**, con el esquema del repositorio de referencia:

- `empirical_id`, país y atributos del perfil.
- `empirical`: respuesta observada; `model`: respuesta simulada; `match`: coincidencia.
- `thinking`: justificación de 2–4 frases solicitada en el JSON.
- `raw_response`: generación completa, incluido el razonamiento nativo si existe.
- `valid`, `error`, `attempt`: validez e historial de fallos.
- `model_name`, `backend`, `temperature`, `seed`, `timestamp`, `elapsed_seconds`.
- `parameters_json`: configuración completa de generación, plantilla de chat,
  revisión del modelo, commit resuelto, versiones, tipo de datos y mapa de dispositivos.
- `system_prompt`, `user_prompt`, `run_id`: instrucciones y corrida identificable.

Después de cada intento se reemplaza el CSV local de forma atómica. Drive se
actualiza cada 10 intentos, al terminar y antes de propagar un error. Un error de generación queda registrado y detiene el
bucle. Una respuesta con JSON u opción inválida queda registrada y el bucle sigue.
Al reanudar, se omiten únicamente los perfiles con una respuesta válida; los
fallidos reciben un nuevo intento y una nueva semilla, sin borrar el historial.
Para analizar respuestas finales, filtrá `valid == True`.

Si falla una subida, el bucle se detiene conservando la copia local. Corregí la
conexión y volvé a ejecutar la última celda: primero une las copias local y remota
y sincroniza lo pendiente. En una sesión nueva, abrí el mismo notebook con la
misma configuración y volvé a autorizar Drive. Si molab pierde su disco antes de
completar una subida, se retoma desde el último intento guardado en Drive.
La generación que estaba en curso cuando se interrumpió se repite.

El nombre del CSV incluye una huella de datos, prompt, pesos, versiones y parámetros.
Cambiar esos elementos inicia otro archivo; ampliar `N_SAMPLE` permite completar
el mismo experimento. La semilla depende del perfil y del número de intento, por
lo que reanudar no cambia las semillas de los demás perfiles. Esto no garantiza
resultados idénticos entre distintos equipos GPU.

Ejecutá una sola sesión por modelo y configuración: la sincronización no coordina
escrituras concurrentes. Para reanudar, conservá la revisión del modelo y las
versiones usadas en la corrida original; quedan registradas en `parameters_json`.

## Verificación

```bash
uv run --with pytest --with pandas pytest tests/test_q121_notebooks.py -q
uvx marimo check notebooks/q121_*.py
```

Las pruebas usan respuestas y Drive simulados. Comprueban la fidelidad del prompt,
la separación de razonamiento y respuesta final, el guardado atómico, las caídas
del modelo, las fallas de Drive, las interrupciones y la recuperación en una sesión
nueva. No descargan pesos ni realizan subidas reales.

## Referencias de ejecución

- [Dependencias autocontenidas de marimo](https://docs.marimo.io/guides/package_management/sandboxes/).
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) y [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) y [su implementación en Transformers](https://huggingface.co/docs/transformers/model_doc/qwen3_5).
- [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) y [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b).

