# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.20",
#     "pandas>=2.2,<4",
#     "torch>=2.6",
#     "transformers>=5.3,<6",
#     "accelerate>=1.10",
#     "kernels>=0.10",
#     "triton>=3.4; sys_platform == 'linux'",
#     "gdrive-fsspec",
#     "google-api-python-client>=2",
#     "google-auth>=2",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import io
    import json
    import os
    import re
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    return Path, datetime, hashlib, io, json, mo, os, pd, re, time, timezone


@app.cell
def _(mo):
    mo.md("""
    # Q121 · GPT-OSS-120B

    Simulación de los perfiles de Argentina, Uruguay y Estados Unidos.
    Conserva el prompt de `experiment_q121.ipynb` y el esquema del tutorial.

    1. Seleccioná una máquina con GPU y revisá la configuración.
    2. En molab, subí el CSV de entrada o guardalo en la carpeta de Drive
       `silicon_sampling`. También podés indicar su ID en `INPUT_DRIVE_FILE_ID`.
    3. Pulsá **Iniciar / reanudar** y completá la conexión OAuth de Drive.

    Se guarda **cada intento** en un CSV local y Drive se sincroniza por lotes. Si la subida falla,
    la ejecución se detiene con la copia local guardada. Volvé a ejecutar la
    última celda para sincronizarla y continuar. En una sesión nueva, pulsá
    nuevamente el botón: se recupera el CSV de Drive.

    `thinking` es la justificación breve solicitada al modelo;
    `raw_response` conserva también el razonamiento nativo, si lo emite.
    Ejecutá una sola sesión por modelo y configuración para evitar escrituras
    simultáneas sobre el mismo CSV.
    """)
    return


@app.cell
def _(Path):
    MODEL_NAME = "openai/gpt-oss-120b"
    MODEL_REVISION = "main"  # Se registra el commit resuelto; podés fijarlo aquí.
    COUNTRIES = ["Argentina", "United States", "Uruguay"]
    N_SAMPLE = 1  # Prueba inicial: un perfil por país (3 respuestas en total).
    DRIVE_SYNC_EVERY = 10  # El CSV local sí se guarda después de cada intento.
    RANDOM_SEED = 42
    GENERATION = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 50,
    }
    CHAT_TEMPLATE = {"reasoning_effort": "low"}

    DATA_PATH = Path("data/input/WVS_wave7_migracion_prompts.csv")
    INPUT_DRIVE_FILE_ID = ""  # Opcional: CSV de entrada en Drive.
    DRIVE_FOLDER_ID = ""  # Opcional: ID de la carpeta silicon_sampling.
    DRIVE_FOLDER_NAME = "silicon_sampling"
    OUTPUT_DIR = Path("data/output")
    CREDENTIALS_PATH = Path.home() / ".config/pydata/pydata_google_credentials.json"
    return (
        CHAT_TEMPLATE, COUNTRIES, CREDENTIALS_PATH, DATA_PATH,
        DRIVE_FOLDER_ID, DRIVE_FOLDER_NAME, DRIVE_SYNC_EVERY, GENERATION, INPUT_DRIVE_FILE_ID,
        MODEL_NAME, MODEL_REVISION, N_SAMPLE, OUTPUT_DIR, RANDOM_SEED,
    )


@app.cell
def _(mo):
    start = mo.ui.run_button(label="Iniciar / reanudar")
    start
    return (start,)


@app.cell
def _():
    OPTIONS = [
        "Very good", "Quite good", "Neither good, nor bad", "Quite bad", "Rather bad"
    ]
    # Mismo texto que experiment_q121.ipynb.
    SYSTEM_PROMPT = """You will be given a self-description by a survey respondent
    in their own words. Your task is to answer the survey question as that person
    would, adopting their perspective, values, and lived experience as described.

    Critical rules:
    1. Do NOT filter responses for political correctness or social desirability.
    2. The five response options are equally valid — do NOT default to the neutral
       middle option. Assign extreme evaluations (Very good / Rather bad) when
       the profile supports them.
    3. Let the respondent's occupation, class, education, and political ideology
       guide the evaluation — these shape how people perceive immigration's impact.
    4. Reason briefly in the "thinking" field (2-4 sentences) in first person,
       as the respondent — not as an outside observer.
    5. Respond EXCLUSIVELY in the JSON format specified. No additional text.
    6. Respond with the EXACT option string provided.
    """
    # Elimina la indentación del cuerpo de la celda, sin cambiar el prompt.
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace("\n    ", "\n")

    def build_profile(row):
        return (
            f"I am a {row['sex']} living in {row['country']}. "
            f"I work as {row['occupation']} and consider myself {row['class']}. "
            f"My highest level of education is {row['education']}. "
            f"In political terms, I would describe myself as {row['political_identity']}."
        )

    def build_prompt_q121(row):
        opts = "\n".join(f'  - "{option}"' for option in OPTIONS)
        return (
            f"Respondent profile:\n{build_profile(row)}\n"
            "Answer the following WVS question as this person would.\n\n"
            "Q121. How would you evaluate the impact of immigrants on the "
            "development of your country?\n"
            f"Valid options (use exact string):\n{opts}\n\n"
            "Return ONLY this JSON (no markdown, no extra text):\n"
            "{\n"
            '  "thinking": "<brief reasoning, 2-4 sentences>",\n'
            '  "Q121": "<exact option string>"\n'
            "}"
        )

    return OPTIONS, SYSTEM_PROMPT, build_prompt_q121


@app.cell
def _(CREDENTIALS_PATH, DRIVE_FOLDER_ID, DRIVE_FOLDER_NAME, json, mo, start):
    mo.stop(not start.value, mo.md("Revisá los parámetros y pulsá **Iniciar / reanudar**."))
    from gdrive_fsspec import GoogleDriveFileSystem
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload

    # Mismo flujo OAuth que conexion_drive_ejemplo.ipynb.
    if not CREDENTIALS_PATH.exists():
        _fs = GoogleDriveFileSystem(
            use_listings_cache=False,
            skip_instance_cache=True,
            auth_kwargs={"use_local_webserver": False},
        )
        mo.output.clear_console()
    _data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    _creds = Credentials(
        token=_data.get("token"),
        refresh_token=_data.get("refresh_token"),
        token_uri=_data.get("token_uri"),
        client_id=_data.get("client_id"),
        client_secret=_data.get("client_secret"),
        scopes=_data.get("scopes", ["https://www.googleapis.com/auth/drive"]),
    )
    if not _creds.valid:
        _creds.refresh(Request())
    drive = build("drive", "v3", credentials=_creds, cache_discovery=False)

    def find_drive_file(name, folder_id=None, folders_only=False):
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        query = f"name='{escaped}' and trashed=false"
        if folder_id:
            query += f" and '{folder_id}' in parents"
        if folders_only:
            query += " and mimeType='application/vnd.google-apps.folder'"
        matches = drive.files().list(q=query, fields="files(id)", pageSize=2).execute(num_retries=3)["files"]
        if len(matches) > 1:
            raise ValueError(f"Hay varios archivos/carpetas llamados {name}. Usá un ID o quitá los duplicados.")
        return matches[0]["id"] if matches else None

    folder_id = DRIVE_FOLDER_ID or find_drive_file(DRIVE_FOLDER_NAME, folders_only=True)
    if not folder_id:
        folder_id = drive.files().create(
            body={"name": DRIVE_FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"},
            fields="id",
        ).execute()["id"]
    print("Conexión con Drive lista.")
    return MediaIoBaseUpload, drive, find_drive_file, folder_id


@app.cell
def _(
    COUNTRIES, DATA_PATH, INPUT_DRIVE_FILE_ID, N_SAMPLE, OPTIONS, Path,
    RANDOM_SEED, drive, find_drive_file, folder_id, hashlib, io, pd,
):
    _candidates = [DATA_PATH, Path("..") / DATA_PATH, Path(DATA_PATH.name)]
    _local = next((p for p in _candidates if p.is_file()), None)
    if INPUT_DRIVE_FILE_ID:
        input_bytes = drive.files().get_media(fileId=INPUT_DRIVE_FILE_ID).execute(num_retries=3)
    elif _local:
        input_bytes = _local.read_bytes()
    else:
        _input_id = find_drive_file(DATA_PATH.name, folder_id)
        if not _input_id:
            raise FileNotFoundError(f"Subí {DATA_PATH.name} a molab o a la carpeta de Drive.")
        input_bytes = drive.files().get_media(fileId=_input_id).execute(num_retries=3)

    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    profiles = pd.read_csv(io.BytesIO(input_bytes), dtype=str, keep_default_na=False).rename(columns={
        "B_COUNTRY_LABEL": "country", "sex_label": "sex", "education_label": "education",
        "class_label": "class", "occupation_label": "occupation",
        "political_ident": "political_identity", "Q121": "q121_response",
    })
    profiles = profiles.loc[
        profiles["country"].isin(COUNTRIES) & profiles["q121_response"].isin(OPTIONS),
        ["id", "country", "sex", "education", "class", "occupation", "political_identity", "q121_response"],
    ].copy()
    if profiles.empty or profiles["id"].eq("").any() or profiles.duplicated(["country", "id"]).any():
        raise ValueError("Se necesitan perfiles con identificadores únicos por país y respuestas válidas.")
    if N_SAMPLE is not None:
        if not isinstance(N_SAMPLE, int) or N_SAMPLE < 1:
            raise ValueError("N_SAMPLE debe ser None o un entero positivo.")
        profiles = pd.concat([
            group.sample(n=min(N_SAMPLE, len(group)), random_state=RANDOM_SEED)
            for _, group in profiles.groupby("country", sort=True)
        ])
    profiles = profiles.sort_values(["country", "id"]).reset_index(drop=True)
    print(profiles.groupby("country").size().to_string())
    return input_sha256, profiles


@app.cell
def _(CHAT_TEMPLATE, GENERATION, MODEL_NAME, MODEL_REVISION, SYSTEM_PROMPT, build_prompt_q121, profiles):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Seleccioná una máquina con GPU en molab antes de cargar el modelo.")
    # La dependencia de profiles valida los datos y Drive antes de descargar pesos.
    print(f"Cargando {MODEL_NAME} para {len(profiles)} perfiles...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, device_map="auto", dtype="auto",
    )
    llm.eval()
    generation_config = transformers.GenerationConfig.from_dict(llm.generation_config.to_dict())
    generation_config.update(**GENERATION)
    runtime = {
        "model_commit": getattr(llm.config, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "dtype": str(llm.dtype),
        "device_map": getattr(llm, "hf_device_map", {}),
        "generation_config": generation_config.to_dict(),
    }

    def generate_one(row, seed):
        set_seed(seed)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt_q121(row)},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", **CHAT_TEMPLATE,
        ).to(llm.device)
        with torch.inference_mode():
            output = llm.generate(**inputs, generation_config=generation_config)
        generated = output[0, inputs["input_ids"].shape[-1]:]
        # Conserva tokens de canales/razonamiento para no confundirlos con el JSON final.
        return tokenizer.decode(generated, skip_special_tokens=False)

    return generate_one, runtime


@app.cell
def _(CHAT_TEMPLATE, MODEL_NAME, OPTIONS, json, re):
    def parse_response(raw):
        text = raw.strip()
        if MODEL_NAME.startswith("openai/gpt-oss"):
            # Harmony separa analysis de final; nunca aceptar JSON del análisis.
            marker = "<|channel|>final<|message|>"
            if marker not in text:
                raise ValueError("La generación no llegó al canal final de Harmony.")
            text = text.rsplit(marker, 1)[1]
        elif "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        elif CHAT_TEMPLATE.get("enable_thinking") or "<think>" in text:
            raise ValueError("El razonamiento quedó truncado antes de la respuesta final.")
        text = re.sub(r"<\|[^>]+\|>", "", text).strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict) or parsed.get("Q121") not in OPTIONS:
            raise ValueError("La respuesta no contiene una opción válida de Q121.")
        if not isinstance(parsed.get("thinking"), str) or not parsed["thinking"].strip():
            raise ValueError("La respuesta no contiene una justificación en thinking.")
        return parsed

    return (parse_response,)


@app.cell
def _(
    CHAT_TEMPLATE, GENERATION, MODEL_NAME, MODEL_REVISION, OUTPUT_DIR,
    RANDOM_SEED, SYSTEM_PROMPT, build_prompt_q121, hashlib, input_sha256, json, runtime,
):
    # Cambiar datos, prompt, pesos o parámetros crea otro CSV, sin mezclar corridas.
    _example = dict.fromkeys(["sex", "country", "occupation", "class", "education", "political_identity"], "{profile}")
    parameters = {
        "schema_version": 1, "model_name": MODEL_NAME, "model_revision": MODEL_REVISION,
        "question": "Q121", "input_sha256": input_sha256, "random_seed": RANDOM_SEED,
        "generation": GENERATION, "chat_template": CHAT_TEMPLATE,
        "system_prompt": SYSTEM_PROMPT, "prompt_template": build_prompt_q121(_example),
        **runtime,
    }
    parameters_json = json.dumps(parameters, ensure_ascii=False, sort_keys=True, default=str)
    run_id = hashlib.sha256(parameters_json.encode()).hexdigest()[:16]
    output_path = OUTPUT_DIR / f"{MODEL_NAME.replace('/', '_')}_Q121_{run_id}.csv"
    print(f"CSV: {output_path.name}")
    return output_path, parameters_json, run_id


@app.cell
def _(MediaIoBaseUpload, drive, find_drive_file, folder_id, io, os, pd):
    def save_local(records, path):
        """Reemplazo atómico: una interrupción no deja el CSV anterior a medias."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            pd.DataFrame(records).to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def upload_checkpoint(path):
        """Busca por nombre también al reintentar una creación de resultado incierto."""
        file_id = find_drive_file(path.name, folder_id)
        media = MediaIoBaseUpload(io.BytesIO(path.read_bytes()), mimetype="text/csv", resumable=True)
        if file_id:
            drive.files().update(fileId=file_id, media_body=media).execute(num_retries=3)
        else:
            drive.files().create(
                body={"name": path.name, "parents": [folder_id]}, media_body=media, fields="id",
            ).execute()

    def restore_checkpoint(path, expected_run_id):
        """Une la copia remota y la local, conservando cada intento una sola vez."""
        sources = []
        file_id = find_drive_file(path.name, folder_id)
        if file_id:
            sources.append(io.BytesIO(drive.files().get_media(fileId=file_id).execute(num_retries=3)))
        if path.exists():
            sources.append(path)
        records = {}
        for source in sources:
            frame = pd.read_csv(source, dtype=str, keep_default_na=False)
            required = {"run_id", "country", "empirical_id", "attempt", "valid", "model", "thinking"}
            if not required.issubset(frame.columns) or not frame["run_id"].eq(expected_run_id).all():
                raise ValueError("El checkpoint no corresponde a esta corrida o tiene un esquema inválido.")
            for record in frame.to_dict("records"):
                record["attempt"] = int(record["attempt"])
                record["valid"] = record["valid"].lower() == "true"
                key = (record["country"], record["empirical_id"], record["attempt"])
                if key in records and records[key] != record:
                    raise ValueError("Hay intentos distintos con la misma clave. Revisá las copias local y remota.")
                records[key] = record
        return list(records.values())

    return restore_checkpoint, save_local, upload_checkpoint


@app.cell
def _(
    DRIVE_SYNC_EVERY, GENERATION, MODEL_NAME, RANDOM_SEED, SYSTEM_PROMPT, build_prompt_q121,
    datetime, generate_one, hashlib, output_path, parameters_json, parse_response,
    pd, profiles, restore_checkpoint, run_id, save_local, time, timezone, upload_checkpoint,
):
    def run_experiment():
        # Se lee de disco/Drive en cada ejecución, incluso si se reejecuta solo esta celda.
        records = restore_checkpoint(output_path, run_id)
        done = {(r["country"], r["empirical_id"]) for r in records if r["valid"]}
        if records:
            save_local(records, output_path)
            upload_checkpoint(output_path)  # Primero recupera subidas que quedaron pendientes.
        pending = [row for row in profiles.to_dict("records") if (row["country"], row["id"]) not in done]
        print(f"{len(profiles) - len(pending)} perfiles completos; {len(pending)} pendientes.")
        for index, row in enumerate(pending, start=1):
            previous = [r["attempt"] for r in records if (r["country"], r["empirical_id"]) == (row["country"], row["id"])]
            attempt = max(previous, default=0) + 1
            # Semilla por perfil/intento: reanudar no desplaza la secuencia de los demás.
            key = f"{RANDOM_SEED}:{row['country']}:{row['id']}:{attempt}"
            seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
            raw, parsed, error, generation_error = "", {}, "", None
            started = time.monotonic()
            try:
                raw = generate_one(row, seed)
            except Exception as exc:
                generation_error = exc
                error = f"{type(exc).__name__}: {exc}"
            if generation_error is None:
                try:
                    parsed = parse_response(raw)
                except (ValueError, TypeError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            valid = not error
            records.append({
                "run_id": run_id, "empirical_id": row["id"], "country": row["country"],
                "question": "Q121", "sex": row["sex"], "education": row["education"],
                "class": row["class"], "occupation": row["occupation"],
                "political_ident": row["political_identity"], "empirical": row["q121_response"],
                "model": parsed.get("Q121", ""), "thinking": parsed.get("thinking", ""),
                "match": valid and row["q121_response"] == parsed.get("Q121"),
                "valid": valid, "raw_response": raw, "error": error, "attempt": attempt,
                "backend": "transformers", "model_name": MODEL_NAME,
                "temperature": GENERATION["temperature"], "seed": seed,
                "parameters_json": parameters_json, "system_prompt": SYSTEM_PROMPT,
                "user_prompt": build_prompt_q121(row),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            })
            save_local(records, output_path)
            # Sincroniza por lotes, al terminar y siempre antes de propagar un error.
            if index % DRIVE_SYNC_EVERY == 0 or index == len(pending) or generation_error is not None:
                upload_checkpoint(output_path)
            print(f"{index}/{len(pending)} · {row['country']} · id={row['id']} · válido={valid}")
            if generation_error is not None:
                raise RuntimeError("Falló el modelo. El progreso está guardado; corregí el error y reanudá.") from generation_error
        result = pd.DataFrame(records)
        print(f"Guardado local y en Drive: {output_path.name}")
        return result

    results = run_experiment()
    results[["empirical_id", "country", "empirical", "model", "thinking", "valid", "attempt"]]
    return (results,)


if __name__ == "__main__":
    app.run()
