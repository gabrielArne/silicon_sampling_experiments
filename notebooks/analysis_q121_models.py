# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.20",
#     "pandas>=2.2,<4",
#     "numpy>=2,<3",
#     "matplotlib>=3.9,<4",
#     "tabulate>=0.9",
#     "gdrive-fsspec",
#     "google-api-python-client>=2",
#     "google-auth>=2",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import io
    import json
    import math
    import textwrap
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    return LinearSegmentedColormap, Path, TwoSlopeNorm, io, json, math, mo, np, pd, plt, textwrap


@app.cell
def _(mo):
    mo.md("""
    # Análisis comparado de Q121

    Lee desde Google Drive los CSV producidos por los cinco notebooks, selecciona
    automáticamente la corrida más completa de cada modelo y compara sus respuestas
    con la distribución empírica de la WVS.

    El análisis replica para Q121 las métricas del repositorio de referencia:
    entropía normalizada, concentración en el punto medio, masa extrema,
    Jensen–Shannon, Wasserstein ordinal, posición media, errores por categoría e
    intervalos bootstrap. Como los archivos conservan cada perfil, agrega también
    exactitud individual y error absoluto ordinal.

    La comparación `prompt 1` vs `prompt 2` del tutorial no se ejecuta porque
    estos datos contienen una sola versión del prompt. El notebook reproduce la
    comparación de cada modelo con WVS, que sí es identificable con estos archivos.

    Al pulsar **Ejecutar análisis**, se muestran las tablas y figuras y se suben a
    `silicon_sampling/q121_analysis` los CSV, gráficos y el informe Markdown.
    """)
    return


@app.cell
def _(Path):
    DRIVE_FOLDER_ID = ""  # Opcional: ID de silicon_sampling si hay carpetas duplicadas.
    DRIVE_FOLDER_NAME = "silicon_sampling"
    ANALYSIS_FOLDER_NAME = "q121_analysis"
    INPUT_DATA_NAME = "WVS_wave7_migracion_prompts.csv"
    INPUT_DRIVE_FILE_ID = ""  # Opcional: ID del CSV original de WVS.
    RESULT_NAME_CONTAINS = "_Q121_"
    EXPECTED_MODELS = [
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3.5-4B",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]
    COUNTRIES = ["Argentina", "United States", "Uruguay"]
    N_BOOT = 2000
    RANDOM_SEED = 42
    OUTPUT_DIR = Path("data/analysis_q121")
    CREDENTIALS_PATH = Path.home() / ".config/pydata/pydata_google_credentials.json"
    return (
        ANALYSIS_FOLDER_NAME,
        COUNTRIES,
        CREDENTIALS_PATH,
        DRIVE_FOLDER_ID,
        DRIVE_FOLDER_NAME,
        EXPECTED_MODELS,
        INPUT_DATA_NAME,
        INPUT_DRIVE_FILE_ID,
        N_BOOT,
        OUTPUT_DIR,
        RANDOM_SEED,
        RESULT_NAME_CONTAINS,
    )


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Ejecutar análisis")
    run
    return (run,)


@app.cell
def _(np):
    CATEGORIES = [
        "Very good",
        "Quite good",
        "Neither good, nor bad",
        "Quite bad",
        "Rather bad",
    ]
    POSITIONS = np.arange(1, 6)
    MIDPOINT_INDEX = 2
    INVALID_LABEL = "invalid/no answer"

    MODEL_LABELS = {
        "Qwen/Qwen3-14B": "Qwen3-14B",
        "Qwen/Qwen3-32B": "Qwen3-32B",
        "Qwen/Qwen3.5-4B": "Qwen3.5-4B",
        "openai/gpt-oss-20b": "GPT-OSS-20B",
        "openai/gpt-oss-120b": "GPT-OSS-120B",
    }
    COLORS = {
        "WVS": "#2a78d6",
        "Qwen3.5-4B": "#008300",
        "Qwen3-14B": "#7b5fc7",
        "Qwen3-32B": "#b6538b",
        "GPT-OSS-20B": "#eda100",
        "GPT-OSS-120B": "#e34948",
    }
    return CATEGORIES, COLORS, INVALID_LABEL, MIDPOINT_INDEX, MODEL_LABELS, POSITIONS


@app.cell
def _(
    CREDENTIALS_PATH,
    DRIVE_FOLDER_ID,
    DRIVE_FOLDER_NAME,
    json,
    mo,
    run,
):
    mo.stop(not run.value, mo.md("Revisá la configuración y pulsá **Ejecutar análisis**."))

    from gdrive_fsspec import GoogleDriveFileSystem
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload

    if not CREDENTIALS_PATH.exists():
        _fs = GoogleDriveFileSystem(
            use_listings_cache=False,
            skip_instance_cache=True,
            auth_kwargs={"use_local_webserver": False},
        )
        mo.output.clear_console()

    _credentials_data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    _credentials = Credentials(
        token=_credentials_data.get("token"),
        refresh_token=_credentials_data.get("refresh_token"),
        token_uri=_credentials_data.get("token_uri"),
        client_id=_credentials_data.get("client_id"),
        client_secret=_credentials_data.get("client_secret"),
        scopes=_credentials_data.get("scopes", ["https://www.googleapis.com/auth/drive"]),
    )
    if not _credentials.valid:
        _credentials.refresh(Request())
    drive = build("drive", "v3", credentials=_credentials, cache_discovery=False)

    def escape_drive_query(value):
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def list_drive_files(query, fields="id,name,mimeType,modifiedTime,size"):
        files = []
        page_token = None
        while True:
            response = drive.files().list(
                q=query,
                fields=f"nextPageToken,files({fields})",
                pageSize=1000,
                pageToken=page_token,
            ).execute(num_retries=3)
            files.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return files

    def find_unique_folder(name, parent_id=None):
        query = (
            f"name='{escape_drive_query(name)}' and trashed=false "
            "and mimeType='application/vnd.google-apps.folder'"
        )
        if parent_id:
            query += f" and '{parent_id}' in parents"
        matches = list_drive_files(query, fields="id,name")
        if len(matches) > 1:
            raise ValueError(f"Hay varias carpetas llamadas {name}. Indicá su ID en la configuración.")
        return matches[0]["id"] if matches else None

    drive_folder_id = DRIVE_FOLDER_ID or find_unique_folder(DRIVE_FOLDER_NAME)
    if not drive_folder_id:
        raise FileNotFoundError(f"No se encontró la carpeta {DRIVE_FOLDER_NAME} en Drive.")
    print("Conexión con Google Drive lista.")
    return MediaIoBaseUpload, drive, drive_folder_id, escape_drive_query, find_unique_folder, list_drive_files


@app.cell
def _(
    EXPECTED_MODELS,
    RESULT_NAME_CONTAINS,
    drive,
    drive_folder_id,
    escape_drive_query,
    io,
    list_drive_files,
    pd,
):
    _query = (
        f"'{drive_folder_id}' in parents and trashed=false and mimeType='text/csv' "
        f"and name contains '{escape_drive_query(RESULT_NAME_CONTAINS)}'"
    )
    _candidates = list_drive_files(_query)
    if not _candidates:
        raise FileNotFoundError("No se encontraron CSV de Q121 en la carpeta de Drive.")

    _required = {
        "run_id", "empirical_id", "country", "empirical", "model", "valid",
        "attempt", "model_name", "thinking", "parameters_json",
    }
    _loaded = []
    for _metadata in _candidates:
        try:
            _content = drive.files().get_media(fileId=_metadata["id"]).execute(num_retries=3)
            _frame = pd.read_csv(io.BytesIO(_content), dtype=str, keep_default_na=False)
        except Exception as _error:
            print(f"Se omite {_metadata['name']}: no pudo leerse ({_error}).")
            continue
        if _frame.empty or not _required.issubset(_frame.columns):
            print(f"Se omite {_metadata['name']}: no tiene el esquema esperado.")
            continue
        _models = _frame["model_name"].drop_duplicates().tolist()
        if len(_models) != 1 or _models[0] not in EXPECTED_MODELS:
            print(f"Se omite {_metadata['name']}: modelo inesperado {_models}.")
            continue
        _frame["valid"] = _frame["valid"].str.lower().eq("true")
        _frame["attempt"] = pd.to_numeric(_frame["attempt"], errors="coerce")
        _frame = _frame.dropna(subset=["attempt"]).copy()
        _frame["attempt"] = _frame["attempt"].astype(int)
        _keys = ["country", "empirical_id"]
        _valid_profiles = _frame.loc[_frame["valid"]].drop_duplicates(_keys).shape[0]
        _all_profiles = _frame.drop_duplicates(_keys).shape[0]
        _loaded.append({
            "metadata": _metadata,
            "data": _frame,
            "model_name": _models[0],
            "valid_profiles": _valid_profiles,
            "all_profiles": _all_profiles,
        })

    if not _loaded:
        raise ValueError("Los archivos encontrados no tienen el esquema de los notebooks Q121.")

    # Si hay varias configuraciones para un modelo, toma la de mayor cobertura;
    # ante empate, la más reciente. Nunca mezcla run_id diferentes.
    selected_runs = {}
    selection_rows = []
    for _model_name in EXPECTED_MODELS:
        _choices = [item for item in _loaded if item["model_name"] == _model_name]
        if not _choices:
            continue
        _chosen = max(
            _choices,
            key=lambda item: (
                item["valid_profiles"], item["all_profiles"], item["metadata"].get("modifiedTime", "")
            ),
        )
        _run_counts = _chosen["data"].groupby("run_id").size().sort_values(ascending=False)
        _run_id = _run_counts.index[0]
        _chosen_frame = _chosen["data"].loc[_chosen["data"]["run_id"] == _run_id].copy()
        selected_runs[_model_name] = _chosen_frame
        selection_rows.append({
            "model_name": _model_name,
            "drive_file": _chosen["metadata"]["name"],
            "drive_file_id": _chosen["metadata"]["id"],
            "modified_time": _chosen["metadata"].get("modifiedTime", ""),
            "run_id": _run_id,
            "profiles_total": _chosen_frame.drop_duplicates(["country", "empirical_id"]).shape[0],
            "profiles_with_valid_answer": _chosen_frame.loc[_chosen_frame["valid"]].drop_duplicates(["country", "empirical_id"]).shape[0],
            "attempts": len(_chosen_frame),
        })

    missing_models = [model for model in EXPECTED_MODELS if model not in selected_runs]
    if missing_models:
        print("Advertencia: no se encontraron resultados para " + ", ".join(missing_models))
    source_selection = pd.DataFrame(selection_rows)
    if source_selection.empty:
        raise ValueError("No hay corridas utilizables para analizar.")
    source_selection
    return missing_models, selected_runs, source_selection


@app.cell
def _(CATEGORIES, INVALID_LABEL, MODEL_LABELS, pd, selected_runs):
    def choose_profile_result(group):
        """Última respuesta válida; si no existe, el último intento fallido."""
        valid = group.loc[group["valid"]].sort_values("attempt")
        return valid.iloc[-1] if not valid.empty else group.sort_values("attempt").iloc[-1]

    _final_frames = []
    for _model_name, _attempts in selected_runs.items():
        _final = (
            _attempts.groupby(["country", "empirical_id"], sort=False, group_keys=False)
            .apply(choose_profile_result, include_groups=False)
            .reset_index()
        )
        _final["model_name"] = _model_name
        _final["model_label"] = MODEL_LABELS[_model_name]
        _final["answer_for_distribution"] = _final["model"].where(
            _final["valid"] & _final["model"].isin(CATEGORIES), INVALID_LABEL
        )
        _final_frames.append(_final)

    respondent_results = pd.concat(_final_frames, ignore_index=True)
    if respondent_results.duplicated(["model_name", "country", "empirical_id"]).any():
        raise AssertionError("Quedaron perfiles duplicados después de seleccionar el intento final.")
    respondent_results
    return choose_profile_result, respondent_results


@app.cell
def _(
    CATEGORIES,
    COUNTRIES,
    INPUT_DATA_NAME,
    INPUT_DRIVE_FILE_ID,
    drive,
    drive_folder_id,
    escape_drive_query,
    io,
    list_drive_files,
    pd,
):
    if INPUT_DRIVE_FILE_ID:
        _input_id = INPUT_DRIVE_FILE_ID
    else:
        _query = (
            f"'{drive_folder_id}' in parents and trashed=false "
            f"and name='{escape_drive_query(INPUT_DATA_NAME)}'"
        )
        _matches = list_drive_files(_query, fields="id,name,modifiedTime")
        if len(_matches) != 1:
            raise FileNotFoundError(
                f"Se esperaba un único {INPUT_DATA_NAME} en Drive; se encontraron {len(_matches)}. "
                "Completá INPUT_DRIVE_FILE_ID."
            )
        _input_id = _matches[0]["id"]
    _input_content = drive.files().get_media(fileId=_input_id).execute(num_retries=3)
    wvs_raw = pd.read_csv(io.BytesIO(_input_content), dtype=str, keep_default_na=False)
    _needed = {"B_COUNTRY_LABEL", "Q121"}
    if not _needed.issubset(wvs_raw.columns):
        raise ValueError(f"El archivo WVS no contiene las columnas {_needed}.")
    wvs = wvs_raw.loc[
        wvs_raw["B_COUNTRY_LABEL"].isin(COUNTRIES) & wvs_raw["Q121"].isin(CATEGORIES),
        ["B_COUNTRY_LABEL", "Q121"],
    ].rename(columns={"B_COUNTRY_LABEL": "country", "Q121": "answer"})
    if wvs.empty:
        raise ValueError("No hay respuestas empíricas válidas de Q121 para los países configurados.")
    wvs.groupby("country").size().rename("N WVS").to_frame()
    return wvs, wvs_raw


@app.cell
def _(CATEGORIES, INVALID_LABEL, pd, respondent_results, wvs):
    def aggregate_answers(frame, answer_column, model_label):
        counts = (
            frame.groupby("country")[answer_column]
            .value_counts(dropna=False)
            .rename("n")
            .reset_index()
            .rename(columns={answer_column: "category"})
        )
        counts["proportion_all"] = counts["n"] / counts.groupby("country")["n"].transform("sum")
        counts.insert(0, "model", model_label)
        return counts

    _distribution_frames = [aggregate_answers(wvs, "answer", "WVS")]
    for _model_label, _group in respondent_results.groupby("model_label", sort=False):
        _distribution_frames.append(aggregate_answers(_group, "answer_for_distribution", _model_label))
    distributions_all = pd.concat(_distribution_frames, ignore_index=True)

    _valid = distributions_all.loc[distributions_all["category"].isin(CATEGORIES)].copy()
    _valid["proportion"] = _valid["n"] / _valid.groupby(["model", "country"])["n"].transform("sum")
    distributions_valid = _valid
    invalid_summary = (
        distributions_all.loc[distributions_all["category"] == INVALID_LABEL]
        .rename(columns={"n": "invalid_n", "proportion_all": "invalid_rate"})
        [["model", "country", "invalid_n", "invalid_rate"]]
    )
    return aggregate_answers, distributions_all, distributions_valid, invalid_summary


@app.cell
def _(CATEGORIES, MIDPOINT_INDEX, POSITIONS, math, np):
    def entropy_norm(probabilities):
        positive = probabilities[probabilities > 0]
        return float(-(positive * np.log2(positive)).sum() / np.log2(len(CATEGORIES)))

    def jsd(first, second):
        midpoint = 0.5 * (first + second)
        def kl(left, right):
            mask = left > 0
            return float((left[mask] * np.log2(left[mask] / right[mask])).sum())
        return 0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint)

    def wasserstein_ordinal(first, second):
        return float(np.abs(np.cumsum(first) - np.cumsum(second)).sum())

    def distribution_metrics(probabilities, empirical):
        entropy = entropy_norm(probabilities)
        empirical_entropy = entropy_norm(empirical)
        metrics = {
            "H_norm": entropy,
            "H_norm_wvs": empirical_entropy,
            "entropy_ratio": entropy / empirical_entropy if empirical_entropy else math.nan,
            "midpoint_mass": float(probabilities[MIDPOINT_INDEX]),
            "midpoint_mass_wvs": float(empirical[MIDPOINT_INDEX]),
            "d_midpoint_mass": float(probabilities[MIDPOINT_INDEX] - empirical[MIDPOINT_INDEX]),
            "extreme_mass": float(probabilities[0] + probabilities[-1]),
            "extreme_mass_wvs": float(empirical[0] + empirical[-1]),
            "JSD": jsd(probabilities, empirical),
            "W1": wasserstein_ordinal(probabilities, empirical),
            "mean_position": float((POSITIONS * probabilities).sum()),
            "mean_position_wvs": float((POSITIONS * empirical).sum()),
            "d_mean_position": float((POSITIONS * (probabilities - empirical)).sum()),
        }
        for index in range(len(CATEGORIES)):
            metrics[f"err_{index + 1}"] = float(probabilities[index] - empirical[index])
        return metrics

    return distribution_metrics, entropy_norm, jsd, wasserstein_ordinal


@app.cell
def _(
    CATEGORIES,
    N_BOOT,
    RANDOM_SEED,
    distribution_metrics,
    entropy_norm,
    invalid_summary,
    jsd,
    np,
    pd,
    respondent_results,
    wasserstein_ordinal,
    wvs,
):
    _rng = np.random.default_rng(RANDOM_SEED)

    def ordered_counts(frame, column):
        return frame[column].value_counts().reindex(CATEGORIES, fill_value=0).to_numpy(dtype=int)

    def bootstrap_cis(model_counts, wvs_counts):
        model_n, wvs_n = int(model_counts.sum()), int(wvs_counts.sum())
        model_p, wvs_p = model_counts / model_n, wvs_counts / wvs_n
        model_boot = _rng.multinomial(model_n, model_p, size=N_BOOT) / model_n
        wvs_boot = _rng.multinomial(wvs_n, wvs_p, size=N_BOOT) / wvs_n
        samples = {
            "midpoint_mass": model_boot[:, 2],
            "H_norm": np.array([entropy_norm(row) for row in model_boot]),
            "JSD": np.array([jsd(model_boot[i], wvs_boot[i]) for i in range(N_BOOT)]),
            "W1": np.array([wasserstein_ordinal(model_boot[i], wvs_boot[i]) for i in range(N_BOOT)]),
        }
        intervals = {}
        for name, values in samples.items():
            low, high = np.percentile(values, [2.5, 97.5])
            intervals[f"{name}_lo"] = float(low)
            intervals[f"{name}_hi"] = float(high)
        return intervals

    _rows = []
    for (_model_label, _country), _group in respondent_results.groupby(["model_label", "country"]):
        _valid = _group.loc[_group["valid"] & _group["model"].isin(CATEGORIES)].copy()
        _empirical = wvs.loc[wvs["country"] == _country]
        if _valid.empty or _empirical.empty:
            continue
        _model_counts = ordered_counts(_valid, "model")
        _wvs_counts = ordered_counts(_empirical, "answer")
        _model_p = _model_counts / _model_counts.sum()
        _wvs_p = _wvs_counts / _wvs_counts.sum()
        _ordinal = {category: index + 1 for index, category in enumerate(CATEGORIES)}
        _true_position = _valid["empirical"].map(_ordinal)
        _model_position = _valid["model"].map(_ordinal)
        _paired = _true_position.notna() & _model_position.notna()
        _row = {
            "model": _model_label,
            "country": _country,
            "N_profiles": len(_group),
            "N_valid": len(_valid),
            "valid_rate": float(len(_valid) / len(_group)),
            "exact_match_rate": float((_valid.loc[_paired, "empirical"] == _valid.loc[_paired, "model"]).mean()),
            "ordinal_MAE": float(np.abs(_true_position[_paired] - _model_position[_paired]).mean()),
            **distribution_metrics(_model_p, _wvs_p),
            **bootstrap_cis(_model_counts, _wvs_counts),
        }
        _rows.append(_row)

    metrics = pd.DataFrame(_rows).sort_values(["model", "country"], ignore_index=True)
    metrics = metrics.merge(invalid_summary, on=["model", "country"], how="left")
    metrics[["invalid_n", "invalid_rate"]] = metrics[["invalid_n", "invalid_rate"]].fillna(0)
    if metrics.empty:
        raise ValueError("No hay respuestas válidas suficientes para calcular métricas.")
    metrics
    return bootstrap_cis, metrics, ordered_counts


@app.cell
def _(CATEGORIES, COLORS, COUNTRIES, distributions_valid, metrics, np, plt, textwrap):
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e1e0d9",
        "axes.axisbelow": True,
        "legend.frameon": False,
    })
    _models = metrics["model"].drop_duplicates().tolist()
    _series = ["WVS"] + _models
    _x = np.arange(len(CATEGORIES))
    _width = 0.82 / len(_series)
    fig_distributions, _axes = plt.subplots(1, len(COUNTRIES), figsize=(17, 5), sharey=True)
    for _country_index, _country in enumerate(COUNTRIES):
        _axis = _axes[_country_index]
        for _series_index, _name in enumerate(_series):
            _probabilities = (
                distributions_valid.loc[
                    (distributions_valid["country"] == _country) & (distributions_valid["model"] == _name)
                ].set_index("category")["proportion"].reindex(CATEGORIES, fill_value=0).to_numpy()
            )
            _offset = (_series_index - (len(_series) - 1) / 2) * _width
            _axis.bar(_x + _offset, _probabilities, _width * 0.92, color=COLORS[_name], label=_name)
        _axis.set_title(_country)
        _axis.set_xticks(_x, ["\n".join(textwrap.wrap(value, 15)) for value in CATEGORIES], fontsize=7)
        _axis.set_ylim(0, 1)
        _axis.grid(axis="x", visible=False)
        if _country_index == 0:
            _axis.set_ylabel("Proporción entre respuestas válidas")
    fig_distributions.legend(loc="upper center", ncol=len(_series), bbox_to_anchor=(0.5, 1.04))
    fig_distributions.suptitle("Q121: distribución WVS y respuestas simuladas", y=1.11, fontsize=13)
    fig_distributions.tight_layout()
    fig_distributions
    return fig_distributions


@app.cell
def _(COLORS, COUNTRIES, metrics, np, plt):
    _models = metrics["model"].drop_duplicates().tolist()
    fig_midpoint, _axes = plt.subplots(1, len(COUNTRIES), figsize=(14, 4), sharex=True)
    for _country_index, _country in enumerate(COUNTRIES):
        _axis = _axes[_country_index]
        _subset = metrics.loc[metrics["country"] == _country].set_index("model").reindex(_models)
        _y = np.arange(len(_models))[::-1]
        _wvs_value = _subset["midpoint_mass_wvs"].dropna().iloc[0]
        _axis.axvline(_wvs_value, color=COLORS["WVS"], linestyle="--", linewidth=1.5, label="WVS")
        for _position, _model in zip(_y, _models):
            _row = _subset.loc[_model]
            _axis.hlines(_position, _wvs_value, _row["midpoint_mass"], color="#c3c2b7", linewidth=2)
            _axis.scatter(_row["midpoint_mass"], _position, color=COLORS[_model], s=60, zorder=3)
        _axis.set_yticks(_y, _models if _country_index == 0 else [""] * len(_models))
        _axis.set_title(_country)
        _axis.set_xlabel("Masa en la opción neutral")
        _axis.grid(axis="y", visible=False)
    fig_midpoint.suptitle("Concentración en “Neither good, nor bad” frente a WVS", fontsize=13)
    fig_midpoint.tight_layout()
    fig_midpoint
    return fig_midpoint


@app.cell
def _(COLORS, COUNTRIES, metrics, plt):
    _markers = {country: marker for country, marker in zip(COUNTRIES, ["o", "s", "^"])}
    fig_fidelity, _axis = plt.subplots(figsize=(7, 5.5))
    for _, _row in metrics.iterrows():
        _axis.scatter(
            _row["entropy_ratio"],
            _row["JSD"],
            color=COLORS[_row["model"]],
            marker=_markers[_row["country"]],
            s=90,
            edgecolor="white",
            linewidth=1.5,
        )
    _axis.axvline(1, color="#898781", linestyle="--", linewidth=1)
    _axis.set_xlabel("Razón de entropía modelo / WVS")
    _axis.set_ylabel("Divergencia Jensen–Shannon respecto de WVS")
    _axis.set_title("Dispersión y fidelidad distributiva")
    _model_handles = [
        plt.Line2D([], [], color=COLORS[model], marker="o", linestyle="", label=model)
        for model in metrics["model"].drop_duplicates()
    ]
    _country_handles = [
        plt.Line2D([], [], color="#777777", marker=_markers[country], linestyle="", label=country)
        for country in COUNTRIES
    ]
    _axis.legend(handles=_model_handles + _country_handles, fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig_fidelity.tight_layout()
    fig_fidelity
    return fig_fidelity


@app.cell
def _(CATEGORIES, COUNTRIES, LinearSegmentedColormap, TwoSlopeNorm, metrics, np, plt, textwrap):
    _models = metrics["model"].drop_duplicates().tolist()
    _error_columns = [f"err_{index}" for index in range(1, len(CATEGORIES) + 1)]
    _maximum = float(metrics[_error_columns].abs().max().max()) or 1.0
    _normalization = TwoSlopeNorm(vmin=-_maximum, vcenter=0, vmax=_maximum)
    _cmap = LinearSegmentedColormap.from_list("blue-gray-red", ["#2a78d6", "#f0efec", "#e34948"])
    fig_errors, _axes = plt.subplots(len(COUNTRIES), 1, figsize=(11, 9), sharex=True)
    for _country_index, _country in enumerate(COUNTRIES):
        _axis = _axes[_country_index]
        _data = (
            metrics.loc[metrics["country"] == _country]
            .set_index("model").reindex(_models)[_error_columns].to_numpy()
        )
        _axis.imshow(_data, cmap=_cmap, norm=_normalization, aspect="auto")
        for _row_index in range(_data.shape[0]):
            for _column_index in range(_data.shape[1]):
                _value = _data[_row_index, _column_index]
                _axis.text(_column_index, _row_index, f"{_value:+.2f}", ha="center", va="center", fontsize=8)
        _axis.set_title(_country)
        _axis.set_yticks(range(len(_models)), _models)
        _axis.grid(visible=False)
    _axes[-1].set_xticks(
        range(len(CATEGORIES)),
        ["\n".join(textwrap.wrap(value, 18)) for value in CATEGORIES],
        fontsize=8,
    )
    fig_errors.colorbar(plt.cm.ScalarMappable(norm=_normalization, cmap=_cmap), ax=_axes, shrink=0.7, label="p(modelo) − p(WVS)")
    fig_errors.suptitle("Error firmado por categoría", fontsize=13)
    fig_errors
    return fig_errors


@app.cell
def _(metrics, mo):
    mo.md("## Métricas por modelo y país")
    metrics[
        [
            "model", "country", "N_profiles", "N_valid", "valid_rate",
            "exact_match_rate", "ordinal_MAE", "H_norm", "entropy_ratio",
            "midpoint_mass", "midpoint_mass_wvs", "d_midpoint_mass",
            "extreme_mass", "JSD", "W1", "mean_position", "d_mean_position",
        ]
    ].round(4)
    return


@app.cell
def _(metrics, pd):
    model_summary = (
        metrics.groupby("model", as_index=False)
        .agg(
            countries=("country", "nunique"),
            profiles=("N_profiles", "sum"),
            valid_rate=("valid_rate", "mean"),
            exact_match_rate=("exact_match_rate", "mean"),
            ordinal_MAE=("ordinal_MAE", "mean"),
            entropy_ratio=("entropy_ratio", "mean"),
            d_midpoint_mass=("d_midpoint_mass", "mean"),
            JSD=("JSD", "mean"),
            W1=("W1", "mean"),
        )
        .sort_values(["JSD", "ordinal_MAE"], ignore_index=True)
    )
    model_summary.round(4)
    return (model_summary,)


@app.cell
def _(
    CATEGORIES,
    OUTPUT_DIR,
    distributions_all,
    fig_distributions,
    fig_errors,
    fig_fidelity,
    fig_midpoint,
    invalid_summary,
    metrics,
    model_summary,
    source_selection,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_files = {}

    def save_csv(frame, name):
        path = OUTPUT_DIR / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        output_files[name] = path

    save_csv(source_selection, "source_selection.csv")
    save_csv(distributions_all, "q121_distributions.csv")
    save_csv(metrics, "q121_metrics.csv")
    save_csv(model_summary, "q121_model_summary.csv")
    save_csv(invalid_summary, "q121_invalid_responses.csv")

    _figures = {
        "fig1_q121_distributions.png": fig_distributions,
        "fig2_q121_midpoint_mass.png": fig_midpoint,
        "fig3_q121_entropy_jsd.png": fig_fidelity,
        "fig4_q121_signed_errors.png": fig_errors,
    }
    for _name, _figure in _figures.items():
        _path = OUTPUT_DIR / _name
        _figure.savefig(_path, dpi=300, bbox_inches="tight")
        output_files[_name] = _path

    _best_jsd = model_summary.sort_values("JSD").iloc[0]
    _best_match = model_summary.sort_values("exact_match_rate", ascending=False).iloc[0]
    _compression_cells = int((metrics["entropy_ratio"] < 1).sum())
    _midpoint_excess_cells = int((metrics["d_midpoint_mass"] > 0).sum())
    _cell_count = len(metrics)
    _report = f"""# Análisis comparado de Q121

## Datos

- Modelos analizados: {metrics['model'].nunique()}.
- Países: {', '.join(metrics['country'].drop_duplicates())}.
- Celdas modelo × país: {_cell_count}.
- Categorías ordinales: {' | '.join(CATEGORIES)}.
- Solo la última respuesta válida por perfil entra al análisis; si un perfil no
  tiene ninguna respuesta válida, se conserva como `invalid/no answer`.

## Resultados principales

- {_compression_cells}/{_cell_count} celdas tienen menor entropía que la WVS
  (`entropy_ratio < 1`), señal de compresión distributiva.
- {_midpoint_excess_cells}/{_cell_count} celdas concentran más respuestas en
  “Neither good, nor bad” que la WVS (`d_midpoint_mass > 0`).
- Menor JSD promedio: **{_best_jsd['model']}** ({_best_jsd['JSD']:.3f}).
- Mayor coincidencia individual promedio: **{_best_match['model']}**
  ({_best_match['exact_match_rate']:.1%}).

## Resumen por modelo

{model_summary.round(4).to_markdown(index=False)}

## Métricas por modelo y país

{metrics.round(4).to_markdown(index=False)}

## Interpretación

- `H_norm` mide diversidad de respuestas; `entropy_ratio < 1` indica una
  distribución más concentrada que la WVS.
- `d_midpoint_mass > 0` indica exceso de respuestas neutrales.
- `JSD` y `W1` miden distancia respecto de la WVS; valores menores son mejores.
- `exact_match_rate` y `ordinal_MAE` comparan cada respuesta simulada con la
  respuesta empírica del mismo perfil.
- Los intervalos del 95% provienen de bootstrap multinomial con remuestreo de
  ambas distribuciones.

## Limitaciones

- El análisis es descriptivo para una pregunta y tres países.
- La coincidencia individual es una exigencia más fuerte que reproducir una
  distribución poblacional: un modelo puede aproximar la distribución y fallar
  en las respuestas persona por persona, o viceversa.
- Las corridas pueden usar hardware y parámetros distintos. `source_selection.csv`
  identifica exactamente el archivo y `run_id` elegido para cada modelo.
"""
    _report_path = OUTPUT_DIR / "q121_analysis_report.md"
    _report_path.write_text(_report, encoding="utf-8")
    output_files[_report_path.name] = _report_path
    print(f"Se generaron {len(output_files)} archivos en {OUTPUT_DIR}.")
    return output_files


@app.cell
def _(
    ANALYSIS_FOLDER_NAME,
    MediaIoBaseUpload,
    drive,
    drive_folder_id,
    escape_drive_query,
    find_unique_folder,
    io,
    list_drive_files,
    output_files,
):
    analysis_folder_id = find_unique_folder(ANALYSIS_FOLDER_NAME, drive_folder_id)
    if not analysis_folder_id:
        analysis_folder_id = drive.files().create(
            body={
                "name": ANALYSIS_FOLDER_NAME,
                "parents": [drive_folder_id],
                "mimeType": "application/vnd.google-apps.folder",
            },
            fields="id",
        ).execute(num_retries=3)["id"]

    uploaded_outputs = []
    for _name, _path in output_files.items():
        _query = (
            f"'{analysis_folder_id}' in parents and trashed=false "
            f"and name='{escape_drive_query(_name)}'"
        )
        _matches = list_drive_files(_query, fields="id,name")
        if len(_matches) > 1:
            raise ValueError(f"Hay varios archivos de salida llamados {_name} en Drive.")
        _mime = "image/png" if _path.suffix == ".png" else "text/csv" if _path.suffix == ".csv" else "text/markdown"
        _media = MediaIoBaseUpload(io.BytesIO(_path.read_bytes()), mimetype=_mime, resumable=True)
        if _matches:
            _file_id = _matches[0]["id"]
            drive.files().update(fileId=_file_id, media_body=_media).execute(num_retries=3)
        else:
            _file_id = drive.files().create(
                body={"name": _name, "parents": [analysis_folder_id]},
                media_body=_media,
                fields="id",
            ).execute(num_retries=3)["id"]
        uploaded_outputs.append({"name": _name, "drive_file_id": _file_id})
    print(f"Resultados subidos a {ANALYSIS_FOLDER_NAME}: {len(uploaded_outputs)} archivos.")
    uploaded_outputs
    return analysis_folder_id, uploaded_outputs


if __name__ == "__main__":
    app.run()
