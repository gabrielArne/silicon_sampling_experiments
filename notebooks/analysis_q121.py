# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["marimo>=0.25", "pandas>=2.2,<4", "numpy>=2", "scipy>=1.14", "matplotlib>=3.9"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats
    from scipy.spatial.distance import jensenshannon
    import marimo as mo
    return Path, hashlib, json, np, pd, plt, stats, jensenshannon, mo


@app.cell
def _(mo):
    mo.md("""
    # Q121: respuestas humanas y simuladas

    Este notebook audita los CSV, crea `dataset_respuestas_modelos.csv` y reproduce
    el análisis de distribuciones del tutorial Q130, adaptado a las **cinco opciones
    de Q121**. Añade exactitud, matrices de confusión y error ordinal individual.

    **En Molab:** subí el dataset unificado al directorio de trabajo (o los 13 CSV
    originales a `data/output`) y ejecutá todas las celdas. No requiere API key,
    GPU ni conexión con Drive. Las tablas, figuras y el informe se guardan en
    `data/analysis_q121`. El análisis usa exclusivamente las respuestas ya generadas.

    La referencia principal son las respuestas WVS de los **mismos perfiles**
    evaluados. No se aplican ponderadores muestrales: las conclusiones describen
    esta muestra, no estimaciones representativas de la población nacional.
    """)
    return


@app.cell
def _(Path):
    # Editá estas rutas si colocás los archivos en otra carpeta de Molab.
    INPUT_DIR = Path("data/output")
    UNIFIED_NAME = "dataset_respuestas_modelos.csv"
    OUTPUT_DIR = Path("data/analysis_q121")
    N_BOOT = 2000
    RANDOM_SEED = 42
    OPTIONS = ["Very good", "Quite good", "Neither good, nor bad", "Quite bad", "Rather bad"]
    SHORT_OPTIONS = ["Muy buena", "Buena", "Neutral", "Mala", "Muy mala"]
    PROMPTS = ["neutral_v1", "anti_midpoint_v2"]
    # Corrida parcial con thinking=True y 4096 tokens, incompatible con la completa.
    EXCLUDED_RUNS = {"c8cc559dc6ee47fa": "Corrida parcial: thinking=True, max_new_tokens=4096"}
    return INPUT_DIR, UNIFIED_NAME, OUTPUT_DIR, N_BOOT, RANDOM_SEED, OPTIONS, SHORT_OPTIONS, PROMPTS, EXCLUDED_RUNS


@app.cell
def _(INPUT_DIR, UNIFIED_NAME, OUTPUT_DIR, Path, pd, json, hashlib, EXCLUDED_RUNS, OPTIONS, PROMPTS):
    _folders = [INPUT_DIR, Path("..") / INPUT_DIR, Path("output"), Path(".")]
    _folder = next((p for p in _folders if list(p.glob("*_Q121_*.csv"))), None)
    _frames = []
    if _folder is not None:
        for _path in sorted(_folder.glob("*_Q121_*.csv")):
            _frame = pd.read_csv(_path, dtype=str, keep_default_na=False)
            _frame["source_file"] = _path.name
            _frame["source_sha256"] = hashlib.sha256(_path.read_bytes()).hexdigest()
            _frames.append(_frame)
        dataset = pd.concat(_frames, ignore_index=True).fillna("")
        unified_path = _folder / UNIFIED_NAME
    else:
        _path = next((p / UNIFIED_NAME for p in _folders if (p / UNIFIED_NAME).is_file()), None)
        if _path is None:
            raise FileNotFoundError(f"Subí {UNIFIED_NAME} o los CSV originales a data/output.")
        dataset = pd.read_csv(_path, dtype=str, keep_default_na=False)
        unified_path = _path

    _required = {"run_id", "empirical_id", "country", "model_name", "system_prompt", "empirical", "model", "valid", "parameters_json", "attempt"}
    if not _required.issubset(dataset.columns):
        raise ValueError(f"Faltan columnas: {_required - set(dataset.columns)}")
    dataset["prompt_version_original"] = dataset.get("prompt_version_original", dataset.get("prompt_version", ""))

    def identify_prompt(text):
        if "Do NOT filter responses for political correctness" in text and "middle option" in text:
            return "anti_midpoint_v2"
        if "Select the single response option this person would most likely choose" in text:
            return "neutral_v1"
        raise ValueError("Prompt desconocido: revisar el texto antes de clasificarlo.")

    dataset["prompt_version"] = dataset.system_prompt.map(identify_prompt)
    _declared = dataset.prompt_version_original.ne("")
    if not dataset.loc[_declared, "prompt_version_original"].eq(dataset.loc[_declared, "prompt_version"]).all():
        raise ValueError("El nombre del prompt contradice su texto.")
    dataset["valid_reported"] = dataset.get("valid_reported", dataset.valid).astype(str).str.lower().eq("true")
    dataset["valid"] = dataset.valid_reported & dataset.model.isin(OPTIONS) & dataset.empirical.isin(OPTIONS)
    dataset["attempt"] = pd.to_numeric(dataset.attempt, errors="raise").astype(int)
    dataset["excluded_reason"] = dataset.run_id.map(EXCLUDED_RUNS).fillna("")
    dataset["included_run"] = dataset.excluded_reason.eq("")
    _keys = ["run_id", "country", "empirical_id", "attempt"]
    if dataset.duplicated(_keys).any():
        raise ValueError("Hay intentos repetidos: revisar archivos duplicados antes de unificar.")
    _runs = dataset.loc[dataset.included_run].groupby(["model_name", "prompt_version"]).run_id.nunique()
    if (_runs != 1).any():
        raise ValueError("Hay más de una configuración para un modelo/prompt; seleccioná explícitamente una corrida.")
    if set(dataset.prompt_version) != set(PROMPTS):
        raise ValueError("Se necesitan ambos prompts para este análisis.")
    # Un resultado válido por perfil; conservar todos los intentos en el dataset.
    dataset["selected_response"] = False
    _eligible = dataset.loc[dataset.included_run & dataset.valid].sort_values("attempt")
    _selected = _eligible.drop_duplicates(["model_name", "prompt_version", "country", "empirical_id"], keep="last").index
    dataset.loc[_selected, "selected_response"] = True
    _scale = {option: index + 1 for index, option in enumerate(OPTIONS)}
    dataset["empirical_ordinal"] = dataset.empirical.map(_scale).astype("Int64")
    dataset["model_ordinal"] = dataset.model.map(_scale).where(dataset.valid).astype("Int64")
    dataset["signed_ordinal_error"] = dataset.model_ordinal - dataset.empirical_ordinal
    dataset["absolute_ordinal_error"] = dataset.signed_ordinal_error.abs()
    dataset["exact_match"] = dataset.empirical.eq(dataset.model) & dataset.valid
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _audit = []
    for _file, _g in dataset.groupby("source_file", sort=True):
        _p = json.loads(_g.parameters_json.iloc[0])
        _audit.append({"file": _file, "run_id": _g.run_id.iloc[0], "model": _g.model_name.iloc[0],
            "prompt": _g.prompt_version.iloc[0], "attempts": len(_g), "valid": int(_g.valid.sum()),
            "invalid": int((~_g.valid).sum()), "profiles": len(_g.drop_duplicates(["country", "empirical_id"])),
            "thinking": _p.get("chat_template", {}).get("enable_thinking", "API/Harmony"),
            "max_tokens": _p["generation"].get("max_new_tokens", _p["generation"].get("max_output_tokens")),
            "transformers": _p.get("transformers_version", ""), "included": bool(_g.included_run.iloc[0]),
            "reason": _g.excluded_reason.iloc[0]})
    audit = pd.DataFrame(_audit)
    audit.to_csv(OUTPUT_DIR / "auditoria_archivos.csv", index=False)
    # La corrida parcial queda solo en la auditoría, nunca en el dataset unificado.
    dataset = dataset.loc[dataset.included_run].copy()
    dataset.to_csv(unified_path, index=False, encoding="utf-8-sig")
    selected = dataset.loc[dataset.selected_response].copy()
    main = dataset.loc[dataset.included_run].copy()
    # La identidad y la respuesta empírica deben mantenerse entre modelos y prompts.
    for _column in ["empirical", "sex", "education", "class", "occupation", "political_ident"]:
        if (main.groupby(["country", "empirical_id"])[_column].nunique() > 1).any():
            raise ValueError(f"Perfiles inconsistentes en {_column}.")
    reference = main.drop_duplicates(["country", "empirical_id"])
    _profile_sets = [set(zip(g.country, g.empirical_id)) for _, g in main.groupby(["model_name", "prompt_version"])]
    if any(s != _profile_sets[0] for s in _profile_sets[1:]):
        raise ValueError("Las corridas no usan los mismos perfiles intentados. Revisá la selección antes de comparar.")
    return dataset, unified_path, audit, selected, main, reference


@app.cell
def _(mo, audit, dataset, main, selected, reference):
    mo.vstack([
        mo.md(f"""## Auditoría de los archivos

        Se auditaron **{len(audit)} archivos**, con **{int(audit.attempts.sum()):,} intentos**.
        Se incluyen **{int(audit.included.sum())} corridas** en el análisis principal:
        **{len(main):,} intentos**, **{len(selected):,} respuestas válidas únicas** y
        **{len(reference):,} perfiles humanos**.

        El archivo adicional es la corrida `c8cc559dc6ee47fa` de Qwen3.5-4B.
        Tiene 21 perfiles de Argentina, 12 respuestas válidas y 9 inválidas;
        activa thinking y permite 4096 tokens. La corrida completa desactiva thinking
        y permite 512 tokens. Se excluye del dataset y del análisis; solo figura en la auditoría.
        No se reemplazan respuestas de una configuración con las de otra.

        Los CSV antiguos no nombran el prompt: se recupera su versión a partir del
        **texto de system_prompt** y se comprueba contra la etiqueta cuando existe.
        `selected_response=True` identifica las filas que forman las distribuciones.
        """),
        mo.ui.table(audit, selection=None),
    ])
    return


@app.cell
def _(np, pd, stats, jensenshannon, OPTIONS, N_BOOT):
    def probabilities(values):
        counts = values.value_counts().reindex(OPTIONS, fill_value=0).to_numpy(dtype=float)
        if counts.sum() == 0:
            raise ValueError("Distribución vacía.")
        return counts / counts.sum()

    def distribution_metrics(p, q):
        def entropy(x):
            positive = x[x > 0]
            return float(-(positive * np.log2(positive)).sum() / np.log2(5))
        h, hq = entropy(p), entropy(q)
        mu, muq = p @ np.arange(1, 6), q @ np.arange(1, 6)
        return {"H_norm": h, "H_norm_wvs": hq, "entropy_ratio": h / hq if hq else np.nan,
                "neutral_mass": p[2], "neutral_wvs": q[2], "neutral_excess": p[2] - q[2],
                "IM": p[1:4].sum(), "IM_wvs": q[1:4].sum(), "dIM": p[1:4].sum() - q[1:4].sum(),
                "EM": p[0] + p[4], "EM_wvs": q[0] + q[4],
                "JSD": float(jensenshannon(p, q, base=2) ** 2),
                "W1": float(np.abs(np.cumsum(p)[:-1] - np.cumsum(q)[:-1]).sum()),
                "mu": mu, "dmu": mu - muq,
                **{f"err_{i+1}": p[i] - q[i] for i in range(5)}}

    def bootstrap_individual(g, rng):
        # Remuestreo de pares humano/modelo: preserva la dependencia por perfil.
        a = g.empirical_ordinal.to_numpy(dtype=int) - 1
        b = g.model_ordinal.to_numpy(dtype=int) - 1
        joint = np.bincount(a * 5 + b, minlength=25) / len(g)
        draws = rng.multinomial(len(g), joint, size=N_BOOT).reshape(-1, 5, 5) / len(g)
        estimates = []
        distance = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
        for draw in draws:
            m = distribution_metrics(draw.sum(axis=0), draw.sum(axis=1))
            m.update(accuracy=np.trace(draw), MAE=float((draw * distance).sum()))
            estimates.append(m)
        frame = pd.DataFrame(estimates)
        return {f"{name}_{bound}": float(frame[name].quantile(quantile))
                for name in ["H_norm", "neutral_mass", "IM", "JSD", "accuracy", "MAE"]
                for bound, quantile in [("lo", .025), ("hi", .975)]}

    def paired_test(values, alternative):
        d = np.asarray(values, dtype=float)
        nz = d[~np.isclose(d, 0, atol=1e-12)]
        if len(nz) == 0:
            return {"n": len(d), "n_nonzero": 0, "wilcoxon_p": 1., "rank_biserial": 0., "sign_p": 1., "n_favor": 0}
        ranks = stats.rankdata(np.abs(nz))
        favor = int((nz < 0).sum() if alternative == "less" else (nz > 0).sum())
        return {"n": len(d), "n_nonzero": len(nz),
                "wilcoxon_p": float(stats.wilcoxon(nz, alternative=alternative, method="auto").pvalue),
                "rank_biserial": float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()),
                "sign_p": float(stats.binomtest(favor, len(nz), .5, alternative="greater").pvalue), "n_favor": favor}
    return probabilities, distribution_metrics, bootstrap_individual, paired_test


@app.cell
def _(main, selected, reference, probabilities, distribution_metrics, bootstrap_individual, np, pd, RANDOM_SEED, OUTPUT_DIR, OPTIONS):
    _rng = np.random.default_rng(RANDOM_SEED)
    _metrics, _distributions, _quality, _confusions = [], [], [], []
    for (_model, _prompt, _country), _all in main.groupby(["model_name", "prompt_version", "country"]):
        _g = selected.loc[(selected.model_name == _model) & (selected.prompt_version == _prompt) & (selected.country == _country)]
        _n_profiles = _all.empirical_id.nunique()
        _key = {"model_name": _model, "prompt_version": _prompt, "country": _country}
        _quality.append({**_key, "attempts": len(_all), "profiles": _n_profiles, "valid_profiles": len(_g),
                         "invalid_attempts": int((~_all.valid).sum()), "missing_valid_profiles": _n_profiles - len(_g),
                         "invalid_profile_rate": 1 - len(_g) / _n_profiles})
        if _g.empty:
            continue
        _p, _q = probabilities(_g.model), probabilities(_g.empirical)
        _q_all = probabilities(reference.loc[reference.country == _country, "empirical"])
        _m = distribution_metrics(_p, _q)
        _m.update(accuracy=float(_g.exact_match.mean()), MAE=float(_g.absolute_ordinal_error.mean()),
                  ordinal_bias=float(_g.signed_ordinal_error.mean()),
                  accuracy_all_profiles=float(_g.exact_match.sum() / _n_profiles),
                  JSD_all_profiles=distribution_metrics(_p, _q_all)["JSD"])
        _metrics.append({**_key, "N_valid": len(_g), **_m, **bootstrap_individual(_g, _rng)})
        for _i, _option in enumerate(OPTIONS):
            _distributions.append({**_key, "category": _option, "position": _i+1,
                "p_model": _p[_i], "p_wvs_matched": _q[_i], "p_wvs_all": _q_all[_i], "signed_error": _p[_i] - _q[_i]})
        _cm = pd.crosstab(_g.empirical, _g.model).reindex(index=OPTIONS, columns=OPTIONS, fill_value=0)
        for _actual in OPTIONS:
            for _predicted in OPTIONS:
                _den = int(_cm.loc[_actual].sum())
                _confusions.append({**_key, "empirical": _actual, "predicted": _predicted,
                    "count": int(_cm.loc[_actual, _predicted]),
                    "row_proportion": _cm.loc[_actual, _predicted] / _den if _den else np.nan})
    metrics = pd.DataFrame(_metrics)
    distributions = pd.DataFrame(_distributions)
    quality = pd.DataFrame(_quality)
    confusion = pd.DataFrame(_confusions)
    for _name, _frame in [("metricas", metrics), ("distribuciones", distributions), ("calidad", quality), ("matrices_confusion", confusion)]:
        _frame.to_csv(OUTPUT_DIR / f"{_name}.csv", index=False)
    return metrics, distributions, quality, confusion


@app.cell
def _(mo, metrics, quality):
    mo.vstack([
        mo.md("""## Métricas e interpretación

        Cada fila corresponde a **modelo × país × prompt**. Las proporciones se
        calculan sobre respuestas válidas. La referencia `wvs` usa esos mismos perfiles;
        `JSD_all_profiles` permite contrastar con todos los perfiles muestreados del país.

        | Métrica | Lectura |
        |---|---|
        | H_norm | Entropía / log₂(5): 0 = una sola respuesta; 1 = distribución uniforme. |
        | entropy_ratio | H(modelo)/H(WVS). Menor que 1 indica compresión. Mayor entropía no implica por sí sola mayor fidelidad. |
        | neutral_mass / neutral_excess | Proporción neutral y diferencia frente a WVS. Exceso positivo indica concentración neutral. |
        | IM / dIM | Masa de las tres categorías interiores (2,3,4) y exceso frente a WVS. |
        | EM | Masa en las dos categorías extremas (1,5). IM + EM = 1. |
        | JSD | Divergencia Jensen–Shannon, base 2, de 0 a 1. Menor es mejor. Se usa la divergencia, no su raíz. |
        | W1 | Distancia ordinal entre distribuciones, de 0 a 4. Menor es mejor. |
        | mu / dmu | Media ordinal y diferencia frente a WVS. Positiva = evaluación más negativa de la inmigración. |
        | err_1 … err_5 | Proporción simulada menos humana por categoría. Positiva = sobrerrepresentación. |
        | accuracy | Coincidencias exactas / respuestas válidas. |
        | accuracy_all_profiles | Coincidencias / todos los perfiles intentados; los fallos no cuentan como aciertos. |
        | MAE | Promedio del valor absoluto de (respuesta simulada − humana), posiciones 1–5. De 0 a 4; menor es mejor. |
        | ordinal_bias | Error individual con signo promedio; equivale a dmu en los mismos pares. |

        Las posiciones son 1=Very good … 5=Rather bad. La escala ordinal se trata
        con separación unitaria; no implica que las distancias psicológicas sean iguales.
        Los intervalos `_lo`, `_hi` son percentiles 2,5 y 97,5 de 2000 remuestreos.
        A diferencia del bootstrap multinomial independiente del original, aquí se
        remuestrea la **tabla conjunta humano/modelo**, preservando el emparejamiento.
        No capturan variabilidad de nuevas generaciones del LLM: hay una por perfil.
        """),
        mo.ui.table(metrics.round(4), selection=None),
        mo.md("""### Respuestas inválidas
        La tasa de perfiles sin respuesta válida informa la cobertura. Eliminar esos
        perfiles puede sesgar las distribuciones si el fallo depende de sus atributos.
        Los errores originales permanecen en el dataset; no se imputan respuestas."""),
        mo.ui.table(quality.round(4), selection=None),
    ])
    return


@app.cell
def _(selected, PROMPTS, probabilities, distribution_metrics, paired_test, pd, np, OUTPUT_DIR):
    _rows, _coverage = [], []
    for (_model, _country), _g in selected.groupby(["model_name", "country"]):
        _a = _g.loc[_g.prompt_version == PROMPTS[0]].set_index("empirical_id")
        _b = _g.loc[_g.prompt_version == PROMPTS[1]].set_index("empirical_id")
        _ids = _a.index.intersection(_b.index).sort_values()
        _coverage.append({"model_name": _model, "country": _country, "neutral_valid": len(_a), "anti_valid": len(_b), "common_valid": len(_ids)})
        if not len(_ids):
            continue
        _a, _b = _a.loc[_ids], _b.loc[_ids]
        assert _a.empirical.equals(_b.empirical)
        _q = probabilities(_a.empirical)
        _ma, _mb = distribution_metrics(probabilities(_a.model), _q), distribution_metrics(probabilities(_b.model), _q)
        _ma.update(accuracy=_a.exact_match.mean(), MAE=_a.absolute_ordinal_error.mean())
        _mb.update(accuracy=_b.exact_match.mean(), MAE=_b.absolute_ordinal_error.mean())
        _row = {"model_name": _model, "country": _country, "N_common": len(_ids)}
        for _metric in ["neutral_mass", "neutral_excess", "IM", "dIM", "H_norm", "entropy_ratio", "JSD", "W1", "accuracy", "MAE"]:
            _row.update({f"{_metric}_neutral": _ma[_metric], f"{_metric}_anti": _mb[_metric], f"delta_{_metric}": _mb[_metric] - _ma[_metric]})
        _row["overshoot_flag"] = ((_ma["neutral_excess"] > 0 and _mb["neutral_excess"] < 0)
                                  or (_ma["dIM"] > 0 and _mb["dIM"] < 0) or _mb["entropy_ratio"] > 1)
        _rows.append(_row)
    paired = pd.DataFrame(_rows)
    coverage = pd.DataFrame(_coverage)
    _metric_names = ["neutral_mass", "IM", "H_norm", "JSD", "W1", "accuracy", "MAE"]
    _tests = []
    # Un modelo aporta un valor al test conjunto, evitando tratar países como réplicas independientes.
    _pooled = paired.groupby("model_name")[[f"delta_{x}" for x in _metric_names]].mean()
    for _scope, _table in [("Media de países por modelo", _pooled), *list(paired.groupby("country"))]:
        for _metric in _metric_names:
            _alternative = "greater" if _metric in ["H_norm", "accuracy"] else "less"
            _tests.append({"scope": _scope, "metric": _metric, "alternative": _alternative,
                           "mean_delta": _table[f"delta_{_metric}"].mean(),
                           **paired_test(_table[f"delta_{_metric}"], _alternative)})
    tests = pd.DataFrame(_tests)
    # Corrección Holm dentro de cada conjunto de siete contrastes exploratorios.
    tests["wilcoxon_p_holm"] = np.nan
    for _scope, _g in tests.groupby("scope"):
        _order = _g.wilcoxon_p.sort_values().index
        _adjusted = np.maximum.accumulate(tests.loc[_order, "wilcoxon_p"].to_numpy() * np.arange(len(_g), 0, -1))
        tests.loc[_order, "wilcoxon_p_holm"] = np.minimum(_adjusted, 1)
    for _name, _frame in [("comparacion_prompts", paired), ("cobertura_pares", coverage), ("tests_pareados", tests)]:
        _frame.to_csv(OUTPUT_DIR / f"{_name}.csv", index=False)
    return paired, coverage, tests


@app.cell
def _(mo, paired, coverage, tests):
    mo.vstack([
        mo.md("""## ¿El prompt anti-punto-medio mejora las respuestas?

        `delta` significa **anti_midpoint_v2 − neutral_v1**. Se recalculan ambas
        distribuciones sobre los perfiles válidos comunes de cada modelo/país.
        Esto evita atribuir al prompt un cambio causado solamente por diferentes
        perfiles descartados. La cobertura de esos pares se informa abajo.

        Reducciones en masa neutral, JSD, W1 o MAE, y aumentos de exactitud favorecen
        las hipótesis respectivas. Aumentar entropía ayuda a corregir compresión solo
        si también mejora la fidelidad. Reducir neutralidad cuando ya estaba por
        debajo de la WVS puede empeorar el sesgo.

        `overshoot_flag` señala cambio de exceso neutral/interior positivo a negativo o ratio
        de entropía mayor que 1. Es una alerta descriptiva, no prueba de daño.

        Se reproducen Wilcoxon unilateral, tamaño de efecto rank-biserial y test de
        signos del original, por país y con una media de países por modelo. Se añade
        Holm para siete contrastes por ámbito. Los seis modelos no son una muestra
        aleatoria y varios comparten familia; los valores p son exploratorios.
        Además, Transformers cambia entre corridas (5.18 a 5.19 en modelos locales),
        por lo que las diferencias no pueden atribuirse exclusivamente al prompt.
        """),
        mo.ui.table(coverage, selection=None),
        mo.ui.table(paired.round(4), selection=None),
        mo.ui.table(tests.round(4), selection=None),
    ])
    return


@app.cell
def _(metrics, distributions, paired, quality, selected, reference, PROMPTS, SHORT_OPTIONS, OPTIONS, OUTPUT_DIR, plt, np, pd):
    plt.rcParams.update({"font.size": 9, "figure.dpi": 110, "savefig.dpi": 200})
    models = sorted(metrics.model_name.unique())
    countries = sorted(metrics.country.unique())
    figures = {}
    figure_notes = {}

    def save_figure(name, fig, note):
        fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight")
        figures[name] = fig
        figure_notes[name] = note
        plt.close(fig)

    # Distribuciones: barras WVS y puntos de cada modelo sobre la misma escala.
    _fig, _axes = plt.subplots(len(countries), 2, figsize=(14, 11), squeeze=False)
    for _r, _country in enumerate(countries):
        for _c, _prompt in enumerate(PROMPTS):
            _ax = _axes[_r, _c]
            _wvs = reference.loc[reference.country == _country, "empirical"].value_counts(normalize=True).reindex(OPTIONS, fill_value=0)
            _ax.bar(np.arange(5), _wvs, color="lightgray", label="WVS: 400 perfiles")
            for _i, _model in enumerate(models):
                _d = distributions.loc[(distributions.country == _country) & (distributions.prompt_version == _prompt) & (distributions.model_name == _model)].sort_values("position")
                _ax.plot(np.arange(5), _d.p_model, marker="o", label=_model, alpha=.8)
            _ax.set(xticks=np.arange(5), xticklabels=SHORT_OPTIONS, ylim=(0, 1), ylabel="Proporción", title=f"{_country} · {_prompt}")
    _axes[0, 0].legend(fontsize=7)
    _fig.tight_layout()
    save_figure("01_distribuciones", _fig, "Barras grises: muestra WVS completa por país. Líneas: respuestas válidas de cada modelo. Los picos muestran concentración. En Qwen3.5-4B la cobertura difiere; las métricas principales usan su referencia humana emparejada.")

    _fig, _axes = plt.subplots(1, len(countries), figsize=(16, 4), sharex=True, squeeze=False)
    for _ax, _country in zip(_axes[0], countries):
        _d = paired.loc[paired.country == _country].set_index("model_name").reindex(models)
        for _i, _model in enumerate(models):
            _row = _d.loc[_model]
            _ax.plot([_row.neutral_mass_neutral, _row.neutral_mass_anti], [_i, _i], color="gray")
        _ax.scatter(_d.neutral_mass_neutral, np.arange(len(models)), label="Neutral", color="#2878b5")
        _ax.scatter(_d.neutral_mass_anti, np.arange(len(models)), label="Anti", color="#df6828")
        _ax.scatter(_d.neutral_mass_neutral - _d.neutral_excess_neutral, np.arange(len(models)), marker="|", s=160, color="black", label="WVS emparejada")
        _ax.set(yticks=np.arange(len(models)), yticklabels=models, xlabel="Masa neutral", xlim=(0, 1), title=_country)
    _axes[0, 0].legend(fontsize=7)
    _fig.tight_layout()
    save_figure("02_cambio_neutralidad", _fig, "Cada segmento compara los prompts en los mismos perfiles. La marca negra indica neutralidad humana: acercarse a ella es más informativo que simplemente disminuir la neutralidad.")

    _fig, _axes = plt.subplots(1, len(countries), figsize=(16, 4), sharex=True, squeeze=False)
    for _ax, _country in zip(_axes[0], countries):
        _d = paired.loc[paired.country == _country].set_index("model_name").reindex(models)
        for _i, _model in enumerate(models):
            _row = _d.loc[_model]
            _ax.plot([_row.IM_neutral, _row.IM_anti], [_i, _i], color="gray")
        _ax.scatter(_d.IM_neutral, np.arange(len(models)), label="Neutral", color="#2878b5")
        _ax.scatter(_d.IM_anti, np.arange(len(models)), label="Anti", color="#df6828")
        _ax.scatter(_d.IM_neutral - _d.dIM_neutral, np.arange(len(models)), marker="|", s=160, color="black", label="WVS emparejada")
        _ax.set(yticks=np.arange(len(models)), yticklabels=models, xlabel="Masa interior (categorías 2–4)", xlim=(0, 1), title=_country)
    _axes[0, 0].legend(fontsize=7)
    _fig.tight_layout()
    save_figure("02b_cambio_masa_interior", _fig, "Adaptación directa del gráfico del tutorial: masa en categorías interiores frente a WVS. Una caída de esta masa equivale a un aumento de respuestas extremas; solo indica mejora si acerca la distribución a la humana.")

    _fig, _ax = plt.subplots(figsize=(10, 6))
    for _model in models:
        _d = paired.loc[paired.model_name == _model]
        _ax.scatter(_d.delta_H_norm, _d.delta_JSD, label=_model, s=60)
        for _row in _d.itertuples():
            _ax.annotate(_row.country, (_row.delta_H_norm, _row.delta_JSD), fontsize=7)
    _ax.axhline(0, color="gray"); _ax.axvline(0, color="gray")
    _ax.set(xlabel="Cambio de entropía (anti − neutral)", ylabel="Cambio de JSD (anti − neutral)", title="Mitigación: abajo a la derecha = más diversidad y mayor fidelidad")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    save_figure("03_cuadrantes", _fig, "Abajo a la derecha: aumenta diversidad y disminuye divergencia. Arriba a la derecha: aumenta diversidad pero se aleja de los humanos. La diversidad sola no demuestra una mejora.")

    _fig, _axes = plt.subplots(len(countries), 2, figsize=(14, 12), squeeze=False, layout="constrained")
    _limit = max(.01, distributions.signed_error.abs().max())
    for _r, _country in enumerate(countries):
        for _c, _prompt in enumerate(PROMPTS):
            _ax = _axes[_r, _c]
            _d = distributions.loc[(distributions.country == _country) & (distributions.prompt_version == _prompt)]
            _matrix = _d.pivot(index="model_name", columns="category", values="signed_error").reindex(index=models, columns=OPTIONS)
            _im = _ax.imshow(_matrix, cmap="RdBu_r", vmin=-_limit, vmax=_limit, aspect="auto")
            _ax.set(xticks=np.arange(5), xticklabels=SHORT_OPTIONS, yticks=np.arange(len(models)), yticklabels=models, title=f"{_country} · {_prompt}")
            for _i in range(len(models)):
                for _j in range(5):
                    _ax.text(_j, _i, f"{_matrix.iloc[_i, _j]:+.2f}", ha="center", va="center", fontsize=8)
    _fig.colorbar(_im, ax=_axes.ravel().tolist(), label="Proporción modelo − WVS emparejada", shrink=.6)
    save_figure("04_errores_categoria", _fig, "Rojo indica sobrerrepresentación y azul subrepresentación. Cada fila compara un modelo con los humanos de sus mismos perfiles válidos. El color tiene una escala compartida.")

    # Una figura de confusión por país: filas = respuesta humana, columnas = modelo.
    for _country in countries:
        _fig, _axes = plt.subplots(len(models), 2, figsize=(11, 3 * len(models)), squeeze=False, layout="constrained")
        for _r, _model in enumerate(models):
            for _c, _prompt in enumerate(PROMPTS):
                _ax = _axes[_r, _c]
                _g = selected.loc[(selected.country == _country) & (selected.model_name == _model) & (selected.prompt_version == _prompt)]
                _cm = pd.crosstab(_g.empirical, _g.model).reindex(index=OPTIONS, columns=OPTIONS, fill_value=0)
                _norm = _cm.div(_cm.sum(axis=1).replace(0, np.nan), axis=0)
                _im = _ax.imshow(_norm, vmin=0, vmax=1, cmap="Blues")
                _ax.set(xticks=np.arange(5), xticklabels=SHORT_OPTIONS, yticks=np.arange(5), yticklabels=SHORT_OPTIONS, title=f"{_model} · {_prompt}", xlabel="Simulada", ylabel="Humana")
                for _i in range(5):
                    for _j in range(5):
                        _ax.text(_j, _i, str(_cm.iloc[_i, _j]), ha="center", va="center", color="white" if _norm.iloc[_i, _j] > .55 else "black")
        _fig.suptitle(f"Matrices de confusión · {_country}: números = conteos; color = proporción de fila")
        save_figure(f"05_confusion_{_country.replace(' ', '_')}", _fig, "La diagonal contiene coincidencias exactas. Los números son conteos; el color se normaliza por respuesta humana para hacer visibles categorías poco frecuentes. Las celdas sin denominador quedan sin color.")

    _fig, _axes = plt.subplots(len(countries), 2, figsize=(14, 11), squeeze=False, layout="constrained")
    for _r, _country in enumerate(countries):
        for _c, (_metric, _label) in enumerate([("accuracy", "Exactitud"), ("MAE", "Error ordinal medio")]):
            _ax = _axes[_r, _c]
            for _i, _prompt in enumerate(PROMPTS):
                _d = metrics.loc[(metrics.country == _country) & (metrics.prompt_version == _prompt)].set_index("model_name").reindex(models)
                _y = np.arange(len(models)) + (_i - .5) * .22
                _errors = np.maximum(0, np.vstack([_d[_metric] - _d[f"{_metric}_lo"], _d[f"{_metric}_hi"] - _d[_metric]]))
                _ax.errorbar(_d[_metric], _y, xerr=_errors, fmt="o", label=_prompt, capsize=3)
            _ax.set(yticks=np.arange(len(models)), yticklabels=models, xlabel=_label, title=_country)
    _axes[0, 0].legend(fontsize=7)
    save_figure("06_exactitud_error", _fig, "Puntos e intervalos bootstrap del 95% sobre respuestas válidas. Mayor exactitud y menor error son mejores; el solapamiento de intervalos no sustituye una comparación apareada.")

    _fig, _axes = plt.subplots(len(countries), 2, figsize=(14, 10), squeeze=False, layout="constrained")
    for _r, _country in enumerate(countries):
        for _c, _prompt in enumerate(PROMPTS):
            _ax = _axes[_r, _c]
            for _model in models:
                _g = selected.loc[(selected.country == _country) & (selected.prompt_version == _prompt) & (selected.model_name == _model)]
                _freq = _g.signed_ordinal_error.value_counts(normalize=True).reindex(range(-4, 5), fill_value=0)
                _ax.plot(range(-4, 5), _freq, marker="o", label=_model)
            _ax.set(xticks=range(-4, 5), xlabel="Respuesta simulada − humana", ylabel="Proporción", title=f"{_country} · {_prompt}")
    _axes[0, 0].legend(fontsize=7)
    save_figure("07_error_individual", _fig, "Cero = coincidencia exacta. Errores negativos: el modelo evalúa la inmigración más positivamente que el humano; positivos: más negativamente. Colas lejanas de cero indican desacuerdos grandes.")
    return figures, figure_notes, models, countries


@app.cell
def _(mo, figures, figure_notes):
    mo.vstack([mo.vstack([mo.md(f"### {name}\n\n{figure_notes[name]}"), mo.as_html(fig)]) for name, fig in figures.items()])
    return


@app.cell
def _(metrics, paired, quality, tests, audit, dataset, main, selected, reference, models, countries, figure_notes, OUTPUT_DIR, pd, mo):
    # Texto calculado a partir de las tablas; se actualiza al cambiar los datos.
    _summary = []
    for (_model, _prompt), _g in selected.groupby(["model_name", "prompt_version"]):
        _total = main.loc[(main.model_name == _model) & (main.prompt_version == _prompt)].drop_duplicates(["country", "empirical_id"])
        _summary.append({"model": _model, "prompt": _prompt, "N_valid": len(_g),
                         "accuracy_valid": _g.exact_match.mean(), "accuracy_all": _g.exact_match.sum() / len(_total),
                         "MAE": _g.absolute_ordinal_error.mean()})
    summary = pd.DataFrame(_summary)
    summary.to_csv(OUTPUT_DIR / "resumen_modelos.csv", index=False)
    _lines = ["# Resultados Q121", "",
        f"Se auditaron {len(audit)} archivos. El dataset unificado y el análisis usan {int(audit.included.sum())} corridas, {len(main):,} intentos y {len(selected):,} respuestas válidas, sobre {len(reference):,} perfiles.", "",
        "## Archivo adicional",
        "La corrida c8cc559dc6ee47fa contiene 21 intentos (12 válidos) de Qwen3.5-4B, con thinking=True y 4096 tokens. Se excluye del dataset y del análisis porque la corrida completa usa thinking=False y 512 tokens. Solo queda documentada en la auditoría.", "",
        "## Calidad y cobertura"]
    for _row in quality.loc[quality.missing_valid_profiles > 0].itertuples():
        _lines.append(f"- {_row.model_name}, {_row.country}, {_row.prompt_version}: {_row.missing_valid_profiles}/{_row.profiles} perfiles sin respuesta válida ({_row.invalid_profile_rate:.1%}).")
    _lines.extend(["", "## Dispersión, neutralidad y fidelidad"])
    for _prompt, _g in metrics.groupby("prompt_version"):
        _lines.append(f"- {_prompt}: entropía menor que la humana en {int((_g.entropy_ratio < 1).sum())}/{len(_g)} celdas; exceso neutral en {int((_g.neutral_excess > 0).sum())}/{len(_g)}; exceso interior en {int((_g.dIM > 0).sum())}/{len(_g)}. JSD media por celda: {_g.JSD.mean():.3f}; W1 media: {_g.W1.mean():.3f}.")
    _lines.extend(["", "## Desempeño individual"])
    for _row in summary.itertuples():
        _lines.append(f"- {_row.model}, {_row.prompt}: exactitud válida {_row.accuracy_valid:.1%}; sobre todos los perfiles {_row.accuracy_all:.1%}; error ordinal medio {_row.MAE:.3f} (0–4).")
    _lines.extend(["", "## Cambio del prompt, sobre perfiles válidos comunes"])
    for _model, _g in paired.groupby("model_name"):
        _lines.append(f"- {_model}: cambio medio entre países de neutralidad {_g.delta_neutral_mass.mean():+.3f}, entropía {_g.delta_H_norm.mean():+.3f}, JSD {_g.delta_JSD.mean():+.3f}, W1 {_g.delta_W1.mean():+.3f}, exactitud {_g.delta_accuracy.mean():+.1%}, MAE {_g.delta_MAE.mean():+.3f}.")
    _lines.append(f"\nLa JSD mejora en {int((paired.delta_JSD < 0).sum())}/{len(paired)} pares modelo/país; la exactitud en {int((paired.delta_accuracy > 0).sum())}/{len(paired)}. Hay {int(paired.overshoot_flag.sum())} alertas de posible sobrecorrección.")
    if (metrics.neutral_excess <= 0).all():
        _lines.append("**No se observa exceso de respuestas neutrales frente a los humanos en ninguna celda.** La hipótesis de sesgo hacia el punto medio no se sostiene descriptivamente en estas corridas. Reducir más la neutralidad no resuelve ese problema y puede aumentar la distancia respecto de WVS.")
    if (paired.delta_JSD > 0).all() and (paired.delta_accuracy < 0).all():
        _lines.append("**El prompt anti-punto-medio empeora la fidelidad de distribución y la exactitud en los 18 pares modelo/país.** Este patrón descriptivo favorece el prompt neutral para estos datos; las diferencias de entorno impiden una atribución causal exclusiva al prompt.")
    _lines.extend(["", "## Lectura de las matrices de confusión"])
    for (_model, _prompt), _g in selected.groupby(["model_name", "prompt_version"]):
        _misses = _g.loc[~_g.exact_match]
        if not _misses.empty:
            _counts = _misses.groupby(["empirical", "model"]).size().sort_values(ascending=False)
            _actual, _predicted = _counts.index[0]
            _lines.append(f"- {_model}, {_prompt}: el desacuerdo más frecuente, sumando países, es humano «{_actual}» → modelo «{_predicted}» ({int(_counts.iloc[0])} casos). El error con signo promedio es {_g.signed_ordinal_error.mean():+.3f}.")
    _lines.extend(["", "## Pruebas exploratorias"])
    for _row in tests.loc[tests.scope == "Media de países por modelo"].itertuples():
        _lines.append(f"- {_row.metric}: delta media {_row.mean_delta:+.3f}; Wilcoxon p={_row.wilcoxon_p:.4f}, Holm p={_row.wilcoxon_p_holm:.4f}; rank-biserial={_row.rank_biserial:+.3f}; test de signos p={_row.sign_p:.4f}.")
    _lines.extend(["", "## Interpretación y límites",
        "Las comparaciones de prompts usan los mismos perfiles válidos en ambas condiciones. Las tablas de métricas individuales usan todos los válidos de cada corrida. La referencia principal son sus respuestas humanas emparejadas; se conserva una comparación adicional con los 400 perfiles por país.",
        "Los intervalos bootstrap (2000 réplicas, semilla 42) remuestrean pares humano/modelo. Describen variación de perfiles, no variación de nuevas generaciones del modelo. Los tests conjuntos usan un promedio de países por modelo, no 18 observaciones independientes.",
        "La falta de respuestas válidas puede seleccionar perfiles; se informa exactitud con y sin penalización de fallos. No se imputa el contenido de una respuesta inválida.",
        "La versión de Transformers cambia entre prompts locales y los modelos tienen configuraciones distintas. Los efectos son asociaciones entre corridas: no identifican un efecto causal puro del prompt ni permiten atribuir diferencias exclusivamente a la arquitectura.",
        "Una sola pregunta, tres países, seis modelos relacionados y una generación por perfil limitan la generalización. Los valores p son exploratorios, con Holm por ámbito. No se aplican pesos de encuesta.",
        "La adaptación de Q130 cambia log₂(4) por log₂(5), mide la masa neutral directamente y define masa interior como categorías 2–4. MAE y W1 suponen intervalos ordinales unitarios.", "", "## Figuras"])
    for _name, _note in figure_notes.items():
        _lines.append(f"\n![{_name}]({_name}.png)\n\n{_note}")
    report = "\n\n".join(_lines)
    (OUTPUT_DIR / "informe.md").write_text(report, encoding="utf-8")
    mo.md(report.split("## Figuras")[0])
    return summary, report


@app.cell
def _(mo, OUTPUT_DIR, unified_path):
    mo.md(f"""
    ## Archivos generados

    - Dataset con procedencia, configuración, razonamiento, respuesta y errores individuales: `{unified_path}`.
    - Auditoría, calidad, métricas, distribuciones, matrices, pares, tests y resumen: `{OUTPUT_DIR}`.
    - Figuras PNG e informe con resultados calculados: `{OUTPUT_DIR}/informe.md`.

    Fuentes metodológicas: `factor_data_silicon_tutorial/src/analyze_q130_bias.py`
    y `results/report.md` del repositorio original. Las fórmulas se reproducen en
    las celdas y la tabla explicativa; las adaptaciones a Q121 están indicadas.
    """)
    return


if __name__ == "__main__":
    app.run()
