"""Checks numéricos del análisis, sin API, Drive ni generaciones de modelos."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def functions():
    tree = ast.parse((ROOT / "notebooks/analysis_q121.py").read_text(encoding="utf-8"))
    scope = dict(np=np, pd=pd, stats=stats, jensenshannon=jensenshannon, N_BOOT=100)
    for cell in tree.body:
        if isinstance(cell, ast.FunctionDef):
            for node in cell.body:
                if isinstance(node, ast.FunctionDef):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), "analysis", "exec"), scope)
    return scope


def test_uniform_and_identical_distributions(functions):
    p = np.ones(5) / 5
    m = functions["distribution_metrics"](p, p)
    assert m["H_norm"] == pytest.approx(1)
    assert m["entropy_ratio"] == pytest.approx(1)
    assert m["IM"] == pytest.approx(.6)
    assert m["EM"] == pytest.approx(.4)
    assert m["neutral_mass"] == pytest.approx(.2)
    assert m["JSD"] == pytest.approx(0)
    assert m["W1"] == pytest.approx(0)
    assert m["dmu"] == pytest.approx(0)


def test_opposite_extremes(functions):
    m = functions["distribution_metrics"](np.array([0, 0, 0, 0, 1.]), np.array([1., 0, 0, 0, 0]))
    assert m["JSD"] == pytest.approx(1)
    assert m["W1"] == pytest.approx(4)
    assert m["dmu"] == pytest.approx(4)
    assert m["H_norm"] == 0
    assert np.isnan(m["entropy_ratio"])


def test_jsd_is_divergence_not_distance(functions):
    p = np.array([.5, .5, 0, 0, 0])
    q = np.array([0, .5, .5, 0, 0])
    m = functions["distribution_metrics"](p, q)
    assert m["JSD"] == pytest.approx(.5)
    assert m["W1"] == pytest.approx(1)


def test_paired_bootstrap_preserves_perfect_matches(functions):
    g = pd.DataFrame({"empirical_ordinal": [1, 2, 3, 4, 5] * 20, "model_ordinal": [1, 2, 3, 4, 5] * 20})
    cis = functions["bootstrap_individual"](g, np.random.default_rng(42))
    assert cis["accuracy_lo"] == pytest.approx(1)
    assert cis["accuracy_hi"] == pytest.approx(1)
    assert cis["MAE_hi"] == 0
    assert cis["JSD_hi"] == pytest.approx(0)


def test_paired_test_direction_and_ties(functions):
    test = functions["paired_test"]
    assert test([0, 0], "less")["wilcoxon_p"] == 1
    result = test([-1, -2, -3, -4, -5, -6], "less")
    assert result["rank_biserial"] == -1
    assert result["sign_p"] == pytest.approx(1 / 64)
    assert result["n_favor"] == 6


def test_local_artifacts_are_consistent():
    output = ROOT / "data/analysis_q121"
    if not (output / "metricas.csv").exists():
        pytest.skip("Ejecutar el notebook para validar los artefactos locales.")
    d = pd.read_csv(ROOT / "data/output/dataset_respuestas_modelos.csv", dtype={"empirical_id": str})
    assert "c8cc559dc6ee47fa" not in set(d.run_id)
    assert len(d) == 14400
    assert d.source_file.nunique() == 12
    assert d.selected_response.sum() == 14132
    assert not d.duplicated(["model_name", "prompt_version", "country", "empirical_id"]).any()
    assert d.loc[~d.valid, "absolute_ordinal_error"].isna().all()
    m = pd.read_csv(output / "metricas.csv")
    assert len(m) == 36
    np.testing.assert_allclose(m.IM + m.EM, 1)
    np.testing.assert_allclose(m.dmu, m.ordinal_bias, atol=1e-12)
    assert (m.W1 <= m.MAE + 1e-12).all()
    cm = pd.read_csv(output / "matrices_confusion.csv")
    assert cm["count"].sum() == d.selected_response.sum()
    for key, group in cm.groupby(["model_name", "prompt_version", "country"]):
        actual = m.loc[(m.model_name == key[0]) & (m.prompt_version == key[1]) & (m.country == key[2])].iloc[0]
        accuracy = group.loc[group.empirical == group.predicted, "count"].sum() / group["count"].sum()
        assert accuracy == pytest.approx(actual.accuracy)


def test_molab_can_load_only_unified_csv(tmp_path, monkeypatch):
    """Molab recibe solo el CSV unificado; no necesita los archivos originales."""
    monkeypatch.chdir(tmp_path)
    rows = []
    for prompt, text in [("neutral_v1", "Select the single response option this person would most likely choose"),
                         ("anti_midpoint_v2", "Do NOT filter responses for political correctness middle option")]:
        for identifier in ["001", "002"]:
            valid = not (prompt == "anti_midpoint_v2" and identifier == "002")
            rows.append(dict(run_id=prompt, empirical_id=identifier, country="Argentina", model_name="test",
                system_prompt=text, empirical="Quite good", model="Quite good" if valid else "",
                valid=str(valid), parameters_json=json.dumps({"generation": {"max_output_tokens": 512}}),
                attempt="1", source_file=prompt + ".csv", sex="woman", education="secondary",
                **{"class": "working"}, occupation="sales", political_ident="center"))
    pd.DataFrame(rows).to_csv(tmp_path / "dataset_respuestas_modelos.csv", index=False)
    ns = dict(Path=Path, pd=pd, json=json, hashlib=hashlib, INPUT_DIR=Path("data/output"),
        UNIFIED_NAME="dataset_respuestas_modelos.csv", OUTPUT_DIR=Path("analysis"), EXCLUDED_RUNS={},
        OPTIONS=["Very good", "Quite good", "Neither good, nor bad", "Quite bad", "Rather bad"],
        PROMPTS=["neutral_v1", "anti_midpoint_v2"])
    tree = ast.parse((ROOT / "notebooks/analysis_q121.py").read_text(encoding="utf-8"))
    cell = next(c for c in tree.body if isinstance(c, ast.FunctionDef)
                and any(isinstance(n, ast.FunctionDef) and n.name == "identify_prompt" for n in c.body))
    code = compile(ast.Module(body=[n for n in cell.body if not isinstance(n, ast.Return)], type_ignores=[]), "loader", "exec")
    for _ in range(2):
        exec(code, ns)
        assert len(ns["dataset"]) == 4
        assert ns["dataset"].selected_response.sum() == 3
        assert set(ns["dataset"].empirical_id) == {"001", "002"}
        assert ns["dataset"].absolute_ordinal_error.isna().sum() == 1
