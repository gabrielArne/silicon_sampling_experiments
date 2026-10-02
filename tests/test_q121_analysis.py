"""Pruebas de las transformaciones estadísticas del notebook de análisis."""

import ast
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "analysis_q121_models.py"


def load_cell(namespace, marker):
    tree = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in tree.body:
        if not isinstance(cell, ast.FunctionDef):
            continue
        definitions = {
            node.name for node in cell.body if isinstance(node, ast.FunctionDef)
        } | {
            target.id
            for node in cell.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        if marker in definitions:
            body = [node for node in cell.body if not isinstance(node, ast.Return)]
            exec(compile(ast.Module(body=body, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
            return
    raise AssertionError(f"No se encontró la celda {marker}")


def metric_namespace():
    namespace = {"np": np, "math": math}
    load_cell(namespace, "CATEGORIES")
    load_cell(namespace, "distribution_metrics")
    return namespace


def test_identical_distribution_has_zero_distance():
    ns = metric_namespace()
    probabilities = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    result = ns["distribution_metrics"](probabilities, probabilities)
    assert result["JSD"] == 0
    assert result["W1"] == 0
    assert result["d_midpoint_mass"] == 0
    assert result["d_mean_position"] == 0
    assert result["entropy_ratio"] == 1


def test_midpoint_and_ordinal_direction_are_q121_specific():
    ns = metric_namespace()
    model = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    empirical = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    result = ns["distribution_metrics"](model, empirical)
    assert result["midpoint_mass"] == 1
    assert result["d_midpoint_mass"] == 1
    assert result["extreme_mass"] == 0
    assert result["W1"] == 2
    assert result["d_mean_position"] == 2


def test_final_attempt_prefers_latest_valid_response():
    namespace = {
        "pd": pd,
        "np": np,
        "selected_runs": {
            "Qwen/Qwen3.5-4B": pd.DataFrame(
                [
                    {"country": "Argentina", "empirical_id": "1", "attempt": 1, "valid": False, "model": ""},
                    {"country": "Argentina", "empirical_id": "1", "attempt": 2, "valid": True, "model": "Quite good"},
                    {"country": "Argentina", "empirical_id": "1", "attempt": 3, "valid": True, "model": "Very good"},
                    {"country": "Uruguay", "empirical_id": "2", "attempt": 1, "valid": False, "model": ""},
                    {"country": "Uruguay", "empirical_id": "2", "attempt": 2, "valid": False, "model": ""},
                ]
            )
        },
    }
    load_cell(namespace, "CATEGORIES")
    load_cell(namespace, "choose_profile_result")
    result = namespace["respondent_results"].set_index(["country", "empirical_id"])
    assert result.loc[("Argentina", "1"), "attempt"] == 3
    assert result.loc[("Argentina", "1"), "model"] == "Very good"
    assert result.loc[("Uruguay", "2"), "attempt"] == 2
    assert result.loc[("Uruguay", "2"), "answer_for_distribution"] == "invalid/no answer"

