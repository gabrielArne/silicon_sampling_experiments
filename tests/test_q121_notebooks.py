"""Pruebas de recuperación sin GPU, credenciales ni llamadas reales a Drive."""

import ast
import hashlib
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("q121_*.py"))
VALID_JSON = json.dumps({"thinking": "My work shapes my view. I expect a positive impact.", "Q121": "Quite good"})


def load_cell(namespace, notebook, marker, functions_only=False):
    """Ejecuta una celda por su definición, sin importar ni iniciar el notebook."""
    tree = ast.parse(notebook.read_text(encoding="utf-8"))
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
            if functions_only:
                body = [node for node in body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=body, type_ignores=[]), str(notebook), "exec"), namespace)
            return
    raise AssertionError(f"No se encontró la celda {marker}")


class Request:
    def __init__(self, action):
        self.action = action

    def execute(self, **kwargs):
        return self.action()


class Media:
    def __init__(self, stream, **kwargs):
        self.content = stream.getvalue()


class FakeDrive:
    def __init__(self):
        self.entries = {}
        self.fail_upload = False
        self.creates = 0

    def files(self):
        return self

    def find(self, name, folder_id):
        return next((key for key, entry in self.entries.items() if entry["name"] == name), None)

    def get_media(self, fileId):
        return Request(lambda: self.entries[fileId]["content"])

    def create(self, body, media_body, **kwargs):
        def action():
            if self.fail_upload:
                raise OSError("Drive no disponible")
            self.creates += 1
            key = str(self.creates)
            self.entries[key] = {"name": body["name"], "content": media_body.content}
            return {"id": key}
        return Request(action)

    def update(self, fileId, media_body):
        def action():
            if self.fail_upload:
                raise OSError("Drive no disponible")
            self.entries[fileId]["content"] = media_body.content
        return Request(action)


@pytest.fixture(params=NOTEBOOKS, ids=lambda path: path.stem)
def experiment(request, tmp_path):
    notebook = request.param
    drive = FakeDrive()
    ns = {
        "Path": Path, "hashlib": hashlib, "io": io, "json": json, "os": os,
        "re": re, "time": time, "datetime": datetime, "timezone": timezone, "pd": pd,
        "drive": drive, "find_drive_file": drive.find, "folder_id": "folder",
        "MediaIoBaseUpload": Media,
    }
    for marker in ("MODEL_NAME", "build_prompt_q121", "parse_response", "save_local"):
        load_cell(ns, notebook, marker)
    ns.update({
        "output_path": tmp_path / "results.csv", "run_id": "test-run",
        "parameters_json": "{}",
        "profiles": pd.DataFrame([
            {"id": "001", "country": country, "sex": "woman", "education": "secondary education",
             "class": "working class", "occupation": "sales worker", "political_identity": "center",
             "q121_response": "Quite bad"}
            for country in ("Argentina", "Uruguay")
        ]),
    })
    def raw_response():
        if ns["MODEL_NAME"].startswith("openai/"):
            return "<|channel|>analysis<|message|>Analysis<|end|><|start|>assistant<|channel|>final<|message|>" + VALID_JSON + "<|fim_suffix|>"
        return "Native reasoning</think>\n" + VALID_JSON + "<|im_end|>"
    ns["good_raw"] = raw_response()
    ns["generate_one"] = lambda row, seed: ns["good_raw"]
    load_cell(ns, notebook, "run_experiment", functions_only=True)
    return ns


def test_models_match_llms_notebook():
    source = (ROOT / "notebooks/llms.ipynb").read_text(encoding="utf-8")
    expected = set(re.findall(r'(?:Qwen|openai)/[\w.\-]+', source))
    actual = set()
    for notebook in NOTEBOOKS:
        ns = {"Path": Path}
        load_cell(ns, notebook, "MODEL_NAME")
        actual.add(ns["MODEL_NAME"])
    assert actual == expected


def test_prompt_matches_original_and_excludes_empirical_answer(experiment):
    original = json.loads((ROOT / "notebooks/experiment_q121.ipynb").read_text(encoding="utf-8"))
    code = "".join(original["cells"][6]["source"]).split("# Vista de ejemplo")[0]
    ns = {"pd": pd}
    exec(code, ns)
    row = experiment["profiles"].iloc[0].to_dict()
    assert ns["SYSTEM_PROMPT_A"] == experiment["SYSTEM_PROMPT"]
    assert ns["build_prompt_q121"](row) == experiment["build_prompt_q121"](row)
    row["q121_response"] = "SECRET_EMPIRICAL_ANSWER"
    assert "SECRET_EMPIRICAL_ANSWER" not in experiment["build_prompt_q121"](row)


def test_only_final_valid_json_is_accepted(experiment):
    parse = experiment["parse_response"]
    assert parse(experiment["good_raw"])["Q121"] == "Quite good"
    if experiment["MODEL_NAME"].startswith("Qwen/"):
        # Con thinking desactivado, el JSON directo es la salida final esperada.
        assert parse(VALID_JSON)["Q121"] == "Quite good"
    else:
        with pytest.raises(ValueError):
            parse(VALID_JSON)
    for raw in ("<think>" + VALID_JSON, "unfinished", experiment["good_raw"].replace("Quite good", "Other")):
        with pytest.raises(ValueError):
            parse(raw)
    for thinking in ("", None, []):
        with pytest.raises(ValueError):
            parse(experiment["good_raw"].replace(json.loads(VALID_JSON)["thinking"], "") if thinking == "" else experiment["good_raw"].replace(json.dumps(json.loads(VALID_JSON)["thinking"]), json.dumps(thinking)))


def test_resume_from_drive_in_new_session_without_repeating_valid_profiles(experiment):
    first = experiment["run_experiment"]()
    assert len(first) == 2
    assert first["empirical_id"].tolist() == ["001", "001"]
    experiment["output_path"].unlink()  # Pérdida del disco efímero de molab.
    def unexpected_call(row, seed):
        raise AssertionError("No debe generar de nuevo perfiles completos")
    experiment["generate_one"] = unexpected_call
    resumed = experiment["run_experiment"]()
    assert len(resumed) == 2
    assert resumed["valid"].all()
    assert experiment["drive"].creates == 1


def test_drive_failure_keeps_local_batch_and_syncs_before_resuming(experiment):
    experiment["drive"].fail_upload = True
    with pytest.raises(OSError, match="Drive"):
        experiment["run_experiment"]()
    local = pd.read_csv(experiment["output_path"], dtype=str)
    assert len(local) == 2
    assert local["empirical_id"].eq("001").all()
    experiment["drive"].fail_upload = False
    def generate(row, seed):
        raise AssertionError("No debe repetir las respuestas válidas del lote local")
    experiment["generate_one"] = generate
    assert len(experiment["run_experiment"]()) == 2
    assert experiment["drive"].entries


def test_model_failure_saved_and_retried_without_losing_history(experiment):
    calls = []
    def generate(row, seed):
        calls.append(row["country"])
        if row["country"] == "Uruguay":
            raise RuntimeError("GPU sin memoria")
        return experiment["good_raw"]
    experiment["generate_one"] = generate
    with pytest.raises(RuntimeError, match="progreso"):
        experiment["run_experiment"]()
    partial = experiment["restore_checkpoint"](experiment["output_path"], "test-run")
    assert len(partial) == 2
    assert partial[0]["valid"] and not partial[1]["valid"]
    assert "GPU sin memoria" in partial[1]["error"]
    experiment["generate_one"] = lambda row, seed: experiment["good_raw"]
    resumed = experiment["run_experiment"]()
    assert resumed["valid"].sum() == 2
    assert resumed["attempt"].tolist() == [1, 1, 2]
    assert resumed.iloc[1]["seed"] != str(resumed.iloc[2]["seed"])


def test_keyboard_interrupt_preserves_preceding_responses(experiment):
    def generate(row, seed):
        if row["country"] == "Uruguay":
            raise KeyboardInterrupt()
        return experiment["good_raw"]
    experiment["generate_one"] = generate
    with pytest.raises(KeyboardInterrupt):
        experiment["run_experiment"]()
    recovered = experiment["restore_checkpoint"](experiment["output_path"], "test-run")
    assert len(recovered) == 1 and recovered[0]["valid"]


def test_invalid_output_keeps_raw_and_retries_next_run(experiment):
    experiment["generate_one"] = lambda row, seed: "not JSON"
    first = experiment["run_experiment"]()
    assert not first["valid"].any()
    assert first["raw_response"].eq("not JSON").all()
    experiment["generate_one"] = lambda row, seed: experiment["good_raw"]
    resumed = experiment["run_experiment"]()
    assert len(resumed) == 4 and resumed["valid"].sum() == 2


def test_checkpoint_rejects_different_run(experiment):
    experiment["run_experiment"]()
    with pytest.raises(ValueError, match="corrida"):
        experiment["restore_checkpoint"](experiment["output_path"], "another-run")


def test_run_identity_changes_with_parameters_but_not_sample_size(experiment):
    ns = experiment.copy()
    ns.update({"input_sha256": "input", "runtime": {"model_commit": "commit"}})
    notebook = next(p for p in NOTEBOOKS if f'MODEL_NAME = "{ns["MODEL_NAME"]}"' in p.read_text(encoding="utf-8"))
    load_cell(ns, notebook, "parameters")
    original = ns["run_id"]
    ns["N_SAMPLE"] = 10
    load_cell(ns, notebook, "parameters")
    assert ns["run_id"] == original
    ns["GENERATION"] = {**ns["GENERATION"], "temperature": 0.7}
    load_cell(ns, notebook, "parameters")
    assert ns["run_id"] != original


def test_atomic_write_failure_preserves_previous_csv(experiment, monkeypatch):
    experiment["run_experiment"]()
    before = experiment["output_path"].read_bytes()
    def fail_replace(*args):
        raise OSError("interrupted replacement")
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        experiment["save_local"]([{"new": "row"}], experiment["output_path"])
    assert experiment["output_path"].read_bytes() == before
