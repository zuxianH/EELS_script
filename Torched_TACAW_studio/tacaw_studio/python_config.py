"""Parse Config-style Python settings without importing or executing TACAW."""
from __future__ import annotations

import ast
import copy
import math
import operator
from pathlib import Path
from pprint import pformat

from .config import (DEFAULT_PARAMETERS, OPTIONAL_PARAMETERS, SCAN_KEYS,
                     apply_parameters, effective_parameters, validate_settings)

STUDIO_KEYS = {"datafolder", "trajectory_file", "trajectory_files", "intensities_zarray"}
ALLOWED_KEYS = set(DEFAULT_PARAMETERS) | OPTIONAL_PARAMETERS | STUDIO_KEYS


def render_python_config(settings: dict) -> str:
    try:
        p = effective_parameters(settings)
    except (ValueError, TypeError, KeyError):
        # Keep the editor accessible when the Scan tab has invalid draft geometry.
        p = copy.deepcopy(settings["parameters"])
    root = Path(settings["input_folder"]).expanduser().absolute()
    output = Path(settings.get("output_root") or root / "tacaw_runs").expanduser().absolute()
    values = {"name": p.pop("name"), "info": p.pop("info"), "datafolder": str(output)}
    for key in ("beam_energy_keV", "beam_conv_ang_mrad", "scanning_mode", "scanning_shape",
                "scanning_batch_shape", "scanning_origin", "scanning_basis_vectors",
                "scanning_box_end", "scanning_box_size", "sample_name", "sample_supercell_int",
                "sample_temperature_K", "sample_structure_file"):
        if key in p:
            values[key] = p.pop(key)
    values.setdefault("sample_structure_file", None)
    files = settings["inputs"]["files"]
    values["trajectory_files" if len(files) > 1 else "trajectory_file"] = list(files) if len(files) > 1 else (files[0] if files else None)
    values.update(p)
    values["intensities_zarray"] = settings["intensities_name"]
    comments = {"datafolder": "parent folder for new Studio runs",
                "sample_structure_file": "None = first prepared trajectory frame",
                "intensities_zarray": "filename inside the new run folder"}
    lines = ["# Edit values, then click Apply configuration (Ctrl/Cmd+Enter).",
             "# config_file and derived axes/status are generated for each run.",
             "config = Config("]
    for key, value in values.items():
        literal = pformat(value, width=95, compact=True, sort_dicts=False).replace("\n", "\n    ")
        line = f"    {key}={literal},"
        if key in comments:
            line += "  # " + comments[key]
        lines.append(line)
    lines.append(")")
    return "\n".join(lines) + "\n"


def _value(node: ast.AST, variables: dict):
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, (str, bool, int, float))):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Line {node.lineno}: '{node.id}' is undefined. Fill in a value or define it above Config(...).")
        return copy.deepcopy(variables[node.id])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_value(item, variables) for item in node.elts]
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise ValueError("Dictionary unpacking is not supported.")
        return {_value(key, variables): _value(value, variables) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _value(node.operand, variables)
        if type(operand) not in (int, float):
            raise ValueError("Signs require numeric values.")
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left, right = _value(node.left, variables), _value(node.right, variables)
        if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
            return left + right
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise ValueError("Arithmetic requires numeric values.")
        operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                      ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                      ast.Mod: operator.mod, ast.Pow: operator.pow}
        if type(node.op) not in operations or (isinstance(node.op, ast.Pow) and abs(right) > 32):
            raise ValueError("Use simple arithmetic with powers no greater than 32.")
        value = operations[type(node.op)](left, right)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Arithmetic must produce a finite real number.")
        return value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1 and not node.keywords:
        return str(_value(node.args[0], variables))
    if isinstance(node, ast.JoinedStr):
        parts = []
        for item in node.values:
            if isinstance(item, ast.Constant):
                parts.append(item.value)
            elif isinstance(item, ast.FormattedValue):
                value = _value(item.value, variables)
                if item.conversion == 114:
                    value = repr(value)
                elif item.conversion == 115:
                    value = str(value)
                elif item.conversion == 97:
                    value = ascii(value)
                spec = _value(item.format_spec, variables) if item.format_spec is not None else ""
                if len(spec) > 32 or any(c.isdigit() for c in spec):
                    # Keep formatting bounded; normal f"prefix {job_tag}" works.
                    raise ValueError("Use unformatted f-string values in this settings editor.")
                parts.append(format(value, spec))
            else:
                raise ValueError("Unsupported f-string value.")
        return "".join(parts)
    raise ValueError(f"Line {getattr(node, 'lineno', '?')}: use values or simple variables instead of '{ast.unparse(node)}'. Replace args.* with your values.")


def parse_python_config(code: str) -> dict:
    if len(code) > 100_000:
        raise ValueError("Configuration is too long (maximum 100,000 characters).")
    try:
        module = ast.parse(code)
    except SyntaxError as exc:
        raise ValueError(f"Line {exc.lineno}: {exc.msg}. Check commas, quotes and parentheses.") from exc
    variables, parameters, found = {}, {}, False
    for statement in module.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "torched_tacaw" and all(
                name.name == "Config" and name.asname is None for name in statement.names):
            continue
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            target, expression = statement.targets[0].id, statement.value
        elif isinstance(statement, ast.Expr):
            target, expression = None, statement.value
        else:
            raise ValueError(f"Line {statement.lineno}: use variable assignments followed by one Config(...) block.")
        is_config = isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id == "Config"
        if is_config:
            if found:
                raise ValueError("Use exactly one Config(...) block.")
            if expression.args:
                raise ValueError("Use named Config arguments, such as beam_energy_keV=60.0.")
            for keyword in expression.keywords:
                if keyword.arg is None:
                    raise ValueError("Config keyword unpacking (**kwargs) is not supported.")
                if keyword.arg not in ALLOWED_KEYS:
                    if keyword.arg == "config_file":
                        raise ValueError("config_file is generated inside each new run folder. Remove this argument and set datafolder to its parent folder.")
                    raise ValueError(f"Unsupported or runtime-owned parameter: {keyword.arg}")
                if keyword.arg in parameters:
                    raise ValueError(f"Parameter appears twice: {keyword.arg}")
                parameters[keyword.arg] = _value(keyword.value, variables)
            found = True
        elif target and not found and target not in ("Config", "str"):
            variables[target] = _value(expression, variables)
        else:
            raise ValueError(f"Line {statement.lineno}: use variable assignments before Config(...).")
    if not found:
        raise ValueError("Enter one Config(...) block, for example config = Config(beam_energy_keV=60.0).")
    return parameters


def apply_python_config(settings: dict, code: str) -> dict:
    try:
        inputs = parse_python_config(code)
    except (TypeError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise ValueError(f"Invalid Python setting: {exc}") from exc
    result = copy.deepcopy(settings)
    valid_geometry = True
    try:
        p = effective_parameters(settings)
    except (ValueError, TypeError, KeyError):
        p = copy.deepcopy(settings["parameters"])
        valid_geometry = False
    old_geometry = {key: p.get(key) for key in SCAN_KEYS}
    p.update({key: value for key, value in inputs.items() if key not in STUDIO_KEYS})
    structure = p.get("sample_structure_file")
    if structure is not None and not isinstance(structure, str):
        raise ValueError("sample_structure_file must be a path string or None.")
    if structure is None:
        p.pop("sample_structure_file", None)
    updated = apply_parameters(result, p)
    geometry = {key: p.get(key) for key in SCAN_KEYS}
    if valid_geometry and geometry == old_geometry and (settings["scan"]["kind"] != "Plane wave" or p["beam_conv_ang_mrad"] == 0):
        updated["scan"] = copy.deepcopy(settings["scan"])
    root = Path(settings["input_folder"]).expanduser().absolute()
    if "trajectory_file" in inputs and "trajectory_files" in inputs:
        raise ValueError("Use trajectory_file or trajectory_files, not both.")
    if "trajectory_file" in inputs or "trajectory_files" in inputs:
        files = inputs.get("trajectory_files") if "trajectory_files" in inputs else ([inputs["trajectory_file"]] if inputs["trajectory_file"] else [])
        if not isinstance(files, list) or any(not isinstance(path, str) or not path.strip() for path in files):
            raise ValueError("trajectory_file must be a path or None; trajectory_files must be a list of paths.")
        paths = [str((Path(path).expanduser() if Path(path).expanduser().is_absolute() else root / Path(path).expanduser()).absolute()) for path in files]
        updated["inputs"]["files"] = paths
        updated["inputs"]["bead_set"] = len(paths) > 1
        if paths and paths != settings["inputs"]["files"]:
            updated["input_folder"] = str(Path(paths[0]).parent)
    if "datafolder" in inputs:
        value = inputs["datafolder"]
        if not isinstance(value, str):
            raise ValueError("datafolder must be a parent folder path.")
        path = Path(value).expanduser()
        resolved = str(path if path.is_absolute() else root / path) if value else ""
        default_root = str(root / "tacaw_runs")
        updated["output_root"] = "" if not settings.get("output_root") and resolved == default_root else resolved
    if "intensities_zarray" in inputs:
        updated["intensities_name"] = inputs["intensities_zarray"]
    validate_settings(updated)
    return updated
