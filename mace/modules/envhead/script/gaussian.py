"""Strict single-point Gaussian force logs -> EnvHead HDF5 or extended XYZ.

This parser supports nonperiodic, nosymm, single-SCF energy/force jobs only.
It stores full MM sources, never precomputes field-only training inputs.
"""
import argparse
import json
import multiprocessing
import os
import re
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HARTREE_EV = 27.211386245988
FORCE_EV_ANGSTROM = 51.4220674763
NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[DEde][+-]?\d+)?"


def _number(value):
    return float(value.replace("D", "E").replace("d", "e"))


def _one(pattern, text, name):
    values = re.findall(pattern, text, re.M)
    if len(values) != 1:
        raise ValueError(f"Expected exactly one {name}; found {len(values)}")
    return values[0]


def _table(text, marker, columns):
    section = text.split(marker)
    if len(section) != 2:
        raise ValueError(f"Expected exactly one {marker}")
    rows = []
    for line in section[1].splitlines():
        match = re.fullmatch(columns, line)
        if match:
            rows.append([_number(v) for v in match.groups()])
        elif rows:
            break
    if not rows:
        raise ValueError(f"Empty or malformed {marker}")
    return np.asarray(rows)


def _identity(path):
    # Preserve background variants in sample_id, join geometry families in pair_id.
    match = re.fullmatch(
        r"(samples|samples-no-background|samples_onlyprotein)__frame_(\d+)__([^_]+(?:-[^_]+)?)__(.+)",
        path.stem,
    )
    if match:
        family, frame, kind, tail = match.groups()
        tail = tail.replace("_onlyprotein__", "__")
        protein = tail.split("__")[0]
        return dict(
            sample_id=path.stem,
            pair_id=tail,
            group_id=f"{protein}__frame_{frame}",
            protein=protein,
            frame=frame,
            fragment_type=kind,
            family=family,
        )
    return dict(
        sample_id=str(path.resolve()),
        pair_id=str(path.resolve()),
        group_id=str(path.resolve()),
        protein="unknown",
        frame="unknown",
        fragment_type="unknown",
        family="unknown",
    )


def parse_log(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if text.count("Normal termination of Gaussian") != 1 or "Error termination" in text:
        raise ValueError(
            "Missing unique normal termination or job terminated with error"
        )
    # The archive also repeats the route, sometimes starting at a print line.
    # Its backslash-delimited payload is not the human-readable route block.
    route_echo = _one(r"(?s:^\s*(\#[^\\\n]*\n[^\\]*?))^\s*-{5,}\s*$", text, "route")
    # Gaussian wraps at a fixed print width, including inside keywords (n/osymm).
    # Remove print indentation, retaining original trailing spaces at word breaks.
    route = " ".join("".join(line.lstrip() for line in route_echo.splitlines()).split())
    if not re.search(r"\bnosymm\b", route, re.I) or re.search(
        r"\b(opt|freq|irc|scan)\b", route, re.I
    ):
        raise ValueError("Only nosymm single-point force jobs are supported")
    if not re.search(r"\bforce\b", route, re.I):
        raise ValueError("Gaussian force keyword required")
    charge, multiplicity = _one(
        r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", text, "charge/multiplicity"
    )
    markers = [m for m in ("Z-Matrix orientation:", "Input orientation:") if m in text]
    if len(markers) != 1 or "Standard orientation:" in text:
        raise ValueError(
            "Require one input-frame orientation, without standard-frame rotation"
        )
    marker = markers[0]
    if "Coordinates (Angstroms)" not in text.split(marker)[1][:400]:
        raise ValueError("Orientation must explicitly use Angstroms")
    atoms = _table(
        text,
        marker,
        rf"\s*(\d+)\s+(\d+)\s+(-?\d+)\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*",
    )
    forces = _table(
        text,
        "Forces (Hartrees/Bohr)",
        rf"\s*(\d+)\s+(\d+)\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s*",
    )
    n = len(atoms)
    if (
        len(forces) != n
        or not np.array_equal(atoms[:, :2], forces[:, :2])
        or not np.array_equal(atoms[:, 0], np.arange(1, n + 1))
        or np.any(atoms[:, 1] < 1)
        or np.any(atoms[:, 1] > 118)
        or np.any(atoms[:, 2] != 0)
    ):
        raise ValueError("QM atom/force order mismatch or unsupported ghost/atom type")
    counts = re.findall(r"\bNAtoms=\s*(\d+)", text)
    if not counts or any(int(value) != n for value in counts):
        raise ValueError("NAtoms does not match orientation")
    scf = _number(
        _one(rf"SCF Done:\s+E\([^\n]+?\)\s*=\s*({NUMBER})", text, "SCF energy")
    )
    pc_lines = re.findall(r"^\s*XYZ=.*$", text, re.M)
    pcs = []
    for line in pc_lines:
        match = re.match(
            rf"\s*XYZ=\s*({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+Q=\s*({NUMBER})(.*)$",
            line,
        )
        if match is None:
            raise ValueError("Malformed point-charge row")
        for label in ("A", "R", "C"):
            extra = re.search(rf"\b{label}=\s*({NUMBER})", match[5])
            if extra and _number(extra[1]) != 0:
                raise ValueError(
                    "Nonzero MM non-electrostatic point-source parameters unsupported"
                )
        pcs.append([_number(v) for v in match.groups()[:4]])
    pc = np.asarray(pcs, dtype=float).reshape(-1, 4)
    nchg = re.findall(r"PtFMM for NChg=\s*(\d+)", text)
    if len(nchg) > 1 or (nchg and int(nchg[0]) < len(pc)):
        raise ValueError("Inconsistent printed point-charge count")
    if "Point Charges:" in text and not len(pc):
        raise ValueError("Point Charges block exists but has no readable sources")
    self_values = re.findall(rf"Self energy of the charges\s*=\s*({NUMBER})", text)
    if len(pc):
        if len(self_values) != 1:
            raise ValueError(
                "Embedded job must print exactly one point-charge self energy"
            )
        self_energy = _number(self_values[0])
    else:
        if self_values or "PtFMM for NChg=" in text:
            raise ValueError("Background diagnostics present but sources unavailable")
        self_energy = 0.0
    mm_energy = re.findall(rf"MM\s+energy of the charges\s*=\s*({NUMBER})", text)
    if any(_number(value) != 0 for value in mm_energy):
        raise ValueError(
            "Nonzero MM energy is outside the electrostatic dataset contract"
        )
    positions = atoms[:, 3:]
    f = forces[:, 2:] * FORCE_EV_ANGSTROM
    arrays = (positions, f, pc, np.array([scf, self_energy]))
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Nonfinite Gaussian values")
    minimum = (
        float(np.linalg.norm(positions[:, None] - pc[None, :, :3], axis=-1).min())
        if len(pc)
        else None
    )
    record = dict(
        atomic_numbers=atoms[:, 1].astype(np.int64),
        positions=positions,
        forces=f,
        energy=(scf - self_energy) * HARTREE_EV,
        mm_positions=pc[:, :3],
        mm_charges=pc[:, 3:],
        total_charge=int(charge),
        multiplicity=int(multiplicity),
        scf_energy_hartree=scf,
        pc_self_energy_hartree=self_energy,
        route=route,
        route_echo=route_echo,
        source_path=str(path.resolve()),
        min_qm_mm_distance=minimum,
        max_force_norm=float(np.linalg.norm(f, axis=1).max()),
        mm_total_charge=float(pc[:, 3].sum()),
        embedding="point_charges" if len(pc) else "vacuum",
    )
    record.update(_identity(path))
    record["printed_nchg"] = int(nchg[0]) if nchg else None
    # Filename variants are metadata, never a substitute for parsing the content.
    if record["family"] in ("samples", "samples_onlyprotein") and not len(pc):
        raise ValueError("Embedded filename family has no point charges")
    if record["family"] == "samples-no-background" and len(pc):
        raise ValueError("Vacuum filename family contains point charges")
    return record


def record_to_atoms(record):
    from ase import Atoms

    atoms = Atoms(
        numbers=record["atomic_numbers"], positions=record["positions"], pbc=False
    )
    atoms.arrays["REF_forces"] = record["forces"]
    atoms.info.update(
        {
            k: record[k]
            for k in ("total_charge", "multiplicity", "mm_positions", "mm_charges")
        }
    )
    atoms.info["REF_energy"] = record["energy"]
    return atoms


def _write_mace_configuration(group, record):
    """Stream the native save_configurations_as_HDF5 layout, without caching logs.

    Standard properties use canonical MACE names in HDF5. The extra point-source
    arrays reside in the same properties group and can be read by native MACE.
    """
    from mace.data import Configuration
    from mace.data.utils import write_value

    properties = {
        k: record[k]
        for k in (
            "energy",
            "forces",
            "total_charge",
            "mm_positions",
            "mm_charges",
            "multiplicity",
        )
    }
    properties["total_spin"] = record["multiplicity"]
    properties["mm_count"] = np.array([[len(record["mm_positions"])]], dtype=np.int64)
    config = Configuration(
        atomic_numbers=record["atomic_numbers"],
        positions=record["positions"],
        properties=properties,
        property_weights={"energy": 1.0, "forces": 1.0},
        pbc=(False, False, False),
        cell=np.zeros((3, 3)),
    )
    for name in ("atomic_numbers", "positions", "cell", "pbc", "weight", "config_type"):
        group[name] = write_value(getattr(config, name))
    for name in ("properties", "property_weights"):
        target = group.create_group(name)
        for key, value in getattr(config, name).items():
            target[key] = write_value(value)


def _parse_result(path):
    try:
        return parse_log(path), None
    except (ValueError, UnicodeError, OSError) as exc:
        return None, str(exc)


def _parsed_logs(paths, workers):
    if workers == 1:
        for path in paths:
            yield path, _parse_result(path)
        return
    # Spawn avoids inheriting the writer's open HDF5 handle. Keep only a bounded
    # number of results in flight; MM arrays can be large even for small QM graphs.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        source = iter(paths)
        pending = deque()
        for _ in range(2 * workers):
            path = next(source, None)
            if path is None:
                break
            pending.append((path, pool.submit(_parse_result, path)))
        while pending:
            path, future = pending.popleft()
            yield path, future.result()
            path = next(source, None)
            if path is not None:
                pending.append((path, pool.submit(_parse_result, path)))


def convert_logs(paths, output, skip_invalid=False, workers=1):
    """Stream files once, write atomically, refuse overwrite; retain rejection audit."""
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    output = Path(output)
    audit_path = Path(str(output) + ".audit.jsonl")
    if output.exists() or audit_path.exists():
        raise FileExistsError(output)
    if output.suffix not in (".h5", ".hdf5", ".xyz", ".extxyz"):
        raise ValueError("Output suffix must be h5/hdf5/xyz/extxyz")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".envhead-", dir=output.parent)
    os.close(fd)
    accepted, rejected, audit, ids = 0, 0, [], set()
    routes = set()
    sink = None
    try:
        if output.suffix in (".h5", ".hdf5"):
            import h5py

            sink = h5py.File(temp, "w")
            sink.attrs[
                "energy_convention"
            ] = "SCF minus printed PC-PC self energy; includes QM dispersion"
            frames = sink.create_group("config_batch_0")
        else:
            from ase.io import write

            sink = open(temp, "w", encoding="utf-8")
        for path, (record, error) in _parsed_logs(paths, workers):
            try:
                if error is not None:
                    raise ValueError(error)
                if record["sample_id"] in ids:
                    raise ValueError("Duplicate sample_id")
                # `charge` enables the external PC input. Vacuum and embedded
                # samples from the same electronic method belong together.
                method_route = " ".join(
                    re.sub(r"\bcharge\b", "", record["route"], flags=re.I)
                    .lower()
                    .split()
                )
                if routes and method_route not in routes:
                    raise ValueError("Mixed Gaussian routes; use separate output files")
            except (ValueError, UnicodeError) as exc:
                if not skip_invalid:
                    raise ValueError(f"{path}: {exc}") from exc
                rejected += 1
                audit.append(
                    dict(source_path=str(path), status="rejected", reason=str(exc))
                )
                continue
            routes.add(method_route)
            if output.suffix in (".h5", ".hdf5"):
                _write_mace_configuration(
                    frames.create_group(f"config_{accepted}"), record
                )
            else:
                write(
                    sink, record_to_atoms(record), format="extxyz", write_results=False
                )
            ids.add(record["sample_id"])
            accepted += 1
            audit.append(
                {
                    k: record[k]
                    for k in (
                        "source_path",
                    )
                }
            )
            audit[-1]["status"] = "accepted"
        if not accepted:
            raise ValueError("No valid Gaussian samples")
        if output.suffix in (".h5", ".hdf5"):
            sink.attrs.update(accepted=accepted, rejected=rejected)
        sink.close()
        sink = None
        # link is an atomic no-clobber publication on the same filesystem.
        os.link(temp, output)
        with audit_path.open("x", encoding="utf-8") as stream:
            for item in audit:
                stream.write(json.dumps(item, allow_nan=False) + "\n")
    finally:
        if sink is not None:
            sink.close()
        os.unlink(temp)
    return dict(
        accepted=accepted, rejected=rejected, output=str(output), audit=str(audit_path)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", help="Gaussian logs or directories searched recursively"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1, help="Parallel log parsers")
    parser.add_argument(
        "--skip-invalid", action="store_true", help="Record rejected jobs and continue"
    )
    args = parser.parse_args()
    paths = set()
    for item in args.inputs:
        path = Path(item)
        paths.update(path.rglob("*.log") if path.is_dir() else [path])
    print(
        json.dumps(
            convert_logs(
                sorted(paths),
                args.output,
                skip_invalid=args.skip_invalid,
                workers=args.workers,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
