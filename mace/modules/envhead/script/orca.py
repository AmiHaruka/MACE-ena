"""Convert ORCA single-point EnGrad outputs to MACE extxyz or HDF5.

Run by file path to avoid importing the MACE training stack for extxyz.
External point-charge files contain: count, then q x y z (e, Angstrom).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from functools import partial
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile

import numpy as np

HARTREE_EV = 27.211386245988
FORCE_EV_ANGSTROM = 51.4220674763
NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[DEde][+-]?\d+)?"


def _number(value):
    return float(value.replace("D", "E").replace("d", "e"))


def _unique(pattern, text, label):
    matches = re.findall(pattern, text, re.M | re.I)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {label}, found {len(matches)}; only single-point jobs supported"
        )
    return matches[0]


def _section(text, marker):
    parts = text.split(marker)
    if len(parts) != 2:
        raise ValueError(f"Expected one {marker} section")
    return parts[1].lstrip("- \n\r").split("\n\n", 1)[0]


def _pc_path(path, reference, pc_dir, symbols, positions):
    # Renamed flat datasets need a matching prefixed companion. Never search by
    # frame number recursively: different amino acids reuse those filenames.
    renamed = path.with_name(re.sub(r"_pc$", "", path.stem) + ".pc")
    candidates = [renamed, path.parent / reference]
    if pc_dir is not None:
        root = Path(pc_dir)
        match = re.fullmatch(r"(Unit-.+)_frame_(\d+)_pc", path.stem)
        mapped = []
        if match:
            mapped = [root / match[1] / "orca_qmmm" / Path(reference).name]
        candidates = mapped + [root / renamed.name, root / reference] + candidates
    for candidate in candidates:
        if candidate.is_file():
            xyz = candidate.with_name(candidate.stem + "_qm.xyz")
            if xyz.is_file():
                from ase.io import read

                atoms = read(xyz)
                if atoms.get_chemical_symbols() != symbols or not np.allclose(
                    atoms.positions, positions, atol=2e-6, rtol=0
                ):
                    raise ValueError(
                        f"Point-charge companion QM geometry mismatch: {xyz}"
                    )
            return candidate
    raise ValueError(
        f"Missing point-charge file {reference!r}; supply {renamed.name} beside output or via --pc-dir"
    )


def parse_out(path, pc_dir=None):
    from ase.data import atomic_numbers

    path = Path(path)
    # Some collected ORCA logs contain NUL padding between output blocks.
    text = path.read_text(encoding="utf-8").replace("\x00", "")
    if text.count("ORCA TERMINATED NORMALLY") != 1:
        raise ValueError("Missing or repeated normal termination")
    inputs = "\n".join(re.findall(r"^\s*\|\s*\d+>\s?(.*)$", text, re.M))
    if re.search(r"\b(DoEQ|Opt|OptTS|Scan|MD|CPCM|SMD)\b", inputs, re.I):
        raise ValueError(
            "Unsupported energy convention/job (DoEQ, optimization or continuum solvent)"
        )
    charge = int(
        _unique(r"Total Charge\s+Charge\s+\.\.\.\.\s+([+-]?\d+)", text, "total charge")
    )
    mult = int(_unique(r"Multiplicity\s+Mult\s+\.\.\.\.\s+(\d+)", text, "multiplicity"))
    energy = _number(
        _unique(
            r"^\s*FINAL SINGLE POINT ENERGY\s+(" + NUMBER + r")\s*$",
            text,
            "final energy",
        )
    )
    coord = _section(text, "CARTESIAN COORDINATES (ANGSTROEM)")
    rows = [line.split() for line in coord.splitlines() if line.strip()]
    if not rows or any(len(row) != 4 or row[0] not in atomic_numbers for row in rows):
        raise ValueError("Invalid Cartesian coordinate table")
    symbols = [row[0] for row in rows]
    positions = np.array([[_number(x) for x in row[1:]] for row in rows])
    grad = _section(text, "\nCARTESIAN GRADIENT\n")
    rows = re.findall(
        r"^\s*(\d+)\s+(\w+)\s*:\s*("
        + NUMBER
        + r")\s+("
        + NUMBER
        + r")\s+("
        + NUMBER
        + r")\s*$",
        grad,
        re.M,
    )
    if len(rows) != len(symbols) or [r[1] for r in rows] != symbols:
        raise ValueError("Gradient/coordinate atom mismatch")
    indices = [int(r[0]) for r in rows]
    if indices not in [list(range(len(rows))), list(range(1, len(rows) + 1))]:
        raise ValueError("Nonsequential gradient atom indices")
    forces = -np.array([[_number(x) for x in r[2:]] for r in rows]) * FORCE_EV_ANGSTROM
    mm = np.empty((0, 4))
    references = re.findall(r'%pointcharges\s+["\']([^"\']+)["\']', inputs, re.I)
    counts = re.findall(r"ok\s*\((\d+) point charges\)", text)
    if references:
        if len(references) != 1:
            raise ValueError("Multiple point-charge declarations")
        pc_path = _pc_path(path, references[0], pc_dir, symbols, positions)
        lines = [
            line.strip() for line in pc_path.read_text().splitlines() if line.strip()
        ]
        n = int(lines[0])
        mm = np.array(
            [[_number(x) for x in line.split()] for line in lines[1:]]
        ).reshape(-1, 4)
        if len(mm) != n or any(int(count) != n for count in counts):
            raise ValueError("Point-charge count mismatch")
    elif (
        counts or re.search(r"%pointcharges", inputs, re.I) or path.stem.endswith("_pc")
    ):
        raise ValueError("Cannot resolve external point-charge declaration")
    record = dict(
        atomic_numbers=np.array([atomic_numbers[s] for s in symbols]),
        positions=positions,
        energy=energy * HARTREE_EV,
        forces=forces,
        total_charge=charge,
        multiplicity=mult,
        mm_positions=mm[:, 1:],
        mm_charges=mm[:, :1],
    )
    if "MULLIKEN ATOMIC CHARGES" in text:
        section = text.rsplit("MULLIKEN ATOMIC CHARGES", 1)[1]
        rows = re.findall(
            r"^\s*(\d+)\s+(\w+)\s*:\s*(" + NUMBER + r")(?:\s|$)",
            section.split("Sum of atomic charges", 1)[0],
            re.M,
        )
        rows = rows[: len(symbols)]
        if len(rows) != len(symbols) or [r[1] for r in rows] != symbols:
            raise ValueError("Mulliken/coordinate atom mismatch")
        q = np.array([_number(r[2]) for r in rows])
        if not np.isclose(q.sum(), charge, atol=max(1e-5, len(q) * 1e-6), rtol=0):
            raise ValueError("Mulliken charges do not sum to total_charge")
        record["charges"] = q
    if len(mm):
        # Explicit element sidecar takes precedence; trajectory metadata retains
        # the original PC order. Never infer species from partial charges.
        sidecar = pc_path.with_suffix(".elements.json")
        metadata = pc_path.parent.parent / "params.json"
        if sidecar.exists():
            elements = json.loads(sidecar.read_text())
            record["mm_atomic_numbers"] = np.array(
                [atomic_numbers[z] if isinstance(z, str) else z for z in elements],
                dtype=int,
            )
        elif metadata.exists():
            from ase.data import atomic_masses

            atoms = [
                a
                for a in json.loads(metadata.read_text())["atoms"]
                if a["category"] != "protein"
            ]
            if len(atoms) != len(mm) or not np.allclose(
                [a["charge_used"] for a in atoms], mm[:, 0], atol=1e-5
            ):
                raise ValueError("PC/trajectory metadata order mismatch")
            masses = np.array([a["mass"] for a in atoms])
            z = np.abs(masses[:, None] - atomic_masses[None, 1:]).argmin(1) + 1
            if not np.allclose(atomic_masses[z], masses, atol=0.15):
                raise ValueError(
                    "Ambiguous environment element mass; supply .elements.json"
                )
            record["mm_atomic_numbers"] = z
    else:
        record["mm_atomic_numbers"] = np.empty(0, dtype=int)
    if not all(np.isfinite(value).all() for value in record.values()):
        raise ValueError("Nonfinite data")
    record["sample_id"] = path.stem
    record["pair_id"] = re.sub(r"_(gas|pc)$", "", path.stem)
    record["group_id"] = (
        path.stem.split("_frame_", 1)[0] if "_frame_" in path.stem else ""
    )
    return record


def _parse(path, pc_dir):
    try:
        return parse_out(path, pc_dir), None
    except (ValueError, OSError, UnicodeError, KeyError) as exc:
        return None, str(exc)


def _records(paths, workers, pc_dir):
    parse = partial(_parse, pc_dir=pc_dir)
    if workers == 1:
        for path in paths:
            yield path, parse(path)
        return
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        source = iter(paths)
        pending = deque()
        for path in source:
            pending.append((path, pool.submit(parse, path)))
            if len(pending) >= 2 * workers:
                first, result = pending.popleft()
                yield first, result.result()
        while pending:
            first, result = pending.popleft()
            yield first, result.result()


def _write_h5(group, record):
    # Match native MACE Configuration HDF5 layout without importing MACE/torch.
    for key, value in dict(
        atomic_numbers=record["atomic_numbers"],
        positions=record["positions"],
        cell=np.zeros((3, 3)),
        pbc=np.zeros(3, dtype=bool),
        weight=1.0,
        config_type="Default",
    ).items():
        group[key] = value
    properties = group.create_group("properties")
    for key, value in record.items():
        if key not in ("atomic_numbers", "positions"):
            properties[key] = value
    properties["total_spin"] = record["multiplicity"]
    properties["mm_count"] = np.array([[len(record["mm_positions"])]])
    weights = group.create_group("property_weights")
    weights["energy"] = 1.0
    weights["forces"] = 1.0
    weights["charges"] = float("charges" in record)


def convert_outs(paths, output, skip_invalid=False, workers=1, pc_dir=None):
    from ase import Atoms
    from ase.io import write

    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output)
    audit = Path(str(output) + ".audit.jsonl")
    if output.exists() or audit.exists():
        raise FileExistsError(output)
    h5 = output.suffix.lower() in (".h5", ".hdf5")
    if not h5 and output.suffix.lower() not in (".xyz", ".extxyz"):
        raise ValueError("Output must be extxyz/xyz/h5/hdf5")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, prefix=".orca-")
    os.close(fd)
    accepted = rejected = 0
    try:
        if h5:
            import h5py

            sink = h5py.File(temporary, "w")
            frames = sink.create_group("config_batch_0")
        else:
            sink = open(temporary, "w", encoding="utf-8")
        with sink, audit.open("x", encoding="utf-8") as log:
            for path, (record, error) in _records(paths, workers, pc_dir):
                log.write(
                    json.dumps(
                        dict(
                            source_path=str(path),
                            status="rejected" if error else "accepted",
                            **({"reason": error} if error else {}),
                        )
                    )
                    + "\n"
                )
                if error:
                    rejected += 1
                    if not skip_invalid:
                        raise ValueError(f"{path}: {error}")
                    continue
                if h5:
                    _write_h5(frames.create_group(f"config_{accepted}"), record)
                else:
                    atoms = Atoms(
                        numbers=record["atomic_numbers"],
                        positions=record["positions"],
                        pbc=False,
                    )
                    atoms.arrays["REF_forces"] = record["forces"]
                    if "charges" in record:
                        atoms.arrays["REF_charges"] = record["charges"]
                    atoms.info.update(
                        REF_energy=record["energy"],
                        total_charge=record["total_charge"],
                        multiplicity=record["multiplicity"],
                        sample_id=record["sample_id"],
                        pair_id=record["pair_id"],
                    )
                    if record["group_id"]:
                        atoms.info["group_id"] = record["group_id"]
                    for key in ("mm_positions", "mm_charges"):
                        atoms.info[key] = "_JSON " + json.dumps(record[key].tolist())
                    if "mm_atomic_numbers" in record:
                        atoms.info["mm_atomic_numbers"] = "_JSON " + json.dumps(
                            record["mm_atomic_numbers"].tolist()
                        )
                    write(sink, atoms, format="extxyz", write_results=False)
                accepted += 1
        if not accepted:
            raise ValueError(f"No valid ORCA samples; see {audit}")
        os.link(temporary, output)
    finally:
        os.unlink(temporary)
    return dict(
        accepted=accepted, rejected=rejected, output=str(output), audit=str(audit)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--pc-dir",
        type=Path,
        help="Companion directory or Traj root containing Unit-*/orca_qmmm",
    )
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args()
    paths = set()
    for item in args.inputs:
        path = Path(item)
        paths.update(path.rglob("*.out") if path.is_dir() else [path])
    print(
        json.dumps(
            convert_outs(
                sorted(paths), args.output, args.skip_invalid, args.workers, args.pc_dir
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
