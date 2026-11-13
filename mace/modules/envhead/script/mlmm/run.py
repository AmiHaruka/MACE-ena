"""JSON-driven OpenMM dynamics for the optional periodic EnvHead interface."""

import argparse
import json
import random
import shutil
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Stage:
    name: str
    ensemble: str
    steps: int


def _steps(duration_ps, timestep_fs):
    value = float(duration_ps) * 1000 / float(timestep_fs)
    result = round(value)
    if (
        value < 0
        or not np.isfinite(value)
        or not np.isclose(value, result, atol=1e-8, rtol=0)
    ):
        raise ValueError(
            "Stage durations must be nonnegative integer multiples of timestep_fs."
        )
    return result


def default_config():
    """Fresh MD defaults; system-specific inputs must be supplied by the user."""
    return {
        "mode": "mlmm-em",
        "model": None,
        "pdb": None,
        "system_xml": None,
        "plumed": None,
        "output": "md",
        "device": "cuda",
        "dtype": "float32",
        "enable_cueq": True,
        "platform": "CUDA",
        "platform_properties": {"Precision": "mixed"},
        "forcefield": ["amber19/protein.ff19SB.xml", "amber19/spce.xml"],
        "solvated": False,
        "water_model": "spce",
        "padding_nm": 1.2,
        "box_nm": None,
        "neutralize": True,
        "ionic_strength_molar": 0.0,
        "ml_indices": [],
        "total_charge": 0.0,
        "keep_ml_constraints": False,
        "nonbonded_cutoff_nm": 1.0,
        "switch_distance_nm": 0.9,
        "ewald_error_tolerance": 1e-4,
        "dispersion_correction": True,
        "timestep_fs": 0.5,
        "temperature_k": 300.0,
        "pressure_bar": 1.0,
        "friction_ps": 1.0,
        "barostat_interval": 25,
        "constraint_tolerance": 1e-6,
        "seed": 114514,
        "minimize": True,
        "minimize_tolerance": 10.0,
        "minimize_max_iterations": 1000,
        "nvt_ps": 10.0,
        "npt_ps": 20.0,
        "ensemble": "npt",
        "production_ps": 100.0,
        "report_interval": 200,
        "checkpoint_interval": 2000,
    }


def load_config(path):
    """Read usage controls, resolving all user file paths beside the JSON file."""
    path = Path(path).expanduser().resolve()
    defaults = default_config()
    supplied = json.loads(path.read_text())
    if not isinstance(supplied, dict):
        raise ValueError("MD configuration must be a JSON object.")
    unknown = set(supplied) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown MD configuration keys: {', '.join(sorted(unknown))}")
    config = dict(defaults, **supplied)
    for name in ("model", "pdb"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"Set {name} in the MD configuration before running.")
    if config["mode"] not in ("mlmm-em", "mlmm-mm"):
        raise ValueError("mode must be mlmm-em or mlmm-mm.")
    for name in ("model", "pdb", "system_xml", "output", "plumed"):
        if config[name] is not None:
            config[name] = str(
                (path.parent / Path(config[name]).expanduser()).resolve()
            )
    if (
        not isinstance(config["forcefield"], list)
        or not config["forcefield"]
        or any(not isinstance(name, str) for name in config["forcefield"])
    ):
        raise ValueError("forcefield must list native OpenMM or local XML filenames.")
    config["forcefield"] = [
        str((path.parent / name).resolve()) if (path.parent / name).is_file() else name
        for name in config["forcefield"]
    ]
    if config["ensemble"] not in {"nve", "nvt", "npt"}:
        raise ValueError("ensemble must be nve, nvt or npt.")
    if config["dtype"] not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64.")
    for name in (
        "minimize",
        "enable_cueq",
        "dispersion_correction",
        "solvated",
        "neutralize",
        "keep_ml_constraints",
    ):
        if type(config[name]) is not bool:
            raise TypeError(f"{name} must be a JSON boolean.")
    for name in (
        "timestep_fs",
        "temperature_k",
        "pressure_bar",
        "friction_ps",
        "minimize_tolerance",
        "constraint_tolerance",
        "nonbonded_cutoff_nm",
        "ewald_error_tolerance",
        "padding_nm",
    ):
        if (
            not isinstance(config[name], (int, float))
            or isinstance(config[name], bool)
            or not np.isfinite(config[name])
            or config[name] <= 0
        ):
            raise ValueError(f"{name} must be finite and positive.")
    for name in ("report_interval", "checkpoint_interval", "barostat_interval", "seed"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if (
        type(config["minimize_max_iterations"]) is not int
        or config["minimize_max_iterations"] < 0
    ):
        raise ValueError("minimize_max_iterations must be a nonnegative integer.")
    if config["box_nm"] is not None:
        box = np.asarray(config["box_nm"], dtype=float)
        if box.shape != (3,) or not np.isfinite(box).all() or np.any(box <= 0):
            raise ValueError(
                "box_nm must contain three positive orthorhombic box lengths."
            )
    if not isinstance(config["water_model"], str) or not config["water_model"].strip():
        raise ValueError("water_model must name an OpenMM solvent template.")
    salt = config["ionic_strength_molar"]
    if (
        not isinstance(salt, (int, float))
        or isinstance(salt, bool)
        or not np.isfinite(salt)
        or salt < 0
    ):
        raise ValueError("ionic_strength_molar must be finite and nonnegative.")
    if not 0 <= config["switch_distance_nm"] < config["nonbonded_cutoff_nm"]:
        raise ValueError(
            "switch_distance_nm must be nonnegative and below nonbonded_cutoff_nm."
        )
    indices = config["ml_indices"]
    if (
        not isinstance(indices, list)
        or not indices
        or any(type(i) is not int or i < 0 for i in indices)
        or len(indices) != len(set(indices))
    ):
        raise ValueError(
            "ml_indices must be a nonempty list of unique nonnegative integers."
        )
    if not np.isfinite(config["total_charge"]):
        raise ValueError("total_charge must be finite.")
    if not isinstance(config["platform_properties"], dict):
        raise ValueError(
            "platform_properties must be an object of OpenMM platform properties."
        )
    stage_plan(config)
    return config


def stage_plan(config, *, resume=False):
    """Return equilibration and production stages with exact integer step counts."""
    result = []
    if not resume:
        for ensemble in ("nvt", "npt"):
            steps = _steps(config[f"{ensemble}_ps"], config["timestep_fs"])
            if steps:
                result.append(Stage(ensemble, ensemble, steps))
    result.append(
        Stage(
            "production",
            config["ensemble"],
            _steps(config["production_ps"], config["timestep_fs"]),
        )
    )
    return result


def remaining_steps(config, current_step, production_start):
    """production_ps is the total production target, including a resumed segment."""
    if current_step < production_start:
        raise ValueError("Checkpoint step precedes the recorded production start.")
    target = _steps(config["production_ps"], config["timestep_fs"])
    return max(0, target - (current_step - production_start))


def _set_topology_box(topology, vectors):
    # System returns a list of unitful Vec3 objects in some OpenMM versions;
    # Topology expects one unitful sequence, otherwise it adds a second unit.
    from openmm import unit

    topology.setPeriodicBoxVectors(
        unit.Quantity(
            [v.value_in_unit(unit.nanometer) for v in vectors], unit.nanometer
        )
    )


def build_input(config):
    """Reuse a native System, parameterize a wet PDB, or solvate a dry solute."""
    import openmm
    from openmm import app, unit

    pdb = app.PDBFile(config["pdb"])
    if config["system_xml"]:
        system = openmm.XmlSerializer.deserialize(
            Path(config["system_xml"]).read_text()
        )
        if system.getNumParticles() != pdb.topology.getNumAtoms():
            raise ValueError("system_xml and pdb contain different particle counts.")
        _set_topology_box(pdb.topology, system.getDefaultPeriodicBoxVectors())
        return system, pdb.topology, pdb.positions
    forcefield = app.ForceField(*config["forcefield"])
    modeller = app.Modeller(pdb.topology, pdb.positions)
    if config["solvated"]:
        if modeller.topology.getPeriodicBoxVectors() is None:
            raise ValueError(
                "A solvated PDB requires periodic box vectors (CRYST1); "
                "provide its box in the PDB or use system_xml."
            )
    else:
        if any(r.name in {"HOH", "WAT", "SOL"} for r in modeller.topology.residues()):
            raise ValueError(
                "PDB already contains water; set solvated=true to keep it."
            )
        modeller.topology.setPeriodicBoxVectors(None)
        box = (
            {"boxSize": openmm.Vec3(*config["box_nm"]) * unit.nanometer}
            if config["box_nm"] is not None
            else {"padding": config["padding_nm"] * unit.nanometer}
        )
        random.seed(config["seed"])
        modeller.addSolvent(
            forcefield,
            model=config["water_model"],
            neutralize=config["neutralize"],
            ionicStrength=config["ionic_strength_molar"] * unit.molar,
            **box,
        )
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=config["nonbonded_cutoff_nm"] * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=config["ewald_error_tolerance"],
    )
    nb = next(
        force for force in system.getForces() if type(force) is openmm.NonbondedForce
    )
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(config["switch_distance_nm"] * unit.nanometer)
    nb.setUseDispersionCorrection(config["dispersion_correction"])
    return system, modeller.topology, modeller.positions


def add_barostat(system, config):
    """Attach after classical preparation; frequency zero disables volume moves."""
    import openmm
    from openmm import unit

    barostat = openmm.MonteCarloBarostat(
        config["pressure_bar"] * unit.bar,
        config["temperature_k"] * unit.kelvin,
        config["barostat_interval"],
    )
    barostat.setRandomNumberSeed(config["seed"] + 1)
    system.addForce(barostat)
    return barostat


def make_simulation(topology, system, ensemble, config, *, state=None, step=0):
    """Create a Context for one ensemble, optionally transferring an earlier State."""
    import openmm
    from openmm import app, unit

    if ensemble not in {"nve", "nvt", "npt"}:
        raise ValueError("ensemble must be nve, nvt or npt.")
    for force in system.getForces():
        if type(force) is openmm.MonteCarloBarostat:
            force.setFrequency(config["barostat_interval"] if ensemble == "npt" else 0)
    timestep = config["timestep_fs"] * unit.femtosecond
    if ensemble == "nve":
        integrator = openmm.VerletIntegrator(timestep)
    else:
        integrator = openmm.LangevinMiddleIntegrator(
            config["temperature_k"] * unit.kelvin,
            config["friction_ps"] / unit.picosecond,
            timestep,
        )
        integrator.setRandomNumberSeed(config["seed"])
    integrator.setConstraintTolerance(config["constraint_tolerance"])
    platform = openmm.Platform.getPlatformByName(config["platform"])
    simulation = app.Simulation(
        topology, system, integrator, platform, config["platform_properties"]
    )
    if state is not None:
        simulation.context.setState(state)
        simulation.context.setStepCount(step)
    return simulation


def initialize_velocities(context, system, config):
    """Sample constrained thermal velocities with zero total linear momentum."""
    from openmm import unit

    context.setVelocitiesToTemperature(
        config["temperature_k"] * unit.kelvin, config["seed"]
    )
    context.applyVelocityConstraints(config["constraint_tolerance"])
    masses = np.array(
        [
            system.getParticleMass(index).value_in_unit(unit.dalton)
            for index in range(system.getNumParticles())
        ]
    )
    if masses.sum() <= 0:
        raise ValueError(
            "Velocity initialization requires positive total particle mass."
        )
    velocity_unit = unit.nanometer / unit.picosecond
    velocities = (
        context.getState(getVelocities=True)
        .getVelocities(asNumpy=True)
        .value_in_unit(velocity_unit)
    )
    velocities -= (masses[:, None] * velocities).sum(axis=0) / masses.sum()
    # Uniform translation preserves every pairwise velocity constraint.
    context.setVelocities(velocities * velocity_unit)


def _load_model(config):
    import torch

    from ...backends import enable_cueq
    from ...train.checkpoint import load_model

    model = load_model(config["model"], device=config["device"])
    model.to(device=config["device"], dtype=getattr(torch, config["dtype"]))
    if config["enable_cueq"]:
        enable_cueq(model)
    model.requires_grad_(False)
    return model.eval()


def _write_pdb(path, topology, state_or_positions, box=None):
    from openmm import app

    if box is not None:
        _set_topology_box(topology, box)
    with open(path, "w") as handle:
        app.PDBFile.writeFile(topology, state_or_positions, handle, keepIds=True)


def _dcd_header(path):
    """Read native OpenMM DCD timing and reject an incomplete trailing frame."""
    with path.open("rb") as handle:
        header = handle.read(96)
        if (
            len(header) != 96
            or header[4:8] != b"CORD"
            or struct.unpack_from("<i", header)[0] != 84
            or struct.unpack_from("<i", header, 88)[0] != 84
        ):
            raise ValueError("Invalid OpenMM DCD header.")
        frames, first, interval, last = struct.unpack_from("<4i", header, 8)
        comments = struct.unpack_from("<i", header, 92)[0]
        box_flag = struct.unpack_from("<i", header, 48)[0]
        if (
            frames < 0
            or first < 0
            or interval <= 0
            or comments < 4
            or box_flag not in (0, 1)
        ):
            raise ValueError("Invalid OpenMM DCD timing or box fields.")
        handle.seek(96 + comments)
        ending = handle.read(16)
        if len(ending) != 16:
            raise ValueError("Incomplete OpenMM DCD header.")
        comment_end, atom_start, atoms, atom_end = struct.unpack("<4i", ending)
        frame_bytes = 24 + 12 * atoms + 56 * box_flag
        if (
            (comment_end, atom_start, atom_end) != (comments, 4, 4)
            or atoms <= 0
            or path.stat().st_size != 112 + comments + frames * frame_bytes
        ):
            raise ValueError("Incomplete or inconsistent OpenMM DCD frames.")
        if frames and last != first + (frames - 1) * interval:
            raise ValueError("Inconsistent OpenMM DCD final step.")
    return dict(frames=frames, first_step=first, interval=interval, last_step=last)


def _last_log_step(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 65536))
        tail = handle.read()
    if not tail.endswith(b"\n"):
        raise ValueError("Incomplete production log line.")
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if line and not line.startswith(b"#"):
            return int(line.split(None, 1)[0])
    return None


def _resume_output(output, config, current_step, *, same_directory):
    """Append a continuous pair of outputs, or retain them and start a segment."""
    segments = [(0, "md")]
    if same_directory:
        for path in output.glob("md_resume_*"):
            step = path.stem.removeprefix("md_resume_")
            if (
                path.suffix in {".dcd", ".log"}
                and step.isdigit()
                and int(step) <= current_step
            ):
                segments.append((int(step), path.stem))
    stem = max(segments)[1]
    dcd, log = output / f"{stem}.dcd", output / f"{stem}.log"

    def nonempty(path):
        return path.exists() and path.stat().st_size > 0

    if not same_directory:
        replaced = (
            dcd,
            log,
            *(
                output / name
                for name in (
                    "effective_config.json",
                    "classical_input.xml",
                    "initial.pdb",
                    "checkpoint.chk",
                    "final_state.xml",
                    "final.pdb",
                )
            ),
        )
        if any(nonempty(path) for path in replaced):
            raise FileExistsError(
                "Resume output directory contains nonempty run files that would be replaced; choose another output directory."
            )
        if (
            config.get("plumed")
            and (output / "plumed").is_dir()
            and any((output / "plumed").iterdir())
        ):
            raise FileExistsError("Resume output directory contains PLUMED files.")
        return stem, False
    try:
        last_log = _last_log_step(log)
        if not nonempty(dcd) and last_log is None:
            return stem, False
        header = _dcd_header(dcd)
        if header["frames"] == 0 and last_log is None:
            return stem, False
        interval = config["report_interval"]
        if (
            header["interval"] == interval
            and header["frames"] > 0
            and header["last_step"] == last_log
            and header["last_step"] == current_step - current_step % interval
        ):
            return stem, True
    except (OSError, ValueError, struct.error):
        pass
    segment = f"md_resume_{current_step}"
    if any(nonempty(output / f"{segment}.{extension}") for extension in ("dcd", "log")):
        raise FileExistsError(
            f"Resume segment {segment} already contains output; choose another output directory to preserve it."
        )
    return segment, False


def _reporters(
    simulation,
    output,
    stage,
    config,
    *,
    append=False,
    stem=None,
    bias=None,
):
    from openmm import app

    class GlobalStepDCDReporter(app.DCDReporter):
        def report(self, simulation, state):
            if self._dcd is None:
                first = (
                    _dcd_header(output / f"{stem}.dcd")["first_step"]
                    if self._append
                    else simulation.currentStep
                )
                self._dcd = app.DCDFile(
                    self._out,
                    simulation.topology,
                    simulation.integrator.getStepSize(),
                    firstStep=first,
                    interval=self._reportInterval,
                    append=self._append,
                )
            super().report(simulation, state)

    simulation.reporters.clear()
    stem = stem or ("md" if stage.name == "production" else stage.name)
    simulation.reporters.append(
        app.StateDataReporter(
            str(output / f"{stem}.log"),
            config["report_interval"],
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            separator="\t",
            append=append,
        )
    )
    simulation.reporters.append(
        GlobalStepDCDReporter(
            str(output / f"{stem}.dcd"),
            config["report_interval"],
            append=append,
            enforcePeriodicBox=True,
        )
    )
    if stage.name == "production":
        simulation.reporters.append(
            bias.reporter(output / "checkpoint.chk")
            if bias is not None
            else app.CheckpointReporter(
                str(output / "checkpoint.chk"), config["checkpoint_interval"]
            )
        )


def run(config, *, resume=None):
    """Run minimization, equilibration and one production ensemble with one model."""
    import openmm
    from openmm import app, unit

    from .bridge import create_mlmm_system

    output = Path(config["output"]).resolve()
    resume = Path(resume).expanduser().resolve() if resume else None
    if resume and output != resume.parent:
        # Preserve existing output before copying the source run's inputs.
        _resume_output(output, config, 0, same_directory=False)
    bias = None
    if config.get("plumed"):
        from .plumed import PlumedBias

        bias = PlumedBias(
            config["plumed"],
            output,
            config["temperature_k"],
            config["checkpoint_interval"],
        )
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        previous = json.loads((resume.parent / "effective_config.json").read_text())
        if previous.get("mode", "mlmm-em") != config["mode"]:
            raise ValueError("Restart mode differs from the checkpoint.")
        if previous["timestep_fs"] != config["timestep_fs"]:
            raise ValueError("Restart timestep_fs differs from the checkpoint run.")
        if previous.get("keep_ml_constraints", False) != config["keep_ml_constraints"]:
            raise ValueError("Restart ML constraint setting differs from the checkpoint.")
        # Retain the original equilibration duration when extending production,
        # including when the user clears the already-completed stages in JSON.
        config = dict(config, nvt_ps=previous["nvt_ps"], npt_ps=previous["npt_ps"])
        source_path, topology_path = (
            resume.parent / "classical_input.xml",
            resume.parent / "initial.pdb",
        )
        source = openmm.XmlSerializer.deserialize(source_path.read_text())
        pdb = app.PDBFile(str(topology_path))
        topology, positions = pdb.topology, pdb.positions
        _set_topology_box(topology, source.getDefaultPeriodicBoxVectors())
    else:
        if (output / "initial.pdb").exists() or (output / "checkpoint.chk").exists():
            raise FileExistsError(
                "Output contains an earlier run; use --resume or choose another output directory."
            )
        source, topology, positions = build_input(config)
        (output / "classical_input.xml").write_text(
            openmm.XmlSerializer.serialize(source)
        )
        _write_pdb(
            output / "initial.pdb",
            topology,
            positions,
            source.getDefaultPeriodicBoxVectors(),
        )
    model = _load_model(config)
    system, callback = create_mlmm_system(
        source,
        topology,
        config["ml_indices"],
        model,
        config["total_charge"],
        mode=config.get("mode", "mlmm-em"),
        keep_ml_constraints=config["keep_ml_constraints"],
    )
    print(f"ML/MM mode: {callback.mode}", flush=True)
    add_barostat(system, config)
    stages = stage_plan(config, resume=bool(resume))
    scope = ExitStack()
    simulation = None
    try:
        if bias is not None:
            if resume and output != resume.parent:
                shutil.copytree(resume.parent / "plumed", bias.path)
            scope.enter_context(bias.directory())
            if resume:
                bias.attach(system, restart=True)
        simulation = make_simulation(topology, system, stages[0].ensemble, config)
        if resume:
            simulation.loadCheckpoint(str(resume))
            if bias is not None:
                bias.check_restart(
                    simulation.context.getState()
                    .getTime()
                    .value_in_unit(unit.picosecond)
                )
        else:
            simulation.context.setPositions(positions)
            simulation.context.applyConstraints(config["constraint_tolerance"])
            if config["minimize"]:
                simulation.minimizeEnergy(
                    tolerance=config["minimize_tolerance"]
                    * unit.kilojoule_per_mole
                    / unit.nanometer,
                    maxIterations=config["minimize_max_iterations"],
                )
                state = simulation.context.getState(getPositions=True)
                _write_pdb(
                    output / "minimized.pdb",
                    topology,
                    state.getPositions(),
                    state.getPeriodicBoxVectors(),
                )
            initialize_velocities(simulation.context, system, config)
        current_ensemble = stages[0].ensemble
        (output / "effective_config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )
        if resume and output != resume.parent:
            (output / "classical_input.xml").write_text(source_path.read_text())
            (output / "initial.pdb").write_text(topology_path.read_text())
        production_start = sum(
            item.steps for item in stage_plan(config) if item.name != "production"
        )
        for stage in stages:
            start_bias = (
                stage.name == "production"
                and bias is not None
                and bias.force_group is None
            )
            if stage.ensemble != current_ensemble or start_bias:
                state = simulation.context.getState(
                    getPositions=True, getVelocities=True, getParameters=True
                )
                step = simulation.currentStep
                previous = simulation
                simulation = None
                previous.reporters.clear()
                del previous.context
                del previous
                if start_bias:
                    bias.attach(system)
                simulation = make_simulation(
                    topology, system, stage.ensemble, config, state=state, step=step
                )
                current_ensemble = stage.ensemble
            if stage.name == "production":
                stem, append = (
                    _resume_output(
                        output,
                        config,
                        simulation.currentStep,
                        same_directory=True,
                    )
                    if resume and output == resume.parent
                    else ("md", False)
                )
                steps = remaining_steps(
                    config, simulation.currentStep, production_start
                )
            else:
                steps = stage.steps
                stem, append = stage.name, False
            _reporters(
                simulation,
                output,
                stage,
                config,
                append=append,
                stem=stem,
                bias=bias,
            )
            print(f"{stage.name}: {stage.ensemble.upper()}, {steps} steps", flush=True)
            simulation.step(steps)
        if bias is not None:
            bias.save(simulation, output / "checkpoint.chk")
        else:
            simulation.saveCheckpoint(str(output / "checkpoint.chk"))
        simulation.saveState(str(output / "final_state.xml"))
        state = simulation.context.getState(getPositions=True)
        _write_pdb(
            output / "final.pdb",
            topology,
            state.getPositions(),
            state.getPeriodicBoxVectors(),
        )
        return {
            "steps": simulation.currentStep,
            "time_ps": state.getTime().value_in_unit(unit.picosecond),
            "output": str(output),
        }
    finally:
        # The callback and its one Torch model remain alive until the Context
        # has been released, including when minimization/integration fails.
        if simulation is not None:
            simulation.reporters.clear()
            del simulation.context
        del simulation, callback
        scope.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    entry = parser.add_mutually_exclusive_group(required=True)
    entry.add_argument(
        "--initial",
        action="store_true",
        help="Write a generic md.json in the current directory without starting MD.",
    )
    entry.add_argument("--config", type=Path)
    parser.add_argument(
        "--mode",
        choices=("mlmm-em", "mlmm-mm"),
        help="Embedding mode; overrides JSON (default: mlmm-em).",
    )
    parser.add_argument(
        "--plumed",
        type=Path,
        help="PLUMED input for production; overrides the JSON path.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a production checkpoint; production_ps is the total target.",
    )
    args = parser.parse_args(argv)
    if args.initial:
        if args.resume is not None or args.mode is not None or args.plumed is not None:
            parser.error("--initial cannot be combined with run options.")
        try:
            with Path("md.json").open("x") as stream:
                stream.write(json.dumps(default_config(), indent=2) + "\n")
        except FileExistsError:
            parser.error("md.json already exists; the existing file was preserved.")
        print("Created md.json; set model, pdb and ml_indices before running.")
        return 0
    config = load_config(args.config)
    if args.mode is not None:
        config["mode"] = args.mode
    if args.plumed is not None:
        config["plumed"] = str(args.plumed.expanduser().resolve())
    result = run(config, resume=args.resume)
    print(
        f"Finished {result['steps']} total steps, {result['time_ps']:.6f} ps; {result['output']}"
    )
    return 0
