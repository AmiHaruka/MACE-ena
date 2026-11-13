"""Optional PLUMED force with native checkpoint and working-directory restart."""

import math
import os
from contextlib import contextmanager
from pathlib import Path


class PlumedBias:
    """Keep PLUMED files in one directory, outside the neural-model interface.

    Input resources can use absolute paths; output/history paths must be relative
    so they remain inside this directory beside the native checkpoint.
    This working-directory scope is for the single-simulation CLI driver.
    """

    def __init__(self, source, output, temperature, interval):
        try:
            from openmmplumed import PlumedForce
        except ImportError as error:
            raise ImportError(
                "PLUMED requested: install openmm-plumed for the active OpenMM environment."
            ) from error
        self.script = Path(source).read_bytes()
        self.path = Path(output).resolve() / "plumed"
        self.temperature = temperature
        self.interval = interval
        self.force_group = None
        self._force_class = PlumedForce

    @contextmanager
    def directory(self):
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "input.dat").write_bytes(self.script)
        previous = Path.cwd()
        os.chdir(self.path)
        try:
            yield
        finally:
            os.chdir(previous)

    def attach(self, system, *, restart=False):
        # Flush output/history at every update, including a final partial block.
        # This changes only I/O, and leaves the user's bias/CV expressions intact.
        force = self._force_class(self.script.decode() + "\nFLUSH STRIDE=1\n")
        force.setTemperature(self.temperature)
        force.setRestart(restart)
        used = {f.getForceGroup() for f in system.getForces()}
        available = sorted(set(range(32)) - used)
        if not available:
            raise ValueError("No unused OpenMM force group for PLUMED.")
        self.force_group = available[-1]
        force.setForceGroup(self.force_group)
        force.setName("PLUMED bias")
        system.addForce(force)
        return force

    def save(self, simulation, checkpoint):
        """Flush the current bias step, then use OpenMM's native checkpoint writer."""
        if self.force_group is not None:
            simulation.context.getState(getEnergy=True, groups={self.force_group})
        simulation.saveCheckpoint(str(checkpoint))

    def check_restart(self, time_ps):
        """Reject time-series tails newer than the loaded native checkpoint."""
        for path in self.path.iterdir():
            if not path.is_file() or path.name.startswith("bck."):
                continue
            with path.open("rb") as stream:
                fields = stream.readline().split()
                if fields[:2] != [b"#!", b"FIELDS"] or b"time" not in fields[2:]:
                    continue
                column = fields.index(b"time") - 2
                stream.seek(max(stream.tell(), path.stat().st_size - 65536))
                tail = stream.read()
            if tail and not tail.endswith(b"\n"):
                raise ValueError(f"Incomplete PLUMED history: {path}")
            for line in reversed(tail.splitlines()):
                if not line.strip() or line.lstrip().startswith(b"#"):
                    continue
                value = float(line.split()[column])
                # PLUMED's default timestamps are rounded to six decimal places.
                if not math.isfinite(value) or value > time_ps + 5e-7:
                    raise ValueError(
                        f"PLUMED history {path.name} extends beyond checkpoint time "
                        f"{time_ps:g} ps; restore matching bias/history files before resuming."
                    )
                break

    def reporter(self, checkpoint):
        bias = self

        class CheckpointReporter:
            def describeNextReport(self, simulation):
                return {
                    "steps": bias.interval - simulation.currentStep % bias.interval,
                    "periodic": None,
                    "include": [],
                }

            def report(self, simulation, state):
                bias.save(simulation, checkpoint)

        return CheckpointReporter()
