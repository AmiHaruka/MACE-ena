"""Calculate an embedded molecular single point with the ASE calculator."""

import argparse
from pathlib import Path

import numpy as np
from ase.io import read

from mace.modules.envhead.calculator import EnvHeadCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Trained MACE-ena model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path(__file__).with_name("water.extxyz"),
    )
    parser.add_argument("--output", type=Path, help="Optional NumPy .npz output")
    args = parser.parse_args()

    atoms = read(args.structure)
    atoms.calc = EnvHeadCalculator(args.model, device=args.device)
    mm_forces = atoms.calc.get_mm_forces(atoms)
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()

    print(f"Energy: {energy:.8f} eV")
    print("Molecular forces (eV/Angstrom):")
    print(forces)
    print("MM force contributions (eV/Angstrom):")
    print(mm_forces)

    if args.output is not None:
        np.savez_compressed(
            args.output,
            energy_eV=energy,
            forces_eV_A=forces,
            mm_forces_eV_A=mm_forces,
        )


if __name__ == "__main__":
    main()
