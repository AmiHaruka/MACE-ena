"""Training arguments and output paths."""

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="Train EnvHead with native MACE data")
    parser.add_argument(
        "--name", required=True, help="Experiment and exported model name"
    )
    parser.add_argument("--train-file", required=True, nargs="+")
    parser.add_argument("--valid-file", nargs="+")
    for key in ("foundation", "head-config", "resume"):
        parser.add_argument("--" + key)
    parser.add_argument("--work-dir", default=".")
    for key in ("log-dir", "results-dir", "checkpoints-dir", "model-dir"):
        parser.add_argument("--" + key)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"])
    integers = (
        "epochs",
        "batch-size",
        "valid-batch-size",
        "seed",
        "num-workers",
        "scheduler-patience",
        "stage-two-start",
    )
    floats = (
        "learning-rate",
        "energy-weight",
        "forces-weight",
        "charges-weight",
        "mm-forces-weight",
        "valid-fraction",
        "weight-decay",
        "foundation-lr-factor",
        "environment-lr-factor",
        "clip-grad",
        "lr-factor",
        "ema-decay",
        "stage-two-lr",
        "stage-two-energy-weight",
        "stage-two-forces-weight",
        "stage-two-charges-weight",
        "stage-two-mm-forces-weight",
    )
    for key in integers:
        parser.add_argument("--" + key, type=int)
    for key in floats:
        parser.add_argument("--" + key, type=float)
    for key in ("amsgrad", "ema", "stage-two", "pin-memory"):
        parser.add_argument(
            "--" + key, action=argparse.BooleanOptionalAction, default=None
        )
    parser.add_argument("--enable-cueq", action="store_true")
    parser.add_argument("--cueq-conv-fusion", action="store_true", default=None)
    parser.add_argument("--execution-mode", choices=["reference", "optimized"])
    parser.add_argument("--neighbor-backend", choices=["auto", "reference", "pyg"])
    return parser


def parse_args(argv=None, previous=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    # Keep the existing single-file argument/checkpoint representation.
    for key in ("train_file", "valid_file"):
        paths = getattr(args, key)
        if paths is not None and len(paths) == 1:
            setattr(args, key, paths[0])
    previous = previous or {}
    args.explicit_learning_rate = args.learning_rate is not None
    explicit_valid_batch = args.valid_batch_size is not None
    defaults = dict(
        epochs=200,
        batch_size=900,
        learning_rate=0.01,
        energy_weight=10.0,
        forces_weight=100.0,
        charges_weight=10.0,
        mm_forces_weight=0.0,
        seed=0,
        valid_fraction=0.1,
        weight_decay=5e-7,
        foundation_lr_factor=1.0,
        environment_lr_factor=1.0,
        clip_grad=10.0,
        amsgrad=True,
        lr_factor=0.8,
        scheduler_patience=50,
        ema=True,
        ema_decay=0.99,
        stage_two=True,
        stage_two_lr=0.001,
        stage_two_energy_weight=100.0,
        stage_two_forces_weight=10.0,
        num_workers=0,
        pin_memory=str(args.device).startswith("cuda"),
        neighbor_backend="reference",
        execution_mode="optimized",
        cueq_conv_fusion=False,
    )
    for key, default in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, previous.get(key, default))
    args.valid_batch_size_linked = not explicit_valid_batch and previous.get(
        "valid_batch_size_linked", True
    )
    if not explicit_valid_batch:
        args.valid_batch_size = (
            args.batch_size
            if args.valid_batch_size_linked
            else previous["valid_batch_size"]
        )
    if args.stage_two_start is None:
        args.stage_two_start = previous.get(
            "stage_two_start", max(1, args.epochs * 3 // 4)
        )
    if args.stage_two_charges_weight is None:
        args.stage_two_charges_weight = previous.get(
            "stage_two_charges_weight", args.charges_weight
        )
    if args.stage_two_mm_forces_weight is None:
        args.stage_two_mm_forces_weight = previous.get(
            "stage_two_mm_forces_weight", args.mm_forces_weight
        )
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name must be a filename without directory components")
    if (
        min(args.epochs, args.batch_size, args.valid_batch_size, args.stage_two_start)
        < 1
    ):
        parser.error(
            "Epochs, batch sizes and the completed-epoch stage boundary must be positive"
        )
    if args.num_workers < 0 or args.scheduler_patience < 0:
        parser.error("Worker count and scheduler patience must be nonnegative")
    if not 0 < args.lr_factor < 1 or not 0 <= args.ema_decay < 1:
        parser.error("Require 0 < lr-factor < 1 and 0 <= ema-decay < 1")
    if min(args.learning_rate, args.stage_two_lr, args.clip_grad) <= 0:
        parser.error("Learning rates and clipping norm must be positive")
    if (
        min(
            args.weight_decay,
            args.foundation_lr_factor,
            args.environment_lr_factor,
            args.energy_weight,
            args.forces_weight,
            args.charges_weight,
            args.mm_forces_weight,
            args.stage_two_energy_weight,
            args.stage_two_forces_weight,
            args.stage_two_charges_weight,
            args.stage_two_mm_forces_weight,
        )
        < 0
    ):
        parser.error("Weights and learning-rate multipliers must be nonnegative")
    return args


class RunPaths:
    """Namespaced artifacts; the public CLI never assumes a dataset or hardware."""

    def __init__(self, args):
        self.name, self.seed = args.name, args.seed
        self.tag = f"{self.name}_run-{self.seed}"
        self.work = Path(args.work_dir)
        self.logs = Path(args.log_dir) if args.log_dir else self.work / "logs"
        self.results = (
            Path(args.results_dir) if args.results_dir else self.work / "results"
        )
        self.checkpoints = (
            Path(args.checkpoints_dir)
            if args.checkpoints_dir
            else self.work / "checkpoints"
        )
        self.models = Path(args.model_dir) if args.model_dir else self.work

    @property
    def metrics(self):
        return self.results / f"{self.tag}_metrics.jsonl"

    @property
    def arguments(self):
        return self.results / f"{self.tag}_arguments.json"

    @property
    def log(self):
        return self.logs / f"{self.tag}.log"

    @property
    def validation_indices(self):
        return self.work / f"{self.name}_valid_indices_{self.seed}.txt"

    def checkpoint(self, kind):
        return self.checkpoints / f"{self.tag}_{kind}.pt"

    def model(self, stage):
        suffix = "" if stage == 1 else "_stagetwo" if stage == 2 else "_last"
        return self.models / f"{self.name}{suffix}.model"

    def create(self):
        for path in (self.work, self.logs, self.results, self.checkpoints, self.models):
            path.mkdir(parents=True, exist_ok=True)

    def ensure_new(self):
        artifacts = (
            self.metrics,
            self.arguments,
            self.checkpoint("last"),
            self.checkpoint("best"),
            self.model(1),
            self.model(2),
            self.model("last"),
        )
        if any(path.exists() for path in artifacts):
            raise FileExistsError(
                "Experiment already exists; use --resume or a different --name/work-dir"
            )
