"""M6 commands; keeps optional JAX loading off the SAC path."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="command", required=True)
for name in ("sac", "dreamer"):
    p = sub.add_parser(name)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--demonstrations",
        type=Path,
        default=Path("work/m6/demonstrations/demonstrations.npz"),
    )
    p.add_argument("--steps", type=int, default=20000 if name == "sac" else 10000)
    p.add_argument("--offline-updates", type=int, default=3000)
    p.add_argument("--bc-weight", type=float, default=100)
    p.add_argument("--seed", type=int, default=17)
p = sub.add_parser("evaluate")
p.add_argument("--algorithm", choices=("sac", "dreamer", "bc"), required=True)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--count", type=int, default=12)
p.add_argument("--split", choices=("validation", "test"), default="validation")
p.add_argument("--validation", type=Path)
args = parser.parse_args()
if args.command == "sac":
    from bb8_rl.guided import train_sac

    result = train_sac(
        args.output,
        args.demonstrations,
        steps=args.steps,
        seed=args.seed,
        bc_weight=args.bc_weight,
    )
elif args.command == "dreamer":
    from bb8_rl.guided_dreamer import train_dreamer

    result = train_dreamer(
        args.output,
        args.demonstrations,
        steps=args.steps,
        offline_updates=args.offline_updates,
        seed=args.seed,
        bc_weight=args.bc_weight,
    )
else:
    from bb8_rl.guided_evaluate import evaluate

    result = evaluate(
        args.algorithm,
        args.checkpoint,
        args.output,
        count=args.count,
        split=args.split,
        validation=args.validation,
    )
print(json.dumps(result, indent=2))
