# EconomyGame

This repository contains a small prototype of an economic simulation engine. The
main entry point is `src/sim.py` which defines a tick based update loop and
simple market mechanics. A deterministic random generator is used to reproduce
runs when a seed is provided.

## Running

The simulation can be run directly with Python. Use `--ticks` to specify the
number of ticks and `--seed` for deterministic runs:

```bash
python src/sim.py --ticks 10 --seed 42
```

You can also run the small demo which prints price movements for a toy market:

```bash
python scripts/demo.py
```

## Testing

Run unit tests with `pytest`:

```
pytest
```



