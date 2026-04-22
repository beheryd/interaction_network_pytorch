# interaction-networks-dmfc

Test whether an interaction network (Battaglia et al. 2016) closes the rapid-early-prediction gap that Rajalingham et al. 2025 flagged between macaque DMFC and their task-optimized RNNs on the Mental Pong task (Fig. 5B). Secondary: verify the IN is at least competitive with those RNNs on the online-simulation metric (Fig. 4).

This repo is a fork of [higgsfield/interaction_network_pytorch](https://github.com/higgsfield/interaction_network_pytorch) extended with a Mental Pong training environment, interaction-network variants across Rajalingham's four loss classes, and the analysis pipeline needed to place the IN on Rajalingham's Fig. 5B and Fig. 4 axes. Class project for *Foundations of Computational Cognitive Neuroscience*.

## Read these files to understand the project

Please read the following files before doing any work on this project:

- `CONSTITUTION.md` — project-specific coding rules (NON-NEGOTIABLE)
- `PRD.md` — goals, non-goals, and requirements
- `PLANNING.md` — architecture, tech stack, workflow
- `TASKS.md` — current task list and milestones
- `SCRATCHPAD.md` — in-progress notes and open problems
- `VERSIONS.md` — pinned dependency versions

## Common commands

```bash
# Environment setup (one-time)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Sanity check the Mental Pong env
python -m dmfc.envs.mental_pong --render --seed 0 --condition 0

# Train one model (config + seed → one run directory under runs/)
python -m dmfc.training.train --config configs/in_vis_occ.yaml --seed 0

# Regenerate the primary deliverable (Fig. 5B with IN added)
python -m dmfc.analysis.reproduce_fig5b --in-runs runs/in_* --rajalingham-data data/dmfc/

# Regenerate the secondary deliverable (Fig. 4 with IN swarm)
python -m dmfc.analysis.reproduce_fig4 --in-runs runs/in_* --rajalingham-data data/dmfc/

# Quality gates (run before every commit)
ruff check . && ruff format .
pytest
mypy dmfc/
```

## Slash commands

- `/freshstart` — re-read context files after `/clear` or `/compact`
- `/summcommit` — summarize session into TASKS/SCRATCHPAD and git commit
