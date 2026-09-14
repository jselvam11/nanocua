"""nanocua: a nanoGPT-style skeleton for training computer-use agents.

Mental model
------------
    Trajectory  --expand-->  SFT samples  --train-->  policy  --eval-->  metrics
         |                        |                       |
    goal + T steps         prefix of history          offline: match gold
    screenshot/thought     + current screenshot       online:  mock env
    + next action          -> next thought+action

Read ``schema.py`` first, then ``data/expand.py``, then ``train/sft.py``.
"""

from nanocua.schema import Action, Observation, SFTSample, Step, Trajectory

__version__ = "0.1.0"
__all__ = [
    "Action",
    "Observation",
    "SFTSample",
    "Step",
    "Trajectory",
    "__version__",
]
