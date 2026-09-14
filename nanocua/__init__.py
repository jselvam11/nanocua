"""nanocua: a nanoGPT-style skeleton for training computer-use agents.

Mental model
------------
    Stage 1  grounding triples  --format-->  localize samples  --train-->  VLM that points
    Stage 2  Trajectory  --expand-->  SFT samples  --train-->  policy  --eval-->  metrics
                  |                        |                       |
             goal + T steps         prefix of history          offline: match gold
             screenshot/thought     + current screenshot       online:  mock env
             + next action          -> next thought+action

Grounding is the sibling of trajectory SFT, not a replacement. Read
``schema.py`` first (including ``GroundingExample``), then ``data/expand.py``,
then ``train/sft.py``.
"""

from nanocua.schema import Action, GroundingExample, Observation, SFTSample, Step, Trajectory

__version__ = "0.1.0"
__all__ = [
    "Action",
    "GroundingExample",
    "Observation",
    "SFTSample",
    "Step",
    "Trajectory",
    "__version__",
]
