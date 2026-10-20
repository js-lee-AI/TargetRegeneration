# Third-party components

- [SpecForge](https://github.com/sgl-project/SpecForge), MIT license. The `specforge`
  package and `core/train_backend.py` are adapted from its training implementation.
  See LICENSE.SpecForge.
- [DFlash](https://github.com/z-lab/dflash), MIT license. `dflash/model.py` provides
  the public draft architecture and text decoder. See LICENSE.DFlash.

The training backend is the one released with
[ALR-IRA](https://github.com/js-lee-AI/ALR-IRA). It already implements token-level
distillation, the survival gate, soft weighting and greedy target labels; this release adds
the L2 anchor to the released drafter. It keeps that release's rollout options, which no run
here uses. Upstream copyright notices are retained. Model and dataset licenses remain with
their respective distributors. The gpt-oss chat templates and the Harmony parser of
SpecForge are removed, since no run in the paper uses them.
