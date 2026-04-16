# References

The implementation follows the RIMER (Rule-base Inference Methodology using
the Evidential Reasoning approach) framework and its adaptive-training
extensions.

## Core methodology

- **RIMER methodology** [@YangEtAl2006] introduced the Belief Rule Base
  inference framework that this library implements.
- **Adaptive training** [@ChenEtAl2011] describes the parameter-learning
  approach used by `BRBModel.fit()`.

## Applications

- **Pipeline leak detection** [@XuEtAl2007] is the canonical BRB
  application, reproduced in `notebooks/03_expert_knowledge.ipynb`.
- **INFRINGER** [@Misitano2020] uses BRBs to learn decision-maker
  preferences in interactive multi-objective optimisation. This library
  originated as the machine-learning core of INFRINGER.
