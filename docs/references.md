# References

The implementation follows the RIMER (Rule-base Inference Methodology using
the Evidential Reasoning approach) framework and its adaptive-training
extensions.

## Core methodology

- **RIMER methodology** [@YangEtAl2006] introduced the Belief Rule Base
  inference framework that this library implements.
- **Adaptive training** [@ChenEtAl2011] describes the parameter-learning
  approach used by `BRBModel.fit()`.
- **The evidential reasoning algorithm** [@YangXu2002] gives the combination
  rule and, in Section II-H, the utility interval that bounds the prediction
  when an assessment is incomplete.
- **Optimization models for training** [@YangEtAl2007] set out the training
  constraints. Their constraint 12b caps a rule's belief degrees at one and
  imposes the sum-to-one equality only when a complete trained rule base is
  wanted, which is what `allow_incomplete` exposes. Section V-E compares
  expert and random initialisation and reports similar final accuracy at
  markedly different training cost.

## Applications

- **Pipeline leak detection** [@XuEtAl2007] is the canonical BRB
  application, reproduced in `notebooks/03_expert_knowledge.ipynb`.
- **INFRINGER** [@Misitano2020] uses BRBs to learn decision-maker
  preferences in interactive multi-objective optimisation. This library
  originated as the machine-learning core of INFRINGER.
