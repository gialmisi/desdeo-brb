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
- **The analytical ER algorithm** [@WangEtAl2006] derives the closed-form
  aggregation function that `compute_combined_belief_degrees` evaluates, in
  place of the recursive algorithm, precisely so that the ER approach can be
  used where an explicit function is needed, as it is for training here. It
  reaches this library through Eq. A-15 of [@ChenEtAl2011].
- **The evidential reasoning rule** [@YangXu2013] states the recursive form of
  the combination in its Eqs. (14) to (17). `tests/unit/test_inference.py`
  implements it independently as an oracle for the analytical formula, over
  both complete and incomplete rule bases.
- **Optimization models for training** [@YangEtAl2007] set out the training
  constraints. Their constraint 12b caps a rule's belief degrees at one and
  imposes the sum-to-one equality only when a complete trained rule base is
  wanted, which is what `allow_incomplete` exposes. Section V-E compares
  expert and random initialisation and reports similar final accuracy at
  markedly different training cost.

## Extended belief rule bases

- **The extended formulation** [@LiuEtAl2013] embeds belief degrees in a rule's
  antecedent terms as well as its consequent, so that a rule may sit between
  referential values rather than only at them, and gives a method for
  generating such a rule base directly from numerical data.
  `RuleBase.antecedent_beliefs` implements this. Note that the *extended belief
  rule base* meant here is this 2013 formulation, which [@ZhuangEtAl2021] call
  Liu-EBRB; it is not the earlier ISKE 2008 paper of a similar name, which
  extends RIMER to fuzzy *consequents* and is unrelated to what this library
  does.
- **Distance-based matching** [@ZhuangEtAl2021] gives the activation used for
  extended antecedents, in their Eqs. (7) to (9): the distance between the
  input's belief distribution and the rule's, halved before the square root so
  that two disjoint distributions are exactly one apart. That halving assumes
  both sum to one, which is why an antecedent may not be incomplete even though
  the formalism admits it. The rule generated from
  a data point in their Eqs. (15) and (16) is checked in
  `tests/integration/test_literature_examples.py`, and the Liu-EBRB accuracies
  of their Table 4 are reproduced to within a point on Iris, Ecoli and Glass in
  `tests/integration/test_published_accuracy.py`. Their fourth dataset, Pima, is
  not covered: UCI has withdrawn it.

## Applications

- **Pipeline leak detection** [@XuEtAl2007] is the canonical BRB
  application, reproduced in `notebooks/03_expert_knowledge.ipynb`.
- **INFRINGER** [@Misitano2020] uses BRBs to learn decision-maker
  preferences in interactive multi-objective optimisation. This library
  originated as the machine-learning core of INFRINGER.
- **The INFRINGER thesis** [@MisitanoThesis2020] develops that method in full.
  Its Eq. 3.20 is the combination formula implemented by
  `compute_combined_belief_degrees`, and the benchmark functions of its
  Sections 3.5 and 3.7 are reproduced in `tests/integration/`.
