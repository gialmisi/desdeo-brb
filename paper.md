---
title: "desdeo-brb: A trainable Belief Rule-Based inference system with a scikit-learn-compatible API"
tags:
  - Python
  - belief rule-base
  - evidential reasoning
  - interpretable machine learning
  - explainable AI
  - decision support
authors:
  - name: Giovanni Misitano
    orcid: 0000-0002-4673-7388
    affiliation: 1
affiliations:
  - name: Faculty of Information Technology, University of Jyväskylä, Finland
    index: 1
date: 24 April 2026
bibliography: paper.bib
archive_doi: 10.5281/zenodo.19731658
---

# Summary

Belief Rule-Based (BRB) systems are a generalization of classical IF-THEN
rule bases in which each rule's consequent is expressed as a belief
distribution over a discrete set of possible output grades rather than a
single value [@yang2006rimer]. Inference is performed using the
Evidential Reasoning (ER) algorithm, which analytically combines all
activated rules into a single output distribution that accounts for rule
weights, attribute weights, and partial rule activation simultaneously. A
key practical advantage of BRB systems is that their parameters—belief
degrees, rule weights, attribute weights, and referential values—can be
initialized from domain-expert knowledge and subsequently refined by fitting
to observed data [@chen2011inference], making them applicable even in
data-scarce regimes.

`desdeo-brb` is a Python library implementing a fully trainable BRB model with a
scikit-learn–compatible [@scikit-learn] interface. It exposes `fit`, `predict`,
`score`, `get_params`, and `set_params` methods, allowing the model to be used
directly in scikit-learn pipelines, cross-validation routines, and
hyperparameter search. The library supports varying-length referential values
per attribute, multiple constrained optimization backends via SciPy [@scipy2020]
(SLSQP, trust-constr, Differential Evolution, and a two-stage DE+SLSQP hybrid),
and an optional JAX [@jax2018github] backend that provides JIT-compiled
inference and exact-gradient training through automatic differentiation. Every
intermediate quantity in the inference pipeline, i.e., which rules fired, their
activation weights, the combined belief distribution, is directly accessible for
inspection and explanation.

# Statement of need

The BRB methodology has generated a substantial body of literature. A recent
survey identified over 400 publications since the foundational RIMER paper
[@yang2006rimer; @survey2024brb], yet, according to the best of our knowledge,
the research community has no
general-purpose, publicly available, open source software library for trainable
BRB systems. The current dominant
software, the Intelligent Decision System (IDS), is a proprietary, closed-source
Microsoft Windows desktop application developed at Manchester [@xu2006ids]. While BRB inference is grounded in Dempster-Shafer theory,
existing open-source implementations of that theory, notably the R packages
`ibelief` [@zhou_ibelief], `dst` [@boivin_dst], and `evclust` [@denoeux_evclust],
provide only the underlying evidence combination primitives. None implements the
higher-level RIMER pipeline: input transformation to belief distributions, rule
activation weight computation, or parameter training, which `desdeo-brb`
offer.

The de facto research platform is MATLAB,
but implementations are invariably paper-specific and not publicly released. The
only Python package described in a formal software publication, ERTool
[@shi2024ertool], implements only the ER evidence combination step and does not
support rule activation, belief degree parameterization, or parameter
optimization. Three additional Python repositories on GitHub:
`brunompacheco/brb` [@brunompacheco_brb], `Eriri/brb` [@eriri_brb], and
`iamrafiul/lib_brb` [@iamrafiul_libbrb], implement BRB inference, and
in one case gradient-based training, but none is
pip-installable, licensed, documented beyond the source code itself,
or exposes a reusable programmatic API, and none are actively^[The most recent
commit found across the listed repositories was around six years ago at the time
of writing this manuscript.] maintained.

Contributing to one of these existing repositories was considered but
was not feasible. Each represents a valuable independent effort to
bring BRB into the Python ecosystem, but none carries an open source
license, all have been inactive since 2020, and none has a named
maintainer accepting external contributions. The design goals of
`desdeo-brb` also differ substantially from those of the existing
implementations: `brunompacheco/brb` focuses on categorical referential values
and inference; `Eriri/brb` explores TensorFlow-based gradient training with
Gaussian activation functions; and `iamrafiul/lib_brb` targets hierarchical
JSON-driven expert systems. `desdeo-brb` pursues a different set of priorities:
scikit-learn compatibility, constrained optimization via SciPy with multiple
solver backends, and differentiable training through an optional JAX backend.
These would have required major fundamental architectural changes to any of the
existing codebases, effectively having to rewrite them.

This absence of open, reproducible tooling creates a concrete barrier to
research: studies cannot build on a shared computational baseline, results are
difficult to reproduce, and practitioners outside the core BRB community cannot
adopt the methodology without reimplementing it from scratch. `desdeo-brb`
addresses these gaps: it is the first pip-installable open source
BRB library with formal documentation, the first to provide a
scikit-learn–compatible API [@sklearn_api], the first to support differentiable
training via automatic differentiation, and the first to be described in a
peer-reviewed software publication.

# State of the field

BRB systems occupy a distinctive niche in the interpretable machine learning
landscape. Compared to decision trees, BRB rules carry richer uncertainty
representations through distributed belief consequents. Compared to Bayesian
networks, BRB uses Dempster-Shafer theory which can represent genuine
ignorance rather than forcing probability assignments to sum to one.
Compared to fuzzy rule-based systems, BRB adds learnable belief distributions
to rule consequents; Cao et al. describe BRB as "the generalization of fuzzy
systems" [@cao2021brb]. Compared to neural networks, BRB provides what
multiple authors have characterised as a "white-box" or "gray-box" model
[@gong2025graybox], where every step of the inference is directly inspectable.

Parameter training has been an active area of research. Yang et al.
formulated BRB training as a constrained optimization problem and solved it
with MATLAB's `fmincon` [@yang2007training]. Subsequent work proposed
Differential Evolution [@chang2016de],
gradient descent with Gaussian activation functions [@guan2021gd], and
analytical gradient computation using the Method of Feasible Direction
[@feng2020mfd]. `desdeo-brb`'s JAX backend makes gradient-based BRB training
accessible as a library feature for the first time.

The growing interest in Explainable AI (XAI) has driven BRB applications
into high-stakes domains including clinical decision support
[@kong2012cardiac], pipeline leak detection [@xu2007pipeline], and
aerospace relay health state assessment [@yin2023aerospace]. These are all
contexts where the traceability of reasoning is not merely desirable
but often required by regulation or professional practice.

# Software design

`desdeo-brb` is structured around three core abstractions. The `RuleBase`
(a validated Pydantic [@pydantic] model) stores the complete parameterization of a BRB
system: precedent and consequent referential values, belief degrees, rule
weights, attribute weights, and rule antecedent indices. The `BRBModel` class
wraps a `RuleBase` and implements the full inference and training pipeline.
The `InferenceResult` container exposes the complete inference trace—input
belief distributions, per-rule activation weights, combined belief degrees,
dominant rules, and the scalar output—for downstream inspection.

The NumPy [@numpy2020] inference pipeline follows the ER algorithm of [@yang2006rimer] as
refined by [@chen2011inference]. Input transformation maps raw attribute values
to belief distributions over the referential values of each attribute using
linear interpolation; values outside the referential range are assigned full
belief to the nearest endpoint. Activation weights are computed as the product
of per-attribute matching degrees scaled by attribute weights and the overall
rule weight. Combined belief degrees are computed by the analytical ER
combination formula. The scalar output is the inner product of the combined
belief distribution with the consequent referential values, or a user-supplied
utility function thereof.

Training minimizes the sum of squared residuals between BRB predictions and
target outputs subject to the BRB parameter constraints (belief degree rows sum
to one, rule weights sum to one, referential values remain sorted).  Multiple
SciPy optimization methods are supported and selectable via the `method`
argument to `fit`.  An optional Pyomo [@pyomo2021] backend provides access to
the IPOPT [@ipopt2006] interior-point solver for problems with custom symbolic
objectives via `fit_custom()`. The optional JAX backend re-implements the full
inference pipeline using JAX primitives, enabling JIT compilation for throughput
and `jax.grad` / `jax.value_and_grad` for exact gradient computation.
Unconstrained training is supported through differentiable reparameterizations:
softmax for belief degree rows and softplus for non-negative weights.

The scikit-learn interface allows `BRBModel` to be used wherever an sklearn
estimator is accepted. The following example from the getting-started notebook
illustrates the workflow for modeling a nonlinear scalar function, following
the example in [@chen2011inference]:

```python
import numpy as np
from desdeo_brb import BRBModel

f = lambda x: x[0] * np.sin(x[0] ** 2)

prv = [np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])]
crv = np.array([-2.5, -1.0, 1.0, 2.0, 3.0])

model = BRBModel(prv, crv, initial_rule_fn=f)

rng = np.random.default_rng(67)
X_train = rng.uniform(0, 3, size=(1000, 1))
y_train = f(X_train.T)

model.fit(X_train, y_train)

result = model.predict(np.linspace(0, 3, 50).reshape(-1, 1))
print(result.dominant_rules(top_k=3))
```

Additional notebooks cover multi-attribute models, expert knowledge
integration using the pipeline leak detection example of
[@chen2011inference], and the explainability interface. Full API
documentation, installation instructions, and a training guide are
hosted on [ReadTheDocs](https://desdeo-brb.readthedocs.io).

# Research impact statement

The original iteration of `desdeo-brb` was developed to support the INFRINGER
method [@misitano2020infringer], which uses a BRB system to learn a
decision-maker's value function during interactive multiobjective optimization.
This constitutes the primary documented research application of the library,
demonstrating its utility beyond BRB research proper as an interpretable,
trainable surrogate model for preference learning in contexts where transparency
of the learned model is required.

More broadly, `desdeo-brb` provides the BRB community with the reproducible,
open computational baseline that has been absent for twenty years of active
research. By standardizing the inference algorithm, the training interface,
and the parameter representation in a pip-installable package, it lowers the
barrier for researchers to reproduce existing results, extend established
methods, and compare approaches on common benchmarks.

# AI usage disclosure

Claude (Anthropic) was used during both software development and paper
preparation. Claude Code assisted with code refactoring, test
generation, and documentation writing during the April 2026 development
phase. Claude Sonnet 4.6 assisted with drafting and editing portions of
this paper text. All code and text were reviewed, edited, and validated
carefully by the human author, who takes full responsibility for accuracy and
originality.

# Acknowledgements

The development of `desdeo-brb` was supported by the Research Council of Finland
(grant number 355346). The software is related to the thematic research area
DEMO (Decision Analytics utilizing Causal Models and Multiobjective
Optimization, [homepage](https://jyu.fi/demo)) of the University of Jyväskylä.

# References
