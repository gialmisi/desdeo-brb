"""Reproduce published accuracies for the extended belief rule base.

Zhuang et al. (2021), Table 2, reports the accuracy of Liu-EBRB, the plain
extended belief rule base without their clustering tree or differential
evolution training, under ten-fold cross validation with five referential
values per attribute.

Nothing here is trained. Referential values are equally spaced over the
training range, each training sample becomes one rule whose antecedent is that
sample's own belief distribution and whose consequent is its class, and rule and
attribute weights are equal throughout. That is what makes these numbers
diagnostic of the extended matching specifically: a discrepancy cannot be
blamed on an optimiser that is not there.

Iris ships with scikit-learn, so that check needs no network. Ecoli and Glass
are fetched from UCI and skip when that fails. Their fourth dataset, Pima, is
not covered: UCI withdrew it, so it can only be had from mirrors.
"""

import numpy as np
import pytest

from desdeo_brb.brb import BRBModel
from desdeo_brb.inference import input_transform
from desdeo_brb.models import RuleBase

REFERENTIALS = 5
FOLDS = 10

# Fold assignment moves the mean by a few tenths of a point, and their split
# seeds are not published, so agreement is asserted to within a point.
TOLERANCE = 1.0


def _liu_ebrb_accuracy(X: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Return ten-fold cross-validated accuracy of an untrained extended rule base."""
    generator = np.random.default_rng(0)
    order = generator.permutation(len(X))
    X, y = X[order], y[order]

    grades = np.arange(n_classes, dtype=float)
    scores = []
    for held_out in np.array_split(np.arange(len(X)), FOLDS):
        mask = np.ones(len(X), dtype=bool)
        mask[held_out] = False
        train_x, train_y = X[mask], y[mask]

        points = []
        for i in range(train_x.shape[1]):
            low, high = train_x[:, i].min(), train_x[:, i].max()
            # A constant attribute would give a degenerate span, which leaves
            # the interpolation undefined.
            high = high if high > low else low + 1e-9
            points.append(np.linspace(low, high, REFERENTIALS))

        beliefs = np.zeros((len(train_x), n_classes))
        beliefs[np.arange(len(train_x)), train_y] = 1.0
        rule_base = RuleBase(
            precedent_referential_values=[p.copy() for p in points],
            consequent_referential_values=grades.copy(),
            belief_degrees=beliefs,
            # A constant rule weight cancels in the activation, so this is
            # Liu-EBRB's theta = 1 for every rule.
            rule_weights=np.full(len(train_x), 1.0 / len(train_x)),
            attribute_weights=np.ones((len(train_x), train_x.shape[1])),
            antecedent_beliefs=[a.copy() for a in input_transform(train_x, points)],
        )
        model = BRBModel(
            precedent_referential_values=[p.copy() for p in points],
            consequent_referential_values=grades.copy(),
            rule_base=rule_base,
        )
        combined = np.asarray(model.predict(X[held_out]).combined_belief_degrees)
        scores.append(float((combined.argmax(axis=1) == y[held_out]).mean()))

    return 100.0 * float(np.mean(scores))


def _from_uci(dataset_id: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Fetch a UCI dataset, skipping the test when it cannot be reached."""
    ucimlrepo = pytest.importorskip("ucimlrepo")
    try:
        fetched = ucimlrepo.fetch_ucirepo(id=dataset_id)
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"could not fetch UCI dataset {dataset_id}: {error}")

    X = fetched.data.features.to_numpy(dtype=float)
    labels = fetched.data.targets.iloc[:, 0].to_numpy()
    names = sorted(set(labels))
    y = np.array([names.index(v) for v in labels])
    return X, y, len(names)


def test_iris_matches_published_accuracy():
    """Zhuang et al. (2021) Table 2 reports 95.26 per cent for Liu-EBRB on Iris."""
    sklearn_datasets = pytest.importorskip("sklearn.datasets")
    data = sklearn_datasets.load_iris()
    accuracy = _liu_ebrb_accuracy(data.data, data.target, 3)
    assert accuracy == pytest.approx(95.26, abs=TOLERANCE)


@pytest.mark.network
def test_ecoli_matches_published_accuracy():
    """Table 2 reports 81.16 per cent for Liu-EBRB on Ecoli.

    Their Table 1 lists Ecoli as having two categories. It has eight, and eight
    is what reproduces their accuracy, so that entry is a typo.
    """
    X, y, n_classes = _from_uci(39)
    assert n_classes == 8
    accuracy = _liu_ebrb_accuracy(X, y, n_classes)
    assert accuracy == pytest.approx(81.16, abs=TOLERANCE)


@pytest.mark.network
def test_glass_matches_published_accuracy():
    """Table 2 reports 67.85 per cent for Liu-EBRB on Glass."""
    X, y, n_classes = _from_uci(42)
    accuracy = _liu_ebrb_accuracy(X, y, n_classes)
    assert accuracy == pytest.approx(67.85, abs=TOLERANCE)
