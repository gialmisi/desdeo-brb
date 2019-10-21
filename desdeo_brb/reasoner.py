from desdeo_brb.brb import BRBResult
from typing import List
import numpy as np


class ReasonerException(Exception):
    """Raised when an error related to Reasoner objects is encountered.

    """

    pass


class Reasoner(object):
    """This class provides utilities to employ ER reasoning on a BRB results
    object.

    Arguments:
        results (BRBResult): A BRBResult object used to initialize the reasoner.
        input_explats (List[List[str]]): A list of lists of strings containing
        explanations for each of the referential values for the input values
        (precedents)
        ouput_explats (List[List[str]]): Like input_explats, but for the consequents.

    Note:
        The dimensions of the precednets in the BRBResult object and
        input_explats, and the dimensions of the consequents in the BRBResult
        object and output_explats must match!

    """

    def __init__(
        self,
        result: BRBResult,
        input_explats: List[List[str]],
        output_explats: List[List[str]],
    ):
        self.result = result
        # check the dimensions of the input explanations
        if self.result.precedents.shape == np.asarray(input_explats).shape:
            self.input_explats = input_explats
        else:
            msg = (
                "Shapes of the precedents and the input explanations don't "
                "match!"
            )
            raise ReasonerException(msg)

        # check the dimensions of the output explanations
        if self.result.consequents.shape == np.asarray(output_explats):
            self.output_explats = output_explats
        else:
            msg = (
                "Shapes of the consequents and the output explanations "
                "don't match!"
            )
            raise ReasonerException(msg)

        def explain(self, result: BRBResult):
            pass


if __name__ == "__main__":
    result = BRBResult(
        precedents=np.array([[0, 0.5, 1], [1, 2, 3]]),
        precedents_belief_degrees=np.array([[0, 0.2, 0.7], [0.1, 0.2, 0.3]]),
        consequents=np.array([[0, 0.5, 1]]),
        consequent_belief_degrees=np.array([0, 0.75, 1]),
    )
    reasoner = Reasoner(
        result,
        [["bad", "fair", "good"], ["low", "medium", "high"]],
        [["poor", "rich", "excellent"]],
    )
