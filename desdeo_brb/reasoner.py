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
        input_explats: List[List[str]],
        output_explats: List[str],
        attribute_names: List[str],
        output_name: str
    ):
        self.input_explats = input_explats
        self.output_explats = output_explats
        self.attribute_names = attribute_names
        self.output_name = output_name

    def explain(self, result: BRBResult):
        att_n = self.attribute_names
        pre_b = result.precedents_belief_degrees[0]
        pre_x = self.input_explats
        con_n = self.output_name
        con_b = result.consequent_belief_degrees[0]
        con_x = self.output_explats

        inputs = []
        for i in range(len(att_n)):
            inputs.append({
                "what": f"{att_n[i]}",
                "is": [f"{int(pre_b[i][j]*100)}%  {pre_x[i][j]}" for j in range(len(pre_x[i]))],
                "ignorance": f"{int((1 - sum(pre_b[i]))*100)}%",
                })

        output = ({
            "what": f"{con_n}",
            "is": [f"{int(con_b[j]*100)}%  {con_x[0][j]}" for j in range(len(con_x[0]))],
            "ignorance": f"{int((1 - sum(con_b))*100)}%",
            })

        # form the explanation
        explanation = (
            "WITH\n\t\t" +
            "\n\tAND\n\t\t".join(
                list((f"{d['what']} BEING {', '.join(d['is'])} WITH {d['ignorance']} ignorance" for d in inputs))
            ) +
            "\nFOLLOWS\n\t\t" +
            f"{output['what']} BEING {', '.join(output['is'])} WITH {output['ignorance']} ignorance"
        )

        return explanation


if __name__ == "__main__":
    result = BRBResult(
        precedents=np.array([[0, 0.5, 1], [1, 2, 3]]),
        precedents_belief_degrees=np.array([[0, 0.2, 0.7], [0.1, 0.2, 0.3]]),
        consequents=np.array([[0, 0.5, 1]]),
        consequent_belief_degrees=np.array([0, 0.75, 0.25]),
    )
    reasoner = Reasoner(
        [[["bad", "fair", "good"], ["low", "medium", "high"]]],
        [[["poor", "rich", "excellent"]]],
        ["condition", "price"],
        "deal quality"
    )

    print(reasoner.explain(result))
