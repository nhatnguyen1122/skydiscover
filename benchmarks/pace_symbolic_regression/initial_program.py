import numpy as np


class LLMSR:
    # EVOLVE-BLOCK-START
    def equation(self, inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Symbolic regression equation.

        The evaluator passes a 2D array of observations:
        - oscillator1: [x, v]
        - oscillator2: [t, x, v]
        - bactgrow: [b, s, temp, pH]
        - stressstrain: [strain, temp]

        Return one prediction per row.
        """
        n_features = inputs.shape[1]
        output = np.zeros(inputs.shape[0], dtype=np.float64)
        for i in range(n_features):
            x = inputs[:, i]
            output += params[i] * x
            output += params[i + n_features] * x * x
        output += params[-1]
        return output
    # EVOLVE-BLOCK-END
