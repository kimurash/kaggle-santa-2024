import math
import random

from metrics.perplexity import PerplexityCalculator


class Optimizer:
    MAX_ITER = 100_000

    INITIAL_TEMP = 100
    COOLING_RATE = 0.99

    def __init__(self):
        self.calculator = PerplexityCalculator(model_path="google/gemma-2-9b")

    def optimize(self, text: str) -> str:
        cur_list = text.split()
        cur_ppl = self.calculator.get_perplexity(text)

        best_list = cur_list.copy()
        best_ppl = cur_ppl

        temperature = self.INITIAL_TEMP

        for iter in range(self.MAX_ITER):
            neigh_list = cur_list.copy()

            idx1, idx2 = random.sample(range(len(neigh_list)), 2)
            neigh_list[idx1], neigh_list[idx2] = neigh_list[idx2], neigh_list[idx1]

            neigh_text = " ".join(neigh_list)
            neigh_ppl = self.calculator.get_perplexity(neigh_text)

            if neigh_ppl < cur_ppl:
                cur_list = neigh_list
                cur_ppl = neigh_ppl

                if neigh_ppl < best_ppl:
                    best_list = neigh_list
                    best_ppl = neigh_ppl
            else:
                diff = neigh_ppl - cur_ppl
                p = math.exp(-diff / temperature)
                r = random.random()
                if r < p:
                    cur_list = neigh_list
                    cur_ppl = neigh_ppl

            temperature *= self.COOLING_RATE

        return " ".join(best_list)
