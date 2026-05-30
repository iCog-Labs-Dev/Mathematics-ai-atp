from maths_ai.gnn_inference.model import GNNModelEngine
from maths_ai.pln_inference.model import PLNInference
class HybridReasoner:
    def __init__(self):
        self.gnn_engine = GNNModelEngine()
        self.petta_chainer = PLNInference()
        self.atomic_tactics = {}

    def predict_next_tactic(self, goal: str):
        self