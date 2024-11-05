from .linear import LinearClassifier
from .nmc import NMC


class ClassifierFactory:
    @staticmethod
    def create_classifier(classifier_type, device, model, dataset, best_prototypes=False, multi_softmax=False):
        if classifier_type == "linear":
            return LinearClassifier(device, model, dataset, multi_softmax)
        elif classifier_type == "nmc":
            return NMC(device, model, dataset, best_prototypes)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
