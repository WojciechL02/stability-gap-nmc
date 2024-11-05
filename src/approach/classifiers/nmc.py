import torch
import numpy as np

from torch.utils.data import DataLoader
from datasets.exemplars_selection import override_dataset_transform


from .classifier import Classifier


class NMC(Classifier):
    """Class implementing the Nearest-Mean-Classifier (NMC)"""

    def __init__(self, device, model, exemplars_dataset, best_prototypes=False):
        self.device = device
        self.model = model
        self.exemplars_dataset = exemplars_dataset
        self.exemplar_means = []
        self.previous_datasets = []
        self.best_prototypes = best_prototypes

    def classify(self, task, outputs, features, targets, return_dists=False):
        # expand means to all batch images
        means = torch.stack(self.exemplar_means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (features - means).pow(2).sum(1, keepdim=True).squeeze(dim=1)
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset:offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device)).float()
        if return_dists:
            return hits_taw, hits_tag, dists
        return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        dataset = self.previous_datasets[0] if self.best_prototypes else self.exemplars_dataset
        if self.best_prototypes:
            if len(self.previous_datasets) > 1:
                for subset in self.previous_datasets[1:-1]:
                    dataset += subset
        with override_dataset_transform(dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarl_loader = DataLoader(_ds, batch_size=trn_loader.batch_size, shuffle=False,
                                      num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            # extract features from the model for all train samples
            extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarl_loader:
                    feats = self.model(images.to(self.device), return_features=True)[1]
                    # normalize
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
            extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    def compute_means_of_current_classes(self, loader, transform):
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            self.model.eval()
            with override_dataset_transform(loader.dataset, transform) as _ds:
                # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
                icarl_loader = DataLoader(_ds, batch_size=loader.batch_size, shuffle=False,
                                        num_workers=loader.num_workers, pin_memory=loader.pin_memory)
                for images, targets in icarl_loader:
                    feats = self.model(images.to(self.device), return_features=True)[1]
                    # normalize
                    extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)
        extracted_features = torch.cat(extracted_features)
        extracted_targets = np.array(extracted_targets)
        for curr_cls in np.unique(extracted_targets):
            if curr_cls >= self.model.task_offset[-1]:
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                cls_feats = extracted_features[cls_ind]
                # add the exemplars to the set and normalize
                cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                self.exemplar_means.append(cls_feats_mean)

    def prototypes_update(self, t, trn_loader, transform):
        if self.exemplars_dataset._is_active():
            self.exemplar_means = []
            if t > 0:
                self.compute_mean_of_exemplars(trn_loader, transform)
            self.compute_means_of_current_classes(trn_loader, transform)
