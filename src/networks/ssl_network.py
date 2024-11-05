import torch
from torch import nn
from .network import LLL_Net


class SSL_Net(LLL_Net):
    def __init__(self, model, remove_existing_head=True, head_init_mode=None, projector_type="mlp"):
        """Class implementing a network for Contrastive Learning approaches."""
        super(SSL_Net, self).__init__(model, remove_existing_head=True, head_init_mode=head_init_mode)

        self.projector = self._init_projector(projector_type)

    def _init_projector(self, projector_type: str):
        if projector_type == "mlp":
            return nn.Sequential(
                        nn.Linear(self.out_size, self.out_size // 2),
                        nn.BatchNorm1d(self.out_size // 2),
                        nn.ReLU(),
                        nn.Linear(self.out_size // 2, self.out_size // 4)
                    )
        elif projector_type == "linear":
            return nn.Linear(self.out_size, self.out_size // 4)

        raise ValueError(f"Projector: {projector_type} not supported.")

    # def add_head(self, num_outputs):
    #     """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
    #     corresponding offsets
    #     """
    #     old_task_cls = [out for out in self.task_cls]
    #     old_task_cls.append(num_outputs)
    #     self.task_cls = torch.tensor(old_task_cls)
    #     self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Args:
            x (tensor): input images
            return_features (bool): return the representations before the projector
        """
        x = self.model(x)

        y = torch.nn.functional.normalize(x, dim=1)
        y = self.projector(y)
        y = torch.nn.functional.normalize(y, dim=1)

        if return_features:
            return y, x
        else:
            return y

    def freeze_last_head(self):
        pass

    def unfreeze_last_head(self):
        pass
