import torch
from torch import nn
from copy import deepcopy


class LLL_Net(nn.Module):
    """Basic class for implementing networks"""

    def __init__(self, model, remove_existing_head=False, head_init_mode=None, use_ssl_projector=False):
        head_var = model.head_var
        assert type(head_var) == str
        assert not remove_existing_head or hasattr(model, head_var), \
            "Given model does not have a variable called {}".format(head_var)
        assert not remove_existing_head or type(getattr(model, head_var)) in [nn.Sequential, nn.Linear], \
            "Given model's head {} does is not an instance of nn.Sequential or nn.Linear".format(head_var)
        super(LLL_Net, self).__init__()

        self.model = model
        self.head_init_mode = head_init_mode
        last_layer = getattr(self.model, head_var)

        if remove_existing_head or use_ssl_projector:
            if type(last_layer) == nn.Sequential:
                self.out_size = last_layer[-1].in_features
                # strips off last linear layer of classifier
                del last_layer[-1]
            elif type(last_layer) == nn.Linear:
                self.out_size = last_layer.in_features
                # converts last layer into identity
                # setattr(self.model, head_var, nn.Identity())
                # WARNING: this is for when pytorch version is <1.2
                setattr(self.model, head_var, nn.Sequential())
        else:
            self.out_size = last_layer.out_features
        
        self.ssl_projector = None
        if use_ssl_projector:
            self.ssl_projector = nn.Sequential(
                nn.Linear(self.out_size, 2 * self.out_size),
                nn.BatchNorm1d(2 * self.out_size),
                nn.ReLU(),
                nn.Linear(2 * self.out_size, self.out_size)
            )

        self.primary_state_dict = deepcopy(self.model.state_dict())

        self.heads = nn.ModuleList()
        self.task_cls = []
        self.task_offset = []
        self._initialize_weights()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(nn.Linear(self.out_size, num_outputs))
        # first head has the same init (zeros) in all methods
        if len(self.heads) == 1:
            nn.init.zeros_(self.heads[-1].weight)
            nn.init.zeros_(self.heads[-1].bias)

        # weights initialization for other heads
        elif self.head_init_mode is not None:
            self._initialize_head_weights()

        # we re-compute instead of append in case an approach makes changes to the heads
        self.task_cls = torch.tensor([head.out_features for head in self.heads])
        self.task_offset = torch.cat([torch.LongTensor(1).zero_(), self.task_cls.cumsum(0)[:-1]])

    def forward(self, x, return_features=False):
        """Applies the forward pass

        Simplification to work on multi-head only -- returns all head outputs in a list
        Args:
            x (tensor): input images
            return_features (bool): return the representations before the heads
        """
        x = self.model(x)
        assert (len(self.heads) > 0), "Cannot access any head"
        
        if self.ssl_projector is not None:
            # x = torch.nn.functional.normalize(x, dim=1)
            y = self.ssl_projector(x)
            y = torch.nn.functional.normalize(y, dim=1)
        else:
            y = []
            for head in self.heads:
                y.append(head(x))

        if return_features:
            return y, x
        else:
            return y

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
        return

    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.model.parameters():
            param.requires_grad = False

    def reset_backbone(self):
        """Reset all parameters from the main model, but not the heads"""
        self.model.load_state_dict(self.primary_state_dict)

    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def unfreeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = True

    def unfreeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_last_head(self):
        for param in self.heads[-1].parameters():
            param.requires_grad = False

    def unfreeze_last_head(self):
        for param in self.heads[-1].parameters():
            param.requires_grad = True

    def _initialize_head_weights(self):
        if self.head_init_mode == 'xavier':
            nn.init.xavier_uniform_(self.heads[-1].weight)
            nn.init.zeros_(self.heads[-1].bias)

        elif self.head_init_mode == 'zeros':
            nn.init.zeros_(self.heads[-1].weight)
            nn.init.zeros_(self.heads[-1].bias)

        elif self.head_init_mode == 'kaiming':
            nn.init.kaiming_uniform_(self.heads[-1].weight)
            nn.init.zeros_(self.heads[-1].bias)

    def _initialize_weights(self):
        """Initialize weights using different strategies"""
        # TODO: add different initialization strategies
        pass