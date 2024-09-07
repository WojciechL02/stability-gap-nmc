import torch
from copy import deepcopy
from argparse import ArgumentParser

from metrics import cka
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Hinge Loss approach.
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr=1e-1, wu_fix_bn=False,
                 wu_scheduler='constant', wu_patience=None, wu_wd=0., fix_bn=False, eval_on_train=False,
                 select_best_model_by_val_loss=True, logger=None, exemplars_dataset=None, scheduler_milestones=False,
                 lamb=1, cka=False, debug_loss=False
                 ):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr, wu_fix_bn, wu_scheduler, wu_patience, wu_wd,
                                   fix_bn, eval_on_train, select_best_model_by_val_loss, logger, exemplars_dataset,
                                   scheduler_milestones)
        self.lamb = lamb
        self.cka = cka
        self.debug_loss = True

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Hinge Loss balance (default=%(default)s)')
        parser.add_argument('--cka', default=False, action='store_true', required=False,
                            help='If set, will compute CKA between current representations and representations at '
                                 'the start of the task. (default=%(default)s)')
        parser.add_argument('--debug-loss', default=False, action='store_true', required=False,
                            help='If set, will log intermediate loss values. (default=%(default)s)')

        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        self.training = True
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)
        self.training = False

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for images, targets in trn_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model(images)
            loss, loss_margin, loss_ce = self.criterion(t, outputs, targets, return_partial_losses=True)
            if self.debug_loss:
                if t > 0:
                    self.logger.log_scalar(task=None, iter=None, name='loss_margin', group=f"debug_t{t}",
                                        value=float(loss_margin.item()))
                    self.logger.log_scalar(task=None, iter=None, name='loss_ce', group=f"debug_t{t}",
                                        value=float(loss_ce.item()))
                self.logger.log_scalar(task=None, iter=None, name='loss_total', group=f"debug_t{t}",
                                       value=float(loss.item()))

            assert not torch.isnan(loss), "Loss is NaN"

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def eval(self, t, val_loader, log_partial_loss=False):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            total_loss_ce = 0
            total_loss_margin = 0
            self.model.eval()

            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)

                outputs = self.model(images)
                loss, loss_margin, loss_ce = self.criterion(t, outputs, targets, return_partial_losses=True)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_loss_margin += loss_margin.data.cpu().numpy().item() * len(targets)
                total_loss_ce += loss_ce.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)

        # if self.cka and t > 0 and self.training:
            # _cka = cka(self.model, self.model_old, val_loader, self.device)
            # self.logger.log_scalar(task=None, iter=None, name=f't_{t}', group=f"cka", value=_cka)

        if log_partial_loss:
            final_loss_margin = total_loss_margin / total_num
            final_loss_ce = total_loss_ce / total_num
            self.logger.log_scalar(task=None, iter=None, name="loss_ce", value=final_loss_ce, group="valid")
            self.logger.log_scalar(task=None, iter=None, name="loss_margin", value=final_loss_margin, group="valid")

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets, return_partial_losses=False):
        """Returns the loss value"""
        # Hinge Loss
        if t > 0:
            hinge_loss = torch.nn.functional.multi_margin_loss(torch.cat(outputs, dim=1), targets)
        else:
            hinge_loss = torch.zeros(1).to(self.device)

        # Current cross-entropy loss
        loss_ce = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)

        if return_partial_losses:
            return self.lamb * hinge_loss + loss_ce, hinge_loss, loss_ce
        else:
            return self.lamb * hinge_loss + loss_ce
