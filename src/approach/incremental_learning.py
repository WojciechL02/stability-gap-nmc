import time
from copy import deepcopy
from collections import Counter
import torch
import numpy as np
from argparse import ArgumentParser
import umap
import pandas as pd

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset
from .classifiers.classifier_factory import ClassifierFactory
from .classifiers.nmc import NMC


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, classifier="linear", nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr=1e-1, wu_fix_bn=False,
                 wu_scheduler='constant', wu_patience=None, wu_wd=0., fix_bn=False,
                 eval_on_train=False, select_best_model_by_val_loss=True, logger: ExperimentLogger = None,
                 exemplars_dataset: ExemplarsDataset = None, scheduler_milestones=False, no_learning=False, log_grad_norm=True, best_prototypes=False, slca=False):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = wu_lr
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.warmup_scheduler = wu_scheduler
        self.warmup_fix_bn = wu_fix_bn
        self.warmup_patience = wu_patience
        self.wu_wd = wu_wd
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.select_best_model_by_val_loss = select_best_model_by_val_loss
        self.optimizer = None
        self.scheduler_milestones = scheduler_milestones
        self.scheduler = None
        self.debug = False
        self.no_learning = no_learning
        self.classifier = ClassifierFactory.create_classifier(classifier, device, self.model, self.exemplars_dataset, best_prototypes, multi_softmax)
        self.val_loader_transform = None

        self.last_e_accs = None  # variable for stability-gap metric
        self.log_grad_norm = False
        self.projector = None  # UMAP
        self.slca = slca

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.slca:  # and len(self.model.heads) > 1:
            backbone_params = {'params': self.model.model.parameters(), 'lr': self.lr * 0.1}
            head_params = {'params': self.model.heads.parameters()}
            network_params = [backbone_params, head_params]
            return torch.optim.SGD(network_params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _get_scheduler(self):
        if self.scheduler_milestones:
            # return torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=self.scheduler_milestones, gamma=0.1)
            return torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=self.nepochs)
        else:
            return None

    def get_embeddings(self, t):
        loaders = self.tst_loader[:t + 1]
        self.model.eval()
        labels = []
        embed_list = []
        with torch.no_grad():
            for i, loader in enumerate(loaders):
                for images, targets in loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model.model(images)  # get only backbone results
                    embed_list.append(outputs)
                    labels.append(targets)
        return torch.cat(embed_list, dim=0), torch.cat(labels, dim=0)
    
    def project_latent_space(self, embeddings, labels):
        if self.projector is None:
            self.projector = umap.UMAP(metric="euclidean", n_neighbors=50)
            projected = self.projector.fit_transform(embeddings.cpu())
        else:
            projected = self.projector.transform(embeddings.cpu())
        data = pd.DataFrame(projected)
        data["label"] = labels.cpu()
        return data

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:

            # Log acc_tag before warmup ---------------------------------------------------------
            outputs = self._evaluate(t, debug=True)
            for name, value in outputs.items():
                if name == "tag_acc_current_task" or name == "tag_acc_all_tasks":
                    self.logger.log_scalar(task=None, iter=None, name=f"before_wu_{name}", group=f"warmup_t{t}", value=value)
            # -----------------------------------------------------------------------------------

            prev_heads_b_norm = torch.cat([h.bias for h in self.model.heads[:-1]], dim=0).detach().norm().item()
            prev_heads_w_norm = torch.cat([h.weight for h in self.model.heads[:-1]], dim=0).detach().norm().item()
            self._log_weight_norms(t, prev_heads_w_norm, prev_heads_b_norm,
                                   self.model.heads[-1].weight.detach().norm().item(),
                                   self.model.heads[-1].bias.detach().norm().item())

            head_params = self.model.heads[-1].parameters()
            optimizer = torch.optim.SGD(head_params, lr=self.warmup_lr, weight_decay=self.wu_wd)

            if self.warmup_scheduler.lower() == 'constant':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
            elif self.warmup_scheduler.lower() == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                                       min_lr=1e-4)
            elif self.warmup_scheduler.lower() == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.warmup_epochs, eta_min=0)
            elif self.warmup_scheduler.lower() == 'onecycle':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.warmup_lr,
                                                                total_steps=self.warmup_epochs,
                                                                pct_start=0.2)
            else:
                raise NotImplementedError(f"Unknown scheduler: {self.warmup_scheduler}.")

            patience = 0
            best_loss = float('inf')
            best_model = self.model.state_dict()

            # loader with bigger batch size for more stable warmup
            # new_trn_loader = torch.utils.data.DataLoader(trn_loader.dataset, batch_size=512, shuffle=True, num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)

            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()

                if self.warmup_fix_bn:
                    self.model.eval()
                else:
                    self.model.train()
                self.model.heads[-1].train()

                for images, targets in trn_loader:
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    outputs = self.model(images)
                    loss = self.warmup_loss(outputs[t], targets - self.model.task_offset[t])
                    self.logger.log_scalar(task=None, iter=None, name="trn_batch_loss", value=loss.item(), group=f"warmup_t{t}")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head_params, self.clipgrad)
                    optimizer.step()
                warmupclock1 = time.time()

                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        images, targets = images.to(self.device), targets.to(self.device)
                        outputs = self.model(images)
                        loss = self.warmup_loss(outputs[t], targets - self.model.task_offset[t])
                        pred = torch.zeros_like(targets)
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0).to(self.device) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=None, iter=e + 1, name="trn_loss", value=trn_loss, group=f"warmup_t{t}")
                self.logger.log_scalar(task=None, iter=e + 1, name="trn_acc", value=100 * trn_acc, group=f"warmup_t{t}")

                # Evaluate -------------------------------------------------------------------------
                warmupclock3 = time.time()
                outputs = self._evaluate(t, debug=True)
                for name, value in outputs.items():
                    if name == "tag_acc_current_task" or name == "tag_acc_all_tasks":
                        self.logger.log_scalar(task=None, iter=e + 1, name=name, group=f"warmup_t{t}", value=value)
                # if self.debug:
                #     self.logger.log_scalar(task=None, iter=e + 1, name='lr', group=f"warmup_t{t}",
                #                            value=optimizer.param_groups[0]["lr"])
                #     self._log_weight_norms(t, prev_heads_w_norm, prev_heads_b_norm,
                #                            self.model.heads[-1].weight.detach().norm().item(),
                #                            self.model.heads[-1].bias.detach().norm().item())

                warmupclock4 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s | Eval: loss={:.3f}, TAg loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock4 - warmupclock3, outputs['ce_taw_current_task'],
                    outputs['ce_tag_current_task'],
                    100 * outputs['taw_acc_current_task']), end=''
                )
                # -----------------------------------------------------------------------------------

                if self.warmup_scheduler == 'plateau':
                    scheduler.step(outputs['tag_acc_current_task'])
                else:
                    scheduler.step()

                if self.warmup_patience is not None:
                    if outputs['ce_taw_current_task'] < best_loss:
                        best_loss = outputs['ce_taw_current_task']
                        best_model = self.model.state_dict()
                        patience = 0
                        print("*")
                    else:
                        patience += 1
                        print()
                        if patience > self.warmup_patience:
                            print(f"Stopping early at epoch {e + 1}")
                            break
                else:
                    best_model = self.model.state_dict()
                    print()

            self.model.load_state_dict(state_dict=best_model)

    def _evaluate(self, t, debug=False):
        if t == 0:
            raise ValueError()

        loaders = self.tst_loader[:t + 1]

        self.model.eval()
        per_task_taw_acc = []
        per_task_tag_acc = []
        per_task_ce_taw = []
        per_task_ce_tag = []

        with torch.no_grad():
            for i, loader in enumerate(loaders):
                if not debug and i != len(loaders) - 1:
                    continue

                total_acc_taw, total_acc_tag, total_ce_taw, total_ce_tag, total_num = 0, 0, 0, 0, 0
                for images, targets in loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model(images)
                    hits_taw, hits_tag = self.classifier.classify(t, outputs, targets)
                    ce_taw = torch.nn.functional.cross_entropy(outputs[i], targets - self.model.task_offset[i],
                                                               reduction='sum')
                    ce_tag = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets, reduction='sum')

                    # Log
                    total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                    total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                    total_ce_taw += ce_taw.cpu().item()
                    total_ce_tag += ce_tag.cpu().item()
                    total_num += len(targets)
                per_task_taw_acc.append(total_acc_taw / total_num)
                per_task_tag_acc.append(total_acc_tag / total_num)
                per_task_ce_taw.append(total_ce_taw / total_num)
                per_task_ce_tag.append(total_ce_tag / total_num)

        if debug:
            output = {
                "tag_acc_current_task": per_task_tag_acc[-1],
                "tag_acc_all_tasks": sum(per_task_tag_acc[:-1]) / len(per_task_tag_acc[:-1]),
                "taw_acc_current_task": per_task_taw_acc[-1],
                "ce_taw_current_task": per_task_ce_taw[-1],
                "ce_taw_all_tasks": sum(per_task_ce_taw) / len(per_task_ce_taw),
                "ce_tag_current_task": per_task_ce_tag[-1],
                "ce_tag_all_tasks": sum(per_task_ce_tag) / len(per_task_ce_tag),
            }
        else:
            output = {
                "tag_acc_current_task": per_task_tag_acc[-1],
                "taw_acc_current_task": per_task_taw_acc[-1],
                "ce_taw_current_task": per_task_ce_taw[-1],
                "ce_tag_current_task": per_task_ce_tag[-1],
            }
        return output

    def _continual_evaluation_step(self, t):
        confusion_matrix = torch.zeros((t+1, t+1))
        prev_t_acc = torch.zeros((t,), requires_grad=False)
        current_t_acc = 0.
        sum_acc = 0.
        total_loss_curr = 0.
        total_num_curr = 0
        current_t_acc_taw = 0
        with torch.no_grad():
            loaders = self.tst_loader[:t + 1]

            self.model.eval()
            for task_id, loader in enumerate(loaders):
                total_acc_tag = 0.
                total_acc_taw = 0.
                total_num = 0
                task_ids = []
                for images, targets in loader:
                    images, targets = images.to(self.device), targets.to(self.device)
                    # Forward current model
                    outputs, feats = self.model(images, return_features=True)

                    if task_id == t:
                        loss = self.criterion(task_id, outputs, targets)
                        total_loss_curr += loss.item() * len(targets)
                        total_num_curr += total_num

                    outputs_stacked = torch.stack(outputs, dim=1)
                    shape = outputs_stacked.shape
                    hits_taw, hits_tag, outputs = self.classifier.classify(task_id, outputs, feats, targets, return_dists=True)
                    # Log
                    total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                    total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                    total_num += len(targets)

                    ### TO JEST DO ROZWAŻENIA ######################
                    if isinstance(self.classifier, NMC):
                        outputs = outputs.view(shape[0], shape[1], shape[2])
                        outputs = torch.min(outputs, dim=-1)[0]
                        outputs = outputs.argmin(dim=-1)
                        task_ids.extend(outputs.tolist())
                    else:
                        outputs = torch.stack(outputs, dim=1)
                        outputs = torch.max(outputs, dim=-1)[0]
                        task_ids.extend(outputs.argmax(dim=-1).tolist())
                    ###############################################

                counts = Counter(task_ids)
                for j, val in counts.items():
                    confusion_matrix[task_id, j] = val / len(loader.dataset)

                acc_tag = total_acc_tag / total_num
                acc_taw = total_acc_taw / total_num
                self.logger.log_scalar(task=task_id, iter=None, name="acc_tag", value=100 * acc_tag, group="cont_eval")
                self.logger.log_scalar(task=task_id, iter=None, name="acc_taw", value=100 * acc_taw, group="cont_eval")
                if task_id < t:
                    sum_acc += acc_tag
                    prev_t_acc[task_id] = acc_tag
                else:
                    current_t_acc = acc_tag
                    current_t_acc_taw = acc_taw

            if t > 0:
                # Average accuracy over all previous tasks
                self.logger.log_scalar(task=None, iter=None, name="avg_acc_tag", value=100 * sum_acc / t, group="cont_eval")

        if t > 0:
            recency_bias = confusion_matrix[:-1, -1].mean()
            self.logger.log_scalar(task=None, iter=None, name="task_recency_bias", value=recency_bias.item(), group="cont_eval")

        avg_prev_acc = sum_acc / t if t > 0 else 0.
        return prev_t_acc, current_t_acc, avg_prev_acc  #, total_loss_curr / total_num_curr, current_t_acc_taw
        # acc poprzednich tasków, acc na aktualnym tasku, średnia z poprzednich tasków

    def _log_weight_norms(self, t, prev_w, prev_b, new_w, new_b):
        self.logger.log_scalar(task=None, iter=None, name='prev_heads_w_norm', group=f"wu_w_t{t}",
                               value=prev_w)
        self.logger.log_scalar(task=None, iter=None, name='prev_heads_b_norm', group=f"wu_w_t{t}",
                               value=prev_b)
        self.logger.log_scalar(task=None, iter=None, name='new_head_w_norm', group=f"wu_w_t{t}",
                               value=new_w)
        self.logger.log_scalar(task=None, iter=None, name='new_head_b_norm', group=f"wu_w_t{t}",
                               value=new_b)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        self.val_loader_transform = val_loader.dataset.transform
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        min_accs_prev = torch.ones((t,), requires_grad=False)

        # Compare head after warmup and after task training
        head_after_wu = deepcopy(self.model.heads[-1])

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader)

            # CONTINUAL EVALUATION
            ####################################################################
            # prev_t_accs, current_acc, avg_prev_acc, _, _ = self._continual_evaluation_step(t)
            prev_t_accs, current_acc, avg_prev_acc = self._continual_evaluation_step(t)
            # valid_locc and valid_acc are on test_loaders so it works when --use-test-as-val

            # save accs on last epoch of task 0
            if t == 0 and e == self.nepochs - 1:
                self.last_e_accs = torch.tensor([current_acc])

            if t > 0:
                # min-ACC
                min_accs_prev = torch.minimum(min_accs_prev, prev_t_accs)
                min_acc = min_accs_prev.mean().item()
                self.logger.log_scalar(task=None, iter=None, name="min_acc", value=100 * min_acc, group="cont_eval")

                # WC-ACC
                k = t + 1
                wc_acc = (1 / k) * current_acc + (1 - (1 / k)) * min_acc
                self.logger.log_scalar(task=None, iter=None, name="wc_acc", value=100 * wc_acc, group="cont_eval")

                # in last epoch of tasks > 0
                if e == self.nepochs - 1:
                    # Stability Gap
                    sg = self.last_e_accs - min_accs_prev
                    sg_normalized = torch.div(sg, self.last_e_accs)
                    # Recovery
                    rec = prev_t_accs - min_accs_prev  # in last epoch prev_t_accs is final_acc of prev tasks
                    rec_normalized = torch.div(rec, self.last_e_accs)
                    # Log SG and REC
                    for ts in range(sg.shape[0]):
                        self.logger.log_scalar(task=ts, iter=None, name="stability_gap", value=100 * sg[ts].item(), group="cont_eval")
                        self.logger.log_scalar(task=ts, iter=None, name="stability_gap_normal", value=100 * sg_normalized[ts].item(), group="cont_eval")

                        self.logger.log_scalar(task=ts, iter=None, name="recovery", value=100 * rec[ts].item(), group="cont_eval")
                        self.logger.log_scalar(task=ts, iter=None, name="recovery_normal", value=100 * rec_normalized[ts].item(), group="cont_eval")
                                            
                    self.logger.log_scalar(task=None, iter=None, name="stability_gap_avg", value=100 * sg.mean().item(), group="cont_eval")
                    self.logger.log_scalar(task=None, iter=None, name="recovery_avg", value=100 * rec.mean().item(), group="cont_eval")
                    self.logger.log_scalar(task=None, iter=None, name="sg_normal_avg", value=100 * sg_normalized.mean().item(), group="cont_eval")
                    self.logger.log_scalar(task=None, iter=None, name="recovery_normal_avg", value=100 * rec_normalized.mean().item(), group="cont_eval")

                    # New last acc of prev tasks (current task becomes a prev task)
                    self.last_e_accs = torch.cat((prev_t_accs, torch.tensor([current_acc])))
            ####################################################################

            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader, log_partial_loss=False)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            ### EVAL DONE IN CONTINUAL_EVALUATION
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader, log_partial_loss=True)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()[0]

            if self.select_best_model_by_val_loss:
                # Adapt learning rate - patience scheme - early stopping regularization
                if valid_loss < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = valid_loss
                    best_model = self.model.get_copy()
                    patience = self.lr_patience
                    print(' *', end='')
                else:
                    patience -= 1
                    if self.scheduler is None:
                        # if the loss does not go down, decrease patience
                        if patience <= 0:
                            # if it runs out of patience, reduce the learning rate
                            lr /= self.lr_factor
                            print(' lr={:.1e}'.format(lr), end='')
                            if lr < self.lr_min:
                                # if the lr decreases below minimum, stop the training session
                                print()
                                break
                            # reset patience and recover best model so far to continue training
                            patience = self.lr_patience
                            self.optimizer.param_groups[0]['lr'] = lr
                            self.model.set_state_dict(best_model)
            else:
                best_model = self.model.get_copy()

            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()

            # ======== LATENT SPACE ANALYSIS =========
            # if ((t == 0 and e == self.nepochs - 1) or (t > 0 and e in (0, self.nepochs // 2, self.nepochs - 1))):
            #     embeddings, labels = self.get_embeddings(t)
            #     data_vis = self.project_latent_space(embeddings, labels)
            #     self.logger.log_latent_vis(group="Latent_Vis", task=t, epoch=e, data=data_vis)
            # ========================================

        self.model.set_state_dict(best_model)

        # MEASURE LAST HEAD DISTANCE/SIMILARITY
        # if t > 0:
        #     head_after_wu_vect = torch.nn.utils.parameters_to_vector(head_after_wu.parameters()).unsqueeze(0)
        #     head_curr_vect = torch.nn.utils.parameters_to_vector(self.model.heads[-1].parameters()).unsqueeze(0)

        #     # L2 distance
        #     l2_dist = torch.linalg.vector_norm(head_after_wu_vect - head_curr_vect, 2)
        #     self.logger.log_scalar(task=None, iter=None, name='L2 distance', group=f"Last head", value=l2_dist.item())

        #     # Cosine Similarity
        #     cos_sim = torch.nn.functional.cosine_similarity(head_after_wu_vect, head_curr_vect)
        #     self.logger.log_scalar(task=None, iter=None, name='Cos-Sim', group=f"Last head", value=cos_sim.item())

        #     # L2 of the final classifier
        #     final_clf_l2 = torch.linalg.vector_norm(head_curr_vect, 2)
        #     self.logger.log_scalar(task=None, iter=None, name='clf L2', group=f"Last head", value=final_clf_l2.item())

            # self._log_heads_activation_statistics(t)

    def _log_heads_activation_statistics(self, t):
        self.model.eval()
        with torch.no_grad():
            loaders = self.tst_loader[:t + 1]

            for task_id, loader in enumerate(loaders):
                actv = [torch.zeros((len(loader.dataset), self.model.heads[i].out_features)) for i in range(len(self.model.heads))]
                act_maxs = torch.zeros((1, len(loader)), requires_grad=False)
                for batch_id, (images, targets) in enumerate(loader):
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model(images)
                    act_maxs[0, batch_id] = outputs[task_id].max(dim=1).values.mean()

                    for head_id in range(len(self.model.heads)):
                        start = batch_id * len(targets)
                        actv[head_id][start:start + len(targets)] = outputs[head_id]

                for head_id in range(len(actv)):
                    # hist = np.histogram(actv[head_id].flatten().cpu().numpy(), bins=100)
                    values = actv[head_id].flatten().cpu().numpy()
                    self.logger.log_histogram(group="Histograms", name=f"Head_{head_id}_act", task=task_id, sequence=values)

                final_max = act_maxs.max()
                self.logger.log_scalar(task=task_id, iter=None, name='Max', group='Head activations', value=final_max.item())

    def _calc_backbone_grad_norm(self):
        total_norm = 0
        for p in self.model.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            if t == 0 or not self.no_learning:
                self.optimizer.zero_grad()
                loss.backward()
                # ======== LOG GRADIENT NORM ========
                if self.log_grad_norm:
                    total_norm = self._calc_backbone_grad_norm()
                    self.logger.log_scalar(task=t, iter=None, name="Backbone", value=total_norm, group="Grad_Norm")
                # ===================================
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)

            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # compute mean of exemplars on every epoch
        self.classifier.prototypes_update(t, trn_loader, self.val_loader_transform)

    def eval(self, t, val_loader, log_partial_loss=False):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                images, targets = images.to(self.device), targets.to(self.device)

                outputs, feats = self.model(images, return_features=True)
                loss = self.criterion(t, outputs, targets)
                hits_taw, hits_tag = self.classifier.classify(t, outputs, feats, targets)

                # hits_tag / len(targets)

                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
