import time
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from .FairGB.models import *
from .FairGB.eval import evaluate_ged3
from .FairGB.utils import seed_everything, get_enc_cls_opt
from .FairGB.mixup import (sampling_idx_individual_dst, neighbor_sampling,
                                   get_ins_neighbor_dist, saliency_mixup)

# Weakness 3 (ydje) / Point 3 (Kips) rebuttal metrics.
# Adjust this import path if your package layout differs.
from utils.metrics import individual_fairness_consistency, sensitive_leakage_probe

EPS = 1e-6


def fair_metric(pred, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity   = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()


def predict_sens_group(pred, labels, sens, idx_test):
    result = []
    for s in [0, 1]:
        mask = (sens[idx_test] == s)
        F1   = f1_score(labels[idx_test][mask], pred[mask], average='micro')
        ACC  = accuracy_score(labels[idx_test][mask], pred[mask])
        AUC  = roc_auc_score(labels[idx_test][mask], pred[mask])
        result.extend([ACC, AUC, F1])
    return result


class FairGB:
    def __init__(self):
        self.val_loss = 0

    def fit(self, data, device='cuda', runs=1, seed=0,
            epochs=1000, c_lr=0.01, c_wd=0, e_lr=0.01, e_wd=0,
            dropout=0.5, hidden=16, encoder='SAGE', alpha=1,
            warmup=5, eta=0.5, trajectory=None,
            use_cal=True, use_cnm=True):
        """`trajectory` optional; None leaves every existing caller unchanged.

        Note `alpha` is the **selector's** fairness weight (it multiplies
        parity + equality in the checkpoint score below), not the strength of
        CAL or CNM, which are this method's claimed interventions. Calling both
        "alpha" is how the two came to be conflated; here it is alpha_select.
        """

        class Args:
            pass

        args = Args()
        args.runs    = runs
        args.epochs  = epochs
        args.c_lr    = c_lr
        args.c_wd    = c_wd
        args.e_lr    = e_lr
        args.e_wd    = e_wd
        args.dropout = dropout
        args.hidden  = hidden
        args.seed    = seed
        args.encoder = encoder
        args.alpha   = alpha
        args.warmup  = warmup
        args.eta     = eta
        args.device  = torch.device(device)
        args.num_features = data.x.shape[1]
        args.num_classes  = 1
        args.sens_idx     = getattr(data, 'sens_idx', 0)

        self.args = args
        self.data = data

        neighbor_dist_list = get_ins_neighbor_dist(
            data.y.size(0), data.edge_index, args.device)

        data = data.to(args.device)
        n_cls = data.y.max().int().item() + 1
        n_sen = data.sens.max().int().item() + 1
        index_list = torch.arange(len(data.y)).to(args.device)

        group_num_list, idx_info = [], []
        for i in range(n_cls):
            for j in range(n_sen):
                mask = ((data.y == i) & (data.sens == j) & data.train_mask)
                group_num_list.append(int(mask.sum().item()))
                idx_info.append(index_list[mask])

        encoder_m, classifier, optimizer_e, optimizer_c = get_enc_cls_opt(args)

        best_val_tradeoff = 0
        best_acc = best_f1 = best_auc = best_sp = best_eo = 0
        self.best_epoch = -1        # stays -1 if the floor is never exceeded

        for epoch in range(args.epochs):
            encoder_m.train()
            classifier.train()
            optimizer_c.zero_grad()
            optimizer_e.zero_grad()

            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(
                group_num_list, idx_info, args.eta)
            beta = torch.distributions.beta.Beta(2, 2)
            lam  = beta.sample((len(sampling_src_idx),)).unsqueeze(1).to(args.device)

            # CNM: counterfactual node mixup. Off -> the else branch below,
            # which is this method's own plain-training path.
            if use_cnm and epoch >= args.warmup:
                new_edge_index = neighbor_sampling(
                    data.x.size(0), data.edge_index, sampling_src_idx, neighbor_dist_list)
                new_x  = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam)
                h      = encoder_m(new_x, new_edge_index)
                output = classifier(h)

                add_num = output.shape[0] - data.train_mask.shape[0]
                new_train_mask = torch.cat([
                    torch.zeros(data.train_mask.shape[0], dtype=torch.bool, device=args.device),
                    torch.ones(add_num, dtype=torch.bool, device=args.device)], dim=0)

                loss_src = F.binary_cross_entropy_with_logits(
                    output[new_train_mask],
                    data.y[sampling_src_idx].unsqueeze(1).to(args.device), reduction='none')
                loss_dst = F.binary_cross_entropy_with_logits(
                    output[new_train_mask],
                    data.y[sampling_dst_idx].unsqueeze(1).to(args.device), reduction='none')

                pos_grad_src = (1. - torch.exp(-loss_src).detach()) * lam
                pos_grad_dst = (1. - torch.exp(-loss_dst).detach()) * (1 - lam)
                grad_count = []
                for i in range(n_cls):
                    for j in range(n_sen):
                        m_src = (data.y[sampling_src_idx] == i) & (data.sens[sampling_src_idx] == j)
                        m_dst = (data.y[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                        grad_count.append(
                            pos_grad_src[m_src].sum().item() + pos_grad_dst[m_dst].sum().item())

                # CAL: contribution alignment. Off -> losses keep their
                # unweighted values and only the mixup remains.
                if use_cal:
                    min_grad = np.min(grad_count)
                    group_weight_list = [float(min_grad) / (float(n) + EPS)
                                         for n in grad_count]

                    for i in range(n_cls):
                        for j in range(n_sen):
                            # Reproduced, not repaired: this line indexes y by
                            # sampling_src_idx and sens by sampling_dst_idx,
                            # while the grad_count loop above uses src for both.
                            # Left as published.
                            m_src = (data.y[sampling_src_idx] == i) & (data.sens[sampling_dst_idx] == j)
                            m_dst = (data.y[sampling_dst_idx] == i) & (data.sens[sampling_dst_idx] == j)
                            loss_src[m_src] *= group_weight_list[i * 2 + j]
                            loss_dst[m_dst] *= group_weight_list[i * 2 + j]

                loss = (lam * loss_src + (1 - lam) * loss_dst).mean()
                loss.backward()
            else:
                h      = encoder_m(data.x, data.edge_index)
                output = classifier(h)
                loss_c = F.binary_cross_entropy_with_logits(
                    output[data.train_mask],
                    data.y[data.train_mask].unsqueeze(1).to(args.device))
                loss_c.backward()

            optimizer_e.step()
            optimizer_c.step()

            self._enc_for_inference, self._cls_for_inference = encoder_m, classifier
            accs, auc_rocs, F1s, tmp_parity, tmp_equality, _out = evaluate_ged3(
                classifier, encoder_m, data, return_output=True)

            if trajectory is not None:
                if trajectory.state_fn is None:
                    # The method names its own inference path. encoder_m and
                    # classifier both contain BatchNorm1d, so state_dict -- which
                    # carries running_mean and running_var -- is required, not
                    # just the parameters.
                    from core.trajectory import inference_state as _inf
                    trajectory.state_fn = lambda: _inf(encoder=encoder_m,
                                                       classifier=classifier)
                vm = data.val_mask
                trajectory.add(epoch,
                               _out[vm].squeeze().detach().cpu().numpy(),
                               stage="classifier",
                               # the composite selector's own inputs, as this
                               # code computed them; the replay recomputes the
                               # same formula from unified metrics and the two
                               # are compared
                               extra={"code_auc": auc_rocs['val'],
                                      "code_f1": F1s['val'],
                                      "code_acc": accs['val'],
                                      "code_dp": tmp_parity['val'],
                                      "code_eo": tmp_equality['val'],
                                      "alpha_select": float(args.alpha)})

            score = (auc_rocs['val'] + F1s['val'] + accs['val']
                     - args.alpha * (tmp_parity['val'] + tmp_equality['val']))
            if score > best_val_tradeoff:
                self.best_epoch = epoch     # exposed for selector replay
                best_val_tradeoff = score
                self.val_loss     = -accs['val']
                best_acc = accs['test']
                best_auc = auc_rocs['test']
                best_f1  = F1s['test']
                best_sp, best_eo = tmp_parity['test'], tmp_equality['test']

                # 테스트셋 상세 결과 저장
                encoder_m.eval()
                classifier.eval()
                with torch.no_grad():
                    emb    = encoder_m(data.x, data.edge_index)
                    out    = classifier(emb)
                preds_all = (out.squeeze() > 0).long().detach().cpu().numpy()
                labels_np = data.y.cpu().numpy()
                sens_np   = data.sens.cpu().numpy()
                idx_test  = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()

                self._preds_test  = preds_all[idx_test]
                self._labels_test = labels_np[idx_test]
                self._sens_test   = sens_np[idx_test]

                # ── Weakness 3 rebuttal metrics: snapshot at the SAME
                # best-epoch point as the above, since `emb`/`out` are
                # otherwise discarded once the loop moves to the next
                # epoch. Stored for ALL nodes (not just idx_test) so
                # individual_fairness_consistency can search neighbors
                # across the full node set.
                self.temp_embeddings = emb.detach()
                self.temp_probs      = torch.sigmoid(out.squeeze()).detach()
                self.temp_idx_train  = data.train_mask.nonzero(as_tuple=False).view(-1)

        self._best = (best_acc, best_auc, best_f1, best_sp, best_eo)

    def inference_modules(self):
        """The modules a saved InferenceState restores into.

        Exposed so a caller can evaluate an arbitrary checkpoint -- the common
        selector may name an epoch this method's own selector never kept.
        """
        return dict(encoder=self._enc_for_inference,
                    classifier=self._cls_for_inference)

    def predict(self):
        best_acc, best_auc, best_f1, best_sp, best_eo = self._best
        pred   = self._preds_test
        labels = self._labels_test
        sens   = self._sens_test

        ACC_sens0, AUC_sens0, F1_sens0, ACC_sens1, AUC_sens1, F1_sens1 = \
            predict_sens_group(pred, labels, sens,
                               np.arange(len(pred)))  # idx는 이미 test만

        return (best_acc, best_auc, best_f1,
                ACC_sens0, AUC_sens0, F1_sens0,
                ACC_sens1, AUC_sens1, F1_sens1,
                best_sp, best_eo)

    def predict_extra_fairness(self, idx_test):
        """
        Weakness 3 rebuttal metrics, computed from the best-epoch snapshot
        (self.temp_embeddings / self.temp_probs) captured inside fit().
        """
        if not hasattr(self, "temp_embeddings"):
            raise RuntimeError(
                "No embedding snapshot found. Re-run fit() with the "
                "patched best-epoch snapshot block before calling this."
            )
        ifair = individual_fairness_consistency(
            self.temp_probs, self.data.x, idx_test, k=5)
        leak = sensitive_leakage_probe(
            self.temp_embeddings, self.data.sens, self.temp_idx_train, idx_test)
        return {**ifair, **leak}