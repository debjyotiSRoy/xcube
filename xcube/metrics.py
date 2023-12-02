# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_metrics.ipynb.

# %% auto 0
__all__ = ['precision_at_k', 'rareprecision_at_k', 'precision_at_r', 'recall_at_k', 'batch_lbs_accuracy', 'accuracy', 'ndcg',
           'ndcg_at_k']

# %% ../nbs/04_metrics.ipynb 2
from fastai.data.all import *
from fastai.metrics import *

# %% ../nbs/04_metrics.ipynb 6
def precision_at_k(yhat_raw, y, k=15):
    """
        Inputs: 
            yhat_raw: activation matrix of ndarray and shape (n_samples, n_labels)
            y: binary ground truth matrix of type ndarray and shape (n_samples, n_labels)
            k: for @k metric
    """
    yhat_raw, y = to_np(yhat_raw), to_np(y)
    # num true labels in the top k predictions / k
    sortd = yhat_raw.argsort()[:,::-1]
    topk = sortd[:, :k]
    
    # get precision at k for each sample
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        vals.append(num_true_in_top_k / float(k))
    
    return np.mean(vals)

# %% ../nbs/04_metrics.ipynb 7
def rareprecision_at_k(yhat_raw, y, rare_idxs=None, k=15):
    """
        Inputs: 
            yhat_raw: activation matrix of ndarray and shape (n_samples, n_labels)
            y: binary ground truth matrix of type ndarray and shape (n_samples, n_labels)
            k: for @k metric
    """
    # import pdb; pdb.set_trace()
    y = y[:, list(rare_idxs)]
    yhat_raw = yhat_raw[:, list(rare_idxs)]
    yhat_raw, y = to_np(yhat_raw), to_np(y)
    # num true labels in the top k predictions / k
    sortd = yhat_raw.argsort()[:,::-1]
    topk = sortd[:, :k]
    
    # get precision at k for each sample
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        vals.append(num_true_in_top_k / float(k))
    
    return np.mean(vals)

# %% ../nbs/04_metrics.ipynb 8
def precision_at_r(yhat_raw, y):
    """
        Inputs: 
            yhat_raw: activation matrix of ndarray and shape (n_samples, n_labels)
            y: binary ground truth matrix of type ndarray and shape (n_samples, n_labels)
    """
    yhat_raw, y = to_np(yhat_raw), to_np(y)
    # num true labels in the top r predictions / r, where r = number of labels associated with that sample 
    sortd = yhat_raw.argsort()[:, ::-1]
    
    # get precision at r for each sample
    vals = []
    for i, sorted_activation_indices in enumerate(sortd):
        # compute the number of labels associated with this sample
        r = int(y[i].sum())
        top_r_indices = sorted_activation_indices[:r] 
        num_true_in_top_r = y[i, top_r_indices].sum()
        vals.append(num_true_in_top_r / float(r))
    
    return np.mean(vals)

# %% ../nbs/04_metrics.ipynb 9
def recall_at_k(pred_probs, true_labels, k=15):
    # num true labels in top k predictions / num true labels
    pred_probs, true_labels = to_np(pred_probs), to_np(true_labels)
    sortd = np.argsort(pred_probs)[:, ::-1]
    topk = sortd[:, :k]

    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = true_labels[i, tk].sum()
        denom = true_labels[i, :].sum()
        vals.append(num_true_in_top_k / float(denom))

    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.

    return np.mean(vals)

# %% ../nbs/04_metrics.ipynb 12
def batch_lbs_accuracy(preds, xb, len=1000, resamps=10, threshold=.5):
    preds = preds.squeeze(-1)
    tok_sl = xb.shape[2]
    acc = 0
    for _ in range(resamps):
        rnd_idxs = torch.randperm(tok_sl)[:len]
        rnd_xb = xb[:, :, rnd_idxs]
        rnd_preds = preds[:, :, rnd_idxs] 
        srtd_relv, srtd_idxs = rnd_xb[:, :, :, -1].sort(descending=True)
        srtd_preds = torch.take_along_dim(rnd_preds, srtd_idxs, dim=-1)
        sl = torch.arange(len if tok_sl > len else tok_sl, device=xb.device)
        ij = torch.cartesian_prod(sl, sl)
        idxs, = torch.nonzero(ij[:, 0] < ij[:, 1], as_tuple=True)
        ij = ij[idxs]
        # si_sj = srtd_relv[:, :, ij]
        # (si_sj[:, :, :, 0] >= si_sj[:, :, :, 1]).shape, (*srtd_relv.shape[:2], 49950)
        # torch.equal(si_sj.new_ones(*si_sj.shape[:-1]), (si_sj[:, :, :, 0] >= si_sj[:, :, :, 1]))
        pi_pj = srtd_preds[:, :, ij]
        probs_hat = torch.sigmoid(pi_pj[:, :, :, 0] - pi_pj[:, :, :, 1])
        probs_hat = (probs_hat > threshold).float()
        # acc += (pi_pj[:, :, :, 0] > pi_pj[:, :, :, 1]).float().mean(dim=-1) # earlier this was wrong
        acc += probs_hat.mean(-1) # the last axis is the token pair (more relevant, less relevant)
    return acc/resamps

# %% ../nbs/04_metrics.ipynb 13
def accuracy(xb, model):
    if len(xb.shape) != 4: xb = xb.unsqueeze(0) # add the batch dim if it is not there (0: batch, 1: lbs, 2: toks, 3: tok_id,lbl_id,score)
    btch_acc = []
    for btch_splt in torch.split(xb, 4, dim=0):
        lbs_acc = []
        for lbs_splt in torch.split(btch_splt, 100, dim=1):
            lbs_acc.append(batch_lbs_accuracy(model(lbs_splt), lbs_splt))
        # import pdb; pdb.set_trace()
        lbs_acc = torch.cat(lbs_acc, dim=-1)
        btch_acc.append(lbs_acc)
    btch_acc = torch.cat(btch_acc, dim=0)
    return btch_acc

# %% ../nbs/04_metrics.ipynb 15
def ndcg(preds, xb, k=None):
    # import pdb; pdb.set_trace()
    preds.squeeze_(-1)
    preds_rank = preds.argsort(dim=-1, descending=True).argsort(dim=-1)
    ideal_rank = xb[:, :, :, -1].argsort(dim=-1, descending=True).argsort(dim=-1)
    discnt_fac = torch.log2(preds_rank+2)
    ideal_discnt_fac = torch.log2(ideal_rank+2)
    # eps = preds.new_empty(1).fill_(1e-15)
    discntd_gain = torch.pow(2, xb[:, :, :, -1])  / (discnt_fac)
    ideal_discntd_gain = torch.pow(2, xb[:, :, :, -1])  / (ideal_discnt_fac)
    dcg = discntd_gain.sum(dim=-1)#.flatten()
    idcg = ideal_discntd_gain.sum(dim=-1)#.flatten()
    ndcg = dcg/idcg
    
    ndcg_at_k = None
    
    if k is not None:
        topk_preds, topk_preds_idxs = torch.topk(preds, k=k, dim=-1, largest=True)
        topk_preds_relv = torch.take_along_dim(xb[:, :, :, -1], topk_preds_idxs, dim=-1)
        topk_df = torch.log2(2 + torch.arange(k)).cuda()# torch.take_along_dim(discnt_fac, topk_preds_idxs, dim=-1)
        dg_at_k = torch.pow(2, topk_preds_relv) / (topk_df) # changed
        dcg_at_k = dg_at_k.sum(dim=-1)

        topk, topk_idxs = torch.topk(xb[:, :, :, -1], k=k, dim=-1, largest=True)
        idg_at_k = torch.pow(2, topk) / (topk_df) # changed
        idcg_at_k = idg_at_k.sum(dim=-1)
        
        ndcg_at_k = dcg_at_k / idcg_at_k

    return preds, preds_rank, ideal_rank, discnt_fac, ideal_discnt_fac, discntd_gain, ideal_discntd_gain, dcg, idcg, ndcg, ndcg_at_k

# %% ../nbs/04_metrics.ipynb 17
def ndcg_at_k(dset, model, k=20):
    dset = dset.unsqueeze(0)
    dset_chnked = torch.split(dset, 100, dim=1)
    ndcg_at_k_list = []
    for chunk in  dset_chnked:
        *_, ndcg_at_k = ndcg(model(chunk), chunk, k=k)
        ndcg_at_k_list.append(ndcg_at_k)
    ndcg_at_k_all = torch.cat(ndcg_at_k_list, dim=-1)
    return ndcg_at_k_all
