import logging

import numpy as np
import os
import time
import torch
from config import cfg
from models.network import BagReID_IBN
from utils.re_ranking import re_ranking as re_ranking_func
from train import build_data_loader
from tensorboardX import SummaryWriter

logger = logging.getLogger('global')

class Evaluator:
    def __init__(self, model, epoch):
        self.model = model
        self.time = time.time()
        self.epoch = i
        if cfg.TRAIN.LOG_DIR:
            self.summary_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
        else:
            summary_writer = None

    def evaluate(self, queryloader, galleryloader, ranks=[1, 3, 5, 10],re_ranking=False):
        self.model.eval()
        qf = []
        imgs_id = []
        imgs_camid=[]
        for inputs in queryloader:
            img, img_id, img_camid = self._parse_data(inputs)
            img_hflip = self.flip_horizontal(img)
            img_vflip = self.flip_vertical(img)
            img_hvflip = self.flip_vertical(img_hflip)
            feature = self._forward(img)
            feature_hflip = self._forward(img_hflip)
            feature_vflip = self._forward(img_vflip)
            feature_hvflip = self._forward(img_hvflip)
            qf.append(torch.max(feature,
                                torch.max(feature_vflip, torch.max(feature_hflip, feature_hvflip))))
            imgs_id.extend(map(int,img_id))
            imgs_camid.extend(img_camid)
        qf = torch.cat(qf, 0)
        #print(imgs_id)
        #print(imgs_camid)
        q_pids = torch.Tensor(imgs_id)
        q_camids = torch.Tensor(imgs_camid)
        logger.info("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf = []
        g_bagids = []
        g_camids = []
        for inputs in galleryloader:
            img, bagid, camid = self._parse_data(inputs)
            img_hflip = self.flip_horizontal(img)
            img_vflip = self.flip_vertical(img)
            img_hvflip = self.flip_vertical(img_hflip)
            feature = self._forward(img)
            feature_hflip = self._forward(img_hflip)
            feature_vflip = self._forward(img_vflip)
            feature_hvflip = self._forward(img_hvflip)
            gf.append(torch.max(feature,
                                torch.max(feature_vflip, torch.max(feature_hflip, feature_hvflip))))
            g_bagids.extend(map(int,bagid))
            g_camids.extend(camid)
        gf = torch.cat(gf, 0)
        g_pids = torch.Tensor(g_bagids)
        g_camids = torch.Tensor(g_camids)
        logger.info("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        logger.info("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        if re_ranking:
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                       torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                       torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist, k1=5, k2=5, lambda_value=0.3))
        else:
            distmat = q_g_dist

        print("Computing CMC and mAP")
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")
        if self.summary_writer:
            self.summary_writer.add_scalar('mAP', mAP, self.epoch)
            self.summary_writer.add_scalars('Rank', {'Rank1': cmc[0], 'Rank3': cmc[2], 'Rank5': cmc[4], 'Rank10': cmc[9]}, self.epoch)
        self._writer(cmc, mAP)
        
        return cmc[0]
        
    def _writer(self, cmc, mAP, ranks=[1, 3, 5, 10]):
        print("result is writing--------")
        with open(cfg.EVA.OUTPUT, 'a') as fi:
            fi.write("Results ----------\n")
            fi.write("mAP: {:.1%}\n".format(mAP))
            fi.write("CMC curve\n")
            for r in ranks:
                fi.write("Rank-{:<3}: {:.1%}\n".format(r, cmc[r - 1]))
            fi.write("--------------------------\n")
            fi.write("\n\n")
        print('finshed   {:3f}s'.format(time.time()-self.time))

    def _parse_data(self, inputs):
        imgs, bad_ids, camids = inputs
        return imgs.cuda(), bad_ids, camids

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()

    def flip_horizontal(self, image):
        '''flip horizontal'''
        inv_idx = torch.arange(image.size(3) - 1, -1, -1, dtype=torch.int64)  # N x C x H x W
        if cfg.CUDA:
            inv_idx = inv_idx.cuda()
        img_flip = image.index_select(3, inv_idx)
        return img_flip

    def flip_vertical(self, image):
        '''flip vertical'''
        inv_idx = torch.arange(image.size(2) - 1, -1, -1, dtype=torch.int64)  # N x C x H x W
        if cfg.CUDA:
            inv_idx = inv_idx.cuda()
        img_flip = image.index_select(2, inv_idx)
        return img_flip
    def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view([num_q, -1]) 
        keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices]  == q_camids.view([num_q, -1])))
        #keep = g_camids[indices]  != q_camids.view([num_q, -1])

        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank+1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP



if __name__ == '__main__':
    dataset, _, query_loader, gallery_loader = build_data_loader()
    model = BagReID_IBN(dataset.num_train_pids, dataset.num_train_mates)
    for i in range(10, 101, 10):
        print('this is the {} epoch'.format(i))
        with open(cfg.EVA.OUTPUT, 'a') as fi:
            fi.write('this is the {} epoch\n'.format(i))
        pre_data = cfg.EVE_PATH.format(i)
        model_paths = {'resnet50_ibn_a': pre_data}
        model.load_state_dict(torch.load(model_paths['resnet50_ibn_a'], map_location='cpu')['state_dict'])
        model.cuda()
        evaluator = Evaluator(model, i)
        evaluator.evaluate(query_loader, gallery_loader, re_ranking=True)
