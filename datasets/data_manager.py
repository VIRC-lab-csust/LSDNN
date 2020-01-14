import glob
import re
from os import path as osp
import os
import torch

class MVB1900(object):
    """
    MVB1900
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 20176 (train) + 1432 (gallery) + 1052 (probe)
    """
    def __init__(self, dataset_dir, train_txt, root='Image'):
        self.dataset_dir = dataset_dir
        self.train_txt = train_txt
        self.train_dir = osp.join(self.dataset_dir, 'MVB_train', root)
        self.gallery_dir = osp.join(self.dataset_dir, 'MVB_val', root, 'gallery')
        self.probe_dir = osp.join(self.dataset_dir, 'MVB_val', root, 'probe')
        
        self._check_before_run()
        train, num_train_imgs, num_train_pids = self._process_dir(self.dataset_dir, model='train')
        num_train_mates = self._get_material(self.train_txt)
        gallery, num_gallery_imgs, num_gallery_pids = self._process_dir(self.gallery_dir, model='val')
        probe, num_probe_imgs, num_probe_pids = self._process_dir(self.probe_dir, model='val')
        num_total_pids = num_train_pids + num_gallery_pids + num_probe_pids
        num_total_imgs = num_train_imgs + num_gallery_imgs + num_probe_imgs
        
        print("=> MVB1900 loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids |  # imgs  | # mate |  ")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:6d}   | {:5d}  |  ".format(num_train_pids, num_train_imgs, num_train_mates))
        print("  gallery  | {:5d} | {:6d}   | {:5d}  |  ".format(num_gallery_pids, num_gallery_imgs, 0))
        print("  probe    | {:5d} | {:6d}   | {:5d}  |  ".format(num_probe_pids, num_probe_imgs, 0))
        print("  ----------------------------------------")
        print("  total    | {:5d} | {:6d}   | {:5d}  |  ".format(num_total_pids, num_total_imgs, num_train_mates))
        print("  ----------------------------------------")
        
        self.train = train
        self.gallery = gallery
        self.probe = probe

        self.num_train_pids = num_train_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_probe_pids = num_probe_pids
        self.num_train_mates = num_train_mates
        
    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        # if not osp.exists(self.train_dir):
            # raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.probe_dir):
            raise RuntimeError("'{}' is not available".format(self.probe_dir))
        
    def _process_dir(self, dir_path, model):
        img_names = os.listdir(dir_path)
        img_paths = [osp.join(dir_path, img_name) for img_name in img_names]
        
        pid_container = set()
        
        dataset = []
        if model=='train':
            with open(self.train_txt, 'r') as file:
                files = file.readlines()[1:]
                for fe in files:
                    mate = fe.strip().split()[3]
                    mate = self._mate_in(mate)
                    pid = int(fe.strip().split()[1])
                    camid = fe.strip().split()[2]
                    path = fe.strip().split()[1] + '_' + camid + '.jpg'
                    camid = self._camid_in(camid)
                    pid_container.add(pid)
                    img_path = osp.join(self.train_dir, path)
                    pid2label = {pid: label for label, pid in enumerate(pid_container)}
                    dataset.append((img_path, pid, camid, mate))
            
            num_pids = len(pid_container)
            num_imgs = len(dataset)
            
        elif model=='val':
            for img_name in img_names:
                img_path = osp.join(dir_path, img_name)
                pid, camid = int(img_name[:4]), img_name[5:8]
                camid = self._camid_in(camid)
                pid_container.add(pid)
                pid2label = {pid: label for label, pid in enumerate(pid_container)}
                dataset.append((img_path, pid, camid))
            
            num_pids = len(pid_container)
            num_imgs = len(dataset)
            

            
        return dataset, num_imgs, num_pids
        
    def _get_material(self, txt):
        material = set()
        with open(txt, 'r') as file:
            data = file.readlines()[1:]
            for line in data:
                _, _, _, mate = line.strip().split()
                mate = self._mate_in(mate)
                material.add(mate)
        return len(material)
        
    def _camid_in(self, cam):
        if cam == 'p_1':
            return 0
        elif cam == 'p_2':
            return 1
        elif cam == 'p_3':
            return 2
        elif cam == 'p_4':
            return 3
        elif cam == 'g_1':
            return 4
        elif cam == 'g_2':
            return 5
        elif cam == 'g_3':
            return 6

    def _mate_in(self, mate):
        if mate == 'hard':
            return 0
        if mate == 'soft':
            return 1
        if mate == 'paperboard': 
            return 2
        if mate == 'others':
            return 3
        
        
def init_dataset(name, txt):
    return MVB1900(name, txt)
