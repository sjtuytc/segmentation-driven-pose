import os
import pickle
import numpy as np
from metric import add_err, projection_error_2d
from utils import meters
from tqdm import tqdm
from scipy.io import loadmat
import glob
import random
opj = os.path.join

class YCB_evaluator:
    def __init__(self, reload=False, ycb_root="/media/data_2/YCB", model_npy = "data/YCB-Video/YCB_vertex.npy",
        pose_pkl = "data/ycb_pose_gt.pkl"):
        self.root = ycb_root
        self.pose_pkl = pose_pkl
        if reload:
            self._load_pose_gt()
        else:
            self._load_from_pkl() # load all gts
        self.vertices = np.load(model_npy)

        self.object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
                                 '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
                                 '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser',
                                 '024_bowl', '025_mug', '035_power_drill',
                                 '036_wood_block', '037_scissors', '040_large_marker', '051_large_clamp',
                                 '052_extra_large_clamp',
                                 '061_foam_brick']
        self.ycb_class_to_idx = {}
        for i, item in enumerate(self.object_names_ycbvideo):
            self.ycb_class_to_idx[item] = i

        self.camera = np.array([[1.06677800e+03, 0.00000000e+00, 3.12986900e+02],
                               [0.00000000e+00, 1.06748700e+03, 2.41310900e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        self.avg_add_err = meters()
        self.avg_add_acc = meters()
        self.avg_rep_err = meters()
        self.avg_rep_acc = meters()

        self.diameters = []
        self._load_diameters()

    def evaluate_one(self, est_pose, class_name=None, image_id="0048_001160"):
        if class_name is None or class_name not in self.ycb_class_to_idx.keys():
            print("Error! Class name not specified!")
            return
        else:
            class_id = self.ycb_class_to_idx[class_name]
        if image_id not in self.pose_gt[class_id].keys():
            # print("Missing gt for", class_id, image_id)
            return
        gt_pose = self.pose_gt[class_id][image_id]
        model_3d = self.vertices[class_id]

        tmp_add_err = add_err(gt_pose, est_pose, model_3d)
        tmp_rep_err = projection_error_2d(gt_pose, est_pose, model_3d, self.camera)

        if tmp_add_err>500:
            return

        # calculate average
        self.avg_add_err.update(tmp_add_err)
        self.avg_add_acc.update(tmp_add_err< 0.1*self.diameters[class_id])
        self.avg_rep_err.update(tmp_rep_err)
        self.avg_rep_acc.update(tmp_rep_err<5)

    def _reset(self):
        self.avg_add_err._reset()
        self.avg_add_acc._reset()
        self.avg_rep_err._reset()
        self.avg_rep_acc._reset()

    def _load_from_pkl(self):
        assert os.path.exists(self.pose_pkl) == True, ".pkl file doesn't exist"
        assert os.path.getsize(self.pose_pkl) > 0, ".pkl file corrupted"
        with open(self.pose_pkl, 'rb') as handle:
            self.pose_gt = pickle.load(handle)
        return self.pose_gt

    def _load_diameters(self, diameter_path = "data/ycb_diameter.pkl"):
        with open(diameter_path, 'rb') as handle:
            self.diameters = pickle.load(handle)


    def _load_class_name(self):
        with open(os.path.join(self.root, 'classes.txt')) as f:
            content = f.readlines()
            self.names = [x.strip() for x in content]

    def _load_pose_gt(self, list_file="ycb-video-testlist.txt"):
        with open(list_file) as f:
            content = f.readlines()
            lists = [x.rstrip("-color-.jpg\n") and x.strip("-color-.png\n") for x in content]
        print("%d gt files loaded."%len(content))
        gt = {x:{} for x in range(21)} # gt: class, img_id
        for item in tqdm(lists):
            img_id = item[-11:-7] + '_' + item[-6:]
            meta = loadmat(item + '-meta.mat')
            poses = meta['poses'].transpose(2, 0, 1)
            idxs = meta['cls_indexes'] - 1 # change idx to start with 0, now maxidx = 20
            for i in range(len(idxs)): # cover all gt classes
                if img_id not in gt[int(idxs[i])].keys():  # add gt of current img
                    gt[int(idxs[i])][img_id] = poses[i] # 1 instance per img

            with open(self.pose_pkl, 'wb') as output:
                pickle.dump(gt, output)
            self.pose_gt = gt
        return gt

    def _cal_diameter(self, diameter_path = "data/ycb_diameter.pkl"):
        sample = 2000
        diameters = []
        for idx, class_vertices in enumerate(self.vertices):
            print("calculating diameters for class", idx)
            class_vertices = np.array(class_vertices)
            dis = []
            for k in range(30):
                A_vertices = random.sample(list(range(1, len(class_vertices))), sample)
                B_vertices = random.sample(list(range(1, len(class_vertices))), sample)
                for i in range(sample):
                    dis.append(np.linalg.norm(class_vertices[A_vertices][i] - class_vertices[B_vertices][i]))
            diameter = max(dis)
            diameters.append(diameter)
        self.diameters = diameters
        with open(diameter_path, 'wb') as output:
            pickle.dump(diameters, output)

    def get_result(self):
        return {"add acc":self.avg_add_acc.value, "REP acc":self.avg_rep_acc.value}

    def print_current(self, idx, total):
        print(idx,"/",total,":", "ADD acc:%0.3f,"%self.avg_add_acc.value,
              "ADD err:%0.3f,"%self.avg_add_err.value, "REP acc:%0.3f,"%self.avg_rep_acc.value,
              "REP err:%0.3f."%self.avg_rep_err.value)

if __name__ == "__main__":
    evaluator = YCB_evaluator(reload=True)
    ycb_result_path = "exp004-Result"
    print("evaluating path:", ycb_result_path)
    evaluator._load_pose_gt()    # use this line to load new pose gt
    evaluator._load_diameters()  # use this line to calculate all diameters

    display_interval = 200
    results = {}

    for class_path in glob.glob(ycb_result_path+"/*"):
        class_name = class_path[class_path.rfind("/")+1:]
        print("Evaluating class:", class_name)
        evaluator._reset()
        total_file_per_class = len(glob.glob(class_path + "/*"))
        for idx, img_path in enumerate(glob.glob(class_path + "/*")):
            img_id = img_path[img_path.rfind("/")+1:img_path.rfind("/")+12]
            pred_pose = np.loadtxt(img_path)
            evaluator.evaluate_one(pred_pose,class_name,img_id)
            if idx % display_interval == display_interval-1:
                evaluator.print_current(idx, total_file_per_class)
                results[class_name] = evaluator.get_result()

    print("Final results of all classes:")
    for class_name in evaluator.object_names_ycbvideo:
        print(class_name, results[class_name])
