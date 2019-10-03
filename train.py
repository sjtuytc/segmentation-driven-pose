import torch.utils.data
import numpy as np
from utils import *
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from ycb_dataset import YCB_Dataset
from segpose_net import SegPoseNet
from darknet import Darknet
from pose_2d_layer import Pose2DLayer
from pose_seg_layer import PoseSegLayer
from tensorboardX import SummaryWriter
opj = os.path.join
import argparse
from tqdm import tqdm

# choose dataset/env/exp info
dataset = 'YCB-Video'
test_env = 'sjtu'
exp_id = '007'
print(exp_id, test_env)
# Paths
if test_env == 'sjtu':
    ycb_root = "/media/data_2/YCB"
    imageset_path = '/media/data_2/YCB/ycb_video_data_share/image_sets'

ycb_data_path = opj(ycb_root, "data")
syn_data_path = opj(ycb_root,"data_syn")
kp_path = "./data/YCB-Video/YCB_bbox.npy"
weight_path = "./model/exp" + exp_id + ".pth"
load_weight_from_path = "./model/exp006.pth"

# Device configuration
if test_env == 'sjtu':
    cuda_visible = "0,1,2,3"
    gpu_id = [0, 1, 2, 3]
    batch_size = 32
    num_workers = 10
    use_real_img = True
    num_syn_img = 0
    bg_path = "/media/data_2/VOCdevkit/VOC2012/JPEGImages"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
initial_lr = 0.001
momentum = 0.9
weight_decay = 5e-4
num_epoch = 30
use_gpu = True
gen_kp_gt = False
number_point = 8
modulating_factor = 1.0

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

# Loss configurations
seg_loss = nn.CrossEntropyLoss()
pos_loss = nn.L1Loss()
pos_loss_factor = 1.8 # 0.02 in original paper
conf_loss = nn.L1Loss()
conf_loss_factor = 0.8 # 0.02 in original paper

# summary writer
if test_env =="sjtu":
    writer = SummaryWriter(log_dir='./log'+exp_id, comment='training log')
else:
    writer = SummaryWriter(logdir='./log' + exp_id, comment='training log')

def train(data_cfg):
    data_options = read_data_cfg(data_cfg)
    m = SegPoseNet(data_options)
    if load_weight_from_path is not None:
        m.load_weights(load_weight_from_path)
        print("Load weights from ", load_weight_from_path)
    i_h = m.height
    i_w = m.width
    o_h = m.output_h
    o_w = m.output_w
    # m.print_network()
    m.train()
    bias_acc = meters()
    optimizer = torch.optim.SGD(m.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.5*num_epoch), int(0.75*num_epoch),
                                                                 int(0.9*num_epoch)], gamma=0.1)
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
        m = torch.nn.DataParallel(m, device_ids=gpu_id)
        m.cuda()

    train_dataset = YCB_Dataset(ycb_data_path, imageset_path, syn_data_path=syn_data_path, target_h=o_h, target_w=o_w,
                      use_real_img=use_real_img, bg_path=bg_path, num_syn_images=num_syn_img,
                                data_cfg="data/data-YCB.cfg", kp_path=kp_path)
    median_balancing_weight = train_dataset.weight_cross_entropy.cuda() if use_gpu \
        else train_dataset.weight_cross_entropy

    print('training on %d images'%len(train_dataset))
    if gen_kp_gt:
        train_dataset.gen_kp_gt()

    # Loss configurations
    seg_loss = nn.CrossEntropyLoss(weight=median_balancing_weight)
    pos_loss = nn.L1Loss()
    pos_loss_factor = 1.3  # 0.02 in original paper
    conf_loss = nn.L1Loss()
    conf_loss_factor = 0.8  # 0.02 in original paper

    # split into train and val
    train_db, val_db = torch.utils.data.random_split(train_dataset, [len(train_dataset)-2000, 2000])


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # not use validation now
                                               batch_size=batch_size, num_workers=num_workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_db,
                                               batch_size=batch_size,num_workers=num_workers,
                                               shuffle=True)
    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epoch):
        i=-1
        for images, seg_label, kp_gt_x, kp_gt_y, mask_front in tqdm(train_loader):
            i += 1
            if use_gpu:
                images = images.cuda()
                seg_label = seg_label.cuda()
                kp_gt_x = kp_gt_x.cuda()
                kp_gt_y = kp_gt_y.cuda()
                mask_front = mask_front.cuda()

            # forward pass
            output = m(images)

            # segmentation
            pred_seg = output[0] # (BxOHxOW,C)
            seg_label = seg_label.view(-1)

            l_seg =seg_loss(pred_seg, seg_label)

            # regression
            mask_front = mask_front.repeat(number_point,1, 1, 1).permute(1,2,3,0).contiguous() # (B,OH,OW,NV)
            pred_x = output[1][0] * mask_front # (B,OH,OW,NV)
            pred_y = output[1][1] * mask_front
            kp_gt_x = kp_gt_x.float() * mask_front
            kp_gt_y = kp_gt_y.float() * mask_front
            l_pos = pos_loss(pred_x, kp_gt_x) + pos_loss(pred_y, kp_gt_y)

            # confidence
            conf = output[1][2] * mask_front # (B,OH,OW,NV)
            bias = torch.sqrt((pred_y-kp_gt_y)**2 + (pred_x-kp_gt_x)**2)
            conf_target = torch.exp(-modulating_factor * bias) * mask_front
            conf_target = conf_target.detach()
            l_conf = conf_loss(conf, conf_target)

            # combine all losses
            all_loss = l_seg + l_pos * pos_loss_factor + l_conf * conf_loss_factor
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                # compute pixel-wise bias to measure training accuracy
                bias_acc.update(abs(pnz((pred_x - kp_gt_x).cpu()).mean()*i_w))
                print('Epoch [{}/{}], Step [{}/{}]: \n seg loss: {:.4f}, pos loss: {:.4f}, conf loss: {:.4f}, '
                      'Pixel-wise bias:{:.4f}'
                      .format(epoch + 1, num_epoch, i + 1, total_step, l_seg.item(), l_pos.item(),
                              l_conf.item(), bias_acc.value))

                writer.add_scalar('seg_loss', l_seg.item(), epoch*total_step+i)
                writer.add_scalar('pos loss', l_pos.item(), epoch*total_step+i)
                writer.add_scalar('conf_loss', l_conf.item(), epoch*total_step+i)
                writer.add_scalar('pixel_wise bias', bias_acc.value, epoch*total_step+i)
        bias_acc._reset()
        scheduler.step()
        if epoch % 5 == 1:
            m.module.save_weights(weight_path)
    m.module.save_weights(weight_path)
    writer.close()

if __name__ == '__main__':
    if dataset == 'Occluded-LINEMOD':
        # intrinsics of LINEMOD dataset
        k_linemod = np.array([[572.41140, 0.0, 325.26110],
                              [0.0, 573.57043, 242.04899],
                              [0.0, 0.0, 1.0]])
        # # 8 objects for LINEMOD dataset
        # object_names_occlinemod = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
        # vertex_linemod = np.load('./data/Occluded-LINEMOD/LINEMOD_vertex.npy')
        # train('./data/data-LINEMOD.cfg',
        #                      './model/occluded-linemod.pth',
        #                      './occluded-linemod-testlist.txt',
        #                      outdir, object_names_occlinemod, k_linemod, vertex_linemod,
        #                      bestCnt=10, conf_thresh=0.3, linemod_index=True)
        #
        # rt_transforms = np.load('./data/Occluded-LINEMOD/Transform_RT_to_OccLINEMOD_meshes.npy')
        # transform_pred_pose(outdir, object_names_occlinemod, rt_transforms)
    elif dataset == 'YCB-Video':
        # 21 objects for YCB-Video dataset
        object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                                 '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                                 '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                                 '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
        train('./data/data-YCB.cfg')
    else:
        print('unsupported dataset \'%s\'.' % dataset)
