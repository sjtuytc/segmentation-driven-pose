import os
from os.path import join as opj

def collect_occluded_linemod_testlist(rootpath, outname):
    path = rootpath + 'RGB-D/rgb_noseg/'
    imgs = [f for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    imgs.sort()
    # write sets
    allf = open(outname, 'w')
    for i in imgs:
        allf.write(path + i +'\n')

def collect_ycb_testlist(testListFile, rootpath, outfile):

    print("root path is ", rootpath)
    with open(testListFile, 'r') as file:
        testlines = file.readlines()
    print("total file number", len(testlines))
    with open(outfile, 'w') as file:
        for l in testlines:
            file.write(opj(rootpath,l.rstrip()+'-color.png')+'\n')

if __name__ == '__main__':
    # modify the path according to your real path of the Occluded-LINEMOD and YCB-Video dataset
    #
    # occluded_linemod_path = '/data/OcclusionChallengeICCV2015/'
    # collect_occluded_linemod_testlist(occluded_linemod_path, './occluded-linemod-testlist.txt')

    print("Generate path of YCB...")
    # ycb_video_path = '/media/data_2/YCB/data'
    # testListFile = '/media/data_2/YCB/YCB_Video_toolbox/keyframe.txt'

    ycb_video_path = '/data/vision/billf/scratch/zelin/YCB/YCB_Video_Dataset/data'
    testListFile = "/data/vision/billf/scratch/zelin/YCB/YCB_Video_Dataset/image_sets/keyframe.txt"
    collect_ycb_testlist(testListFile, ycb_video_path, './ycb-video-testlist.txt')
    print("Generation Done!")

    # print("Generate path of our YCB...")
    # our_ycb_path = './data/our-YCB'
    # collect_our_ycb(our_ycb_path, './ycb-video-testlist.txt')
    # print("Generation Done!")