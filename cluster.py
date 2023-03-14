import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor


import sys
sys.path.insert(1,"/vulcanscratch/gihan/lost")
import object_discovery
import networks

from torch.utils.data.dataloader import default_collate

def doArgparse():
    args = argparse.ArgumentParser()
    args.add_argument("--outputFolder","-o",type=str,default="./out/")
    args.add_argument("--vit","-v",type=str,default="vit_small")
    args.add_argument("--patchSize",type=int,default=8)

    args.add_argument("--pathToDataset",default="/fs/vulcan-datasets/tiny_imagenet/train/")
    args.add_argument("--device",default="cuda")
    args.add_argument("--batchSize",default=64)
    args.add_argument("--segmentationAlgo",default="lost",choices=["lost","dino"])
    args.add_argument("--qkv",default="k",choices=["q","k","v"])

    args.add_argument("--k_patches",
        type=int,
        default=100,
        help="Number of patches with the lowest degree considered.")


    args =  args.parse_args()
    return args

if __name__=="__main__":
    args = doArgparse()

    DATASET_PATH = args.pathToDataset
    VIT_MODEL = args.vit
    DEVICE = args.device
    PATCH_SIZE = args.patchSize
    RESNET_DIALATE=2 #I have not idea why this exist.
    BATCH_SIZE = args.batchSize
    SEGMENTATION_ALGO = args.segmentationAlgo
    K_PATCHES = args.k_patches

    # vit = torch.hub.load('facebookresearch/dino:main', VIT_MODEL)

    vit = networks.get_model(VIT_MODEL, PATCH_SIZE, RESNET_DIALATE, DEVICE)

    print(vit)    

    datasetImageNet1k = torchvision.datasets.ImageFolder(DATASET_PATH,transform=ToTensor())
    dataLoaderImagenet1k = DataLoader(datasetImageNet1k, batch_size=256, shuffle=True,\
                                      collate_fn=lambda x: tuple(x_.to(DEVICE) for x_ in default_collate(x)))



    for batchID, (X, Y) in enumerate(dataLoaderImagenet1k):
        
        # yHat = vit(X)
        imgSize = X.shape[1:]
        imgH = X.shape[2]
        imgW = X.shape[3]


        featureMapH = imgH // PATCH_SIZE
        featureMapW = imgW // PATCH_SIZE 

        outputFeatureDict = {}
        def hookFunctionForwardQKV(module,input,output):
            outputFeatureDict["qkv"]=output
        vit._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hookFunctionForwardQKV)
        
        attentions = vit.get_last_selfattention(X)#(X[None, :, :, :])
        scales = [PATCH_SIZE, PATCH_SIZE]

        noHeads = attentions.size()[1]
        noPatches = attentions.size()[2]




        if SEGMENTATION_ALGO == "dino":
            assert False, "Not implemented yet"
        elif SEGMENTATION_ALGO == "lost":
            print("Hi!")


            qkv = (outputFeatureDict["qkv"].reshape(BATCH_SIZE, noPatches, 3, noHeads, -1 // noHeads).permute(2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(BATCH_SIZE, noPatches, -1)
            q = q.transpose(1, 2).reshape(BATCH_SIZE, noPatches, -1)
            v = v.transpose(1, 2).reshape(BATCH_SIZE, noPatches, -1)


            features = k[:, 1:, :]

            # pred, A, scores, seed = lost(
            #     feats,
            #     [w_featmap, h_featmap],
            #     scales,
            #     init_image_size,
            #     k_patches=args.k_patches,
            # )

            
            object_discovery.lost(features,[featureMapH,featureMapW],scales,K_PATCHES,imgSize)


            print("END of PROG")

            # GIHAN WAS HERE 2023-03-03
        
        # object_discovery.lost(yHat,yHat.size(),)

        # print("DEBIG: (33): ",batchID,X.size(),Y.size(),yHat.size())








    print("END of python program")