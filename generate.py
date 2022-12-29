import torch
import os
import time
from diffusers import StableDiffusionPipeline
from dataset.dataset import CompositionDataset
from utils import *



def Generate(pipe, prompt, save_imgpath, n_shot):
    os.makedirs(save_imgpath, exist_ok=True)
    for i in range(n_shot):
        image = pipe(prompt).images[0]  
        image.save(os.path.join(save_imgpath, prompt.split(' ')[-2] + "_" + prompt.split(' ')[-1] + ".jpg"))


def Generate_overlap(unseen_scores, unseen_mask):
    unseen_scores_vals = unseen_scores.topk(k=500, dim=0)[0]
    unseen_scores = (unseen_scores >= unseen_scores_vals[-1]).float()
    overlap = unseen_scores + unseen_mask
    print(len(torch.where(overlap==2)[0]), len(torch.where(unseen_mask > 0)[0]))
    return len(torch.where(overlap==2)[0]) / len(torch.where(unseen_mask > 0)[0])



if __name__=="__main__":
    print('loading test dataset')
    test_dataset = CompositionDataset("../../../dataset/mit-states", phase='test', split='compositional-split-natural', open_world=True)

    feasibility_path = 'dataset/feasibility_mit-states.pt'
    unseen_scores = torch.load(feasibility_path, map_location='cpu')['feasibility']
    unseen_mask = test_dataset.unseen_mask
    # seen_mask = test_dataset.seen_mask
    # unseen_scores_mask = (unseen_scores > 0).float()
    # unseen_scores_mask += seen_mask
    unseen_mask_idx = unseen_scores.topk(k=500, dim=0)[1]
    overlap = Generate_overlap(unseen_scores, unseen_mask)

    # mask_idx = torch.where(mask==1)[0]
    print("Generate {} pairs, and the overlop with unseen classes is {}".format(len(unseen_mask_idx), overlap))
    Generate_pairs = [test_dataset.open_pairs[idx] for idx in unseen_mask_idx]

    pipe = StableDiffusionPipeline.from_pretrained("../Stable-Diffusion/stable-diffusion-v1-5")
    pipe = pipe.to("cuda")
    for pair in Generate_pairs:
        print("Generating a photo of " + pair[0] + " " + pair[1])
        Generate(pipe, "a photo of " + pair[0] + " " + pair[1], "./IMGS", 2)

