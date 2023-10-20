from tqdm import tqdm
import os
import PIL
from PIL import Image
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
from torchvision import datasets, transforms, models
from util.transform import ThresholdTransform,AddNoise,DetachWhite
from einops import rearrange
from itertools import product
import math
import torchvision.models as models
import argparse
import torchvision
import random

from util.transform import get_transform

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    #! custom argument
    parser.add_argument('--data_path',default='/local_datasets/ai_hub/ai_hub_sketch_mw/01/val/',type=str,
                        help='Data_path')
    
    #Model
    parser.add_argument('--weight', default=10, type=str,
                        help='Model weights path')
    parser.add_argument('--padding_mode',default='zeros',type=str,
                        help='Set padding of Conv2d')
    
    #Transform
    parser.add_argument('--norm_type',default='ai_hub',type=str,
                        help='set normalize value')
    parser.add_argument('--threshold', default=None, type=int,
                        help='Random seed')
    parser.add_argument('--size', nargs=2, type=int, metavar=('H', 'W'), 
                        help='Height, width')
    
    
    
    
    parser.add_argument('--seed', default=777, type=int,
                        help='Random seed')
    # parser.add_argument('--',default=True,type=bool)
    # parser.add_argument('--', action='store_true')#max acc사용할때
    # parser.add_argument('--unfreeze_layers', default=None, nargs='+', type=str)
    # parser.add_argument('--dropout', default=0.2,type=float)
    # parser.add_argument('--stochastic_depth_prob',default=0.2,type=float)
    
    return parser
    
def main(args):
    transform = get_transform(args)

    
    random_seed=777
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    model=models.efficientnet_b1(pretrained=True,progress=False)
    model.classifier[1] = torch.nn.Linear(1280, 5)
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.padding_mode = args.padding_mode
            
    # load model    
    checkpoint = torch.load(args.weight, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.weight)
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    

    
  
    
    # load dataset
    print(data_path)
    dataset = shapley_part(data_path,'/data/datasets/ai_hub_sketch_json_asd_version',task=task,binary_thresholding=240,transform=transform)
    data_loader=DataLoader(dataset,10,shuffle=False,num_workers=8)
    print(dataset)
    
    # ready
    model.eval()
    part_name = ["human_body","face","head","hair", "neck","eye", "nose", "ear", "mouth","pocket","arm","hand", "leg","foot", "upper_body_else_arm"]#원하는 파트
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_ordered_pair,weights = get_ordered_pair()
    part_number = all_ordered_pair.shape[0]
    part_count = {i: 0 for i in range(part_number)}
    
    num_correct = 0
    for new_imgs, original_image, label in tqdm(data_loader):
        # print(new_imgs.shape)
        input_data = new_imgs
        # print('complete')
        batch_size = input_data.shape[0]
        input_data = rearrange(input_data,  'b t c h w -> (b t) c h w')
        
        model.to(device)
        input_data = input_data.to(device)
        original_image = original_image.to(device)
        label = label.to(device)

        with torch.no_grad():
            prediction = model(original_image)
            output = model(input_data)

        output = rearrange(output, '(b t) o -> b t o', b=batch_size) # batch_size, 128, output(2)
        prediction = prediction.argmax(dim=-1)
        # print(output.shape)
        # print(label)
        
        for i in range(batch_size):
            if prediction[i] == label[i]:
                num_correct +=1
                correct_output = output[:,:,label[i]]# Take correct logits,  (b, 128), 밖에서. 
                shapley_matrix = get_shapley_matrix(all_ordered_pair,correct_output[i])
                shapley_contributions = shapley_matrix[:,:,1] - shapley_matrix[:,:,0] 
                shapley_value = (shapley_contributions * 1/weights).sum(dim=1)
                max_part_number = (int(shapley_value.argmax()))
                part_count[max_part_number] += 1
    print(part_count)
    print(num_correct)
    print(num_correct/dataset.__len__())
        
    import matplotlib.pyplot as plt
    # 주어진 딕셔너리
    part=['Hair',"Eye","Nose","Ear","Mouth","Hand","Foot"]
    data=part_count
    data2={}
    for i in range(7):
        data2[part[i]]=list(data.values())[i]
    # {0: 1139, 1: 5, 2: 3, 3: 47, 4: 5, 5: 5, 6: 2}
    # 딕셔너리의 key와 value를 각각 리스트로 추출
    x = list(data2.keys())
    y = list(data2.values())

    # 그래프 생성
    plt.bar(x, y)

    # x축과 y축에 라벨 추가
    plt.xlabel('x')
    plt.ylabel('y')

    # 그래프 제목 추가
    # plt.title(f'{num_correct}/{len(dataset)}={num_correct/len(dataset)*100}%')
    plt.title(f'{task} task  : {class_name} samples\n{num_correct}/{len(dataset)}={num_correct/len(dataset)*100:.2f}%')
    
    save_path = '/data/jong980812/project/mae/Shapley'
    # 그래프 표시
    plt.savefig(os.path.join(save_path,f'{task}_{class_name}_250_0.98_binary.png'))



if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    #! setting #!
    

        
