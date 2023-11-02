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
from einops import rearrange
from itertools import product
import math
import torchvision.models as models
import argparse
import torchvision
import random
        
from shapley.transform import get_transform
from shapley.dataset import Shapley_part,get_dataset_information
from shapley.get_shapley_value import get_ordered_pair,get_shapley_matrix
from shapley.visualizer import shapley_class, shapley_task, representative_each_class
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    #! custom argument
    parser.add_argument('--data_path',default='/local_datasets/ai_hub/ai_hub_sketch_mw/01/val/',type=str,
                        help='Data_path')
    parser.add_argument('--json_path',default='/data/datasets/ai_hub_sketch_json_asd_version',type=str,
                    help='Json_path')
    parser.add_argument('--out_path',default='./',type=str,
                    help='out_path')
    parser.add_argument('--save_path',default='./',type=str,
                    help='figure save path')
    #Model
    parser.add_argument('--weight', default=10, type=str,
                        help='Model weights path')
    parser.add_argument('--padding_mode',default='zeros',type=str,
                        help='Set padding of Conv2d')    
    parser.add_argument('--nb_classes',default=2,type=int,
                        help='number of head class')
    
    #Transform
    parser.add_argument('--norm_type',default='ai_hub',type=str,
                        help='set normalize value')
    parser.add_argument('--threshold', default=None, type=int,
                        help='Random seed')
    parser.add_argument('--size', nargs=2, type=int, metavar=('H', 'W'), 
                        help='Height, width')
    
    #Training
    
    parser.add_argument('--batch_size',default=10, type=int,
                        help='Set batch size')
    
    
    
    
    parser.add_argument('--seed', default=777, type=int,
                        help='Random seed')
    # parser.add_argument('--',default=True,type=bool)
    # parser.add_argument('--', action='store_true')#max acc사용할때
    # parser.add_argument('--unfreeze_layers', default=None, nargs='+', type=str)
    # parser.add_argument('--dropout', default=0.2,type=float)
    # parser.add_argument('--stochastic_depth_prob',default=0.2,type=float)
    
    return parser
    
def main(args):

    
    random_seed=777
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    model=models.efficientnet_b1(pretrained=True,progress=False)
    model.classifier[1] = torch.nn.Linear(1280, args.nb_classes)
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
    

    transform = get_transform(args)
    data_information = get_dataset_information(args)
    print(f"Transform:\n{transform}")
    print(f"Target Classes:{data_information['class_names']}")
    print(f"Target Task:{data_information['task']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device:{device}')

    model.eval()
    # load dataset

    
    # ready
    
    all_ordered_pair,weights = get_ordered_pair()
    part_count_list = []
    nb_data_list = []
    num_correct_list = []
    part_number = all_ordered_pair.shape[0]
    task = data_information['task']
    shapley_img_lists=dict()
    best_part_index = dict()
    for class_name in data_information['class_names']:
        print(f'\n#####################Target_class:{class_name} getting Shapley value#####################')
        target_path = os.path.join(args.data_path, class_name)
        dataset = Shapley_part(target_path,args.json_path,task=task,transform=transform)
        data_loader=DataLoader(dataset,args.batch_size,shuffle=False,num_workers=8)
        print(dataset)
        num_correct = 0
        part_count = {i: 0 for i in range(part_number)}
        image_and_shapley=dict()
        for new_imgs, original_image, label,img_paths in tqdm(data_loader):
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
                    img_name = img_paths[i]
                    correct_output = output[:,:,label[i]]# Take correct logits,  (b, 128), 밖에서. 
                    shapley_matrix = get_shapley_matrix(all_ordered_pair,correct_output[i])
                    shapley_contributions = shapley_matrix[:,:,1] - shapley_matrix[:,:,0] 
                    shapley_value = (shapley_contributions * 1/weights).sum(dim=1)
                    image_and_shapley[img_name]=shapley_value.detach().tolist()
                    max_part_number = (int(shapley_value.argmax()))
                    part_count[max_part_number] += 1
        shapley_img_lists[class_name]=image_and_shapley
        best_part_index[class_name] = max(part_count, key=part_count.get)
        acc = num_correct/len(dataset)
        print(f'Shapley result\n:{part_count}')
        print(f'Inference\n:{num_correct}/{len(dataset)} = {acc}')
        # 주어진 딕셔너리
        part=['Hair',"Eye","Nose","Ear","Mouth","Hand","Foot"]
        num_correct_list.append(num_correct)
        part_count_list.append(part_count) # For Total shapley
        nb_data_list.append(len(dataset)) # For Total shapley
        shapley_class(args,
                    part = part, 
                    part_count= part_count, 
                    task= data_information['task'],
                    class_name = class_name,
                    num_correct=num_correct,
                    nb_data=len(dataset),
                    save_path=args.save_path)
    shapley_task(args,
                part = part, 
                part_count_list= part_count_list, 
                task= data_information['task'],
                class_name = class_name,
                num_correct=sum(num_correct_list),
                nb_data=sum(nb_data_list),
                save_path=args.save_path)
    representative_each_class(shapley_lists=shapley_img_lists,
                              best_part_list=best_part_index,
                              task=task,
                              class_names=data_information['class_names'],
                              n_show=5,
                              save_path=args.save_path,
                              json_path=args.json_path)



if __name__=="__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
    #! setting #!
    

        
