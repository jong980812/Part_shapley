import matplotlib.pyplot as plt
import os
import json
from PIL import Image
import random

def shapley_class(args,part,part_count, task, class_name, num_correct, nb_data, save_path):
    plt.clf()
    data2={}
    for i in range(7):
        data2[part[i]]=(list(part_count.values())[i])/nb_data
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
    plt.title(f'{task} task  : {class_name} samples\n{num_correct}/{nb_data}={num_correct/nb_data*100:.2f}%')
    # 그래프 표시
    plt.savefig(os.path.join(save_path,f'{task}_{class_name}_{args.size}_{args.norm_type}.png'))
    
    
def shapley_task(args,part,part_count_list, task, class_name, num_correct, nb_data, save_path):
    plt.clf()
    data2={}
    for part_name in part:
        data2[part_name]=0
    for i in range(7):
        value=0
        for part_count in part_count_list:
            value+=(list(part_count.values())[i])
        data2[part[i]]=value / nb_data
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
    plt.title(f'{task} task total  : \n{num_correct}/{nb_data}={num_correct/nb_data*100:.2f}%')
    # 그래프 표시
    plt.savefig(os.path.join(save_path,f'{task}_total_{args.size}_{args.norm_type}.png'))

def representative_each_class(shapley_lists, best_part_list,task, class_names, json_path, n_show,save_path):
    '''
    shapley_list: shapley value and path of all images
    task: Task name
    class names: all class name
    json path: json path to get part information
    '''
    num_class = len(class_names)
    part=['Hair',"Eye","Nose","Ear","Mouth","Hand","Foot"]

    for class_index, class_name in enumerate(class_names):
        plt.clf()
        fig, axes = plt.subplots(num_class, n_show, figsize=(12, 10))
        plotting_order = [i for i, _ in enumerate(class_names) if i != class_index]
        best_part = best_part_list[class_name]
        shapley_list = shapley_lists[class_name]
        #가장 큰 값의 key value pair  tuple
        sorted_shapley_dict = {k: v for k, v in sorted(shapley_list.items(), key=lambda item: item[1][best_part],reverse=True)} 
        sorted_shapley_list=list(sorted_shapley_dict.keys())[:n_show]
        for i in range(num_class):
            for j in range(n_show):
                if i==0:#beset
                    axes[i, j].imshow(get_humna_body(sorted_shapley_list[j],json_path), cmap='gray')
                else:
                    img_list=shapley_lists[class_names[plotting_order[i-1]]]#another class imig lists
                    random_key = random.choice(list(img_list.keys()))
                    axes[i, j].imshow(get_humna_body(random_key,json_path), cmap='gray')
        for i in range(num_class):        
            if i ==0:
                axes[i, 0].set_title(f'Best {class_name} - {part[best_part]}', fontsize=10)
            else:
                axes[i, 0].set_title(f'{class_name} - {part[best_part]}', fontsize=10)
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(os.path.join(save_path,f'{class_name}_{part[best_part]}_vs_another.png'))

def get_humna_body(image_path,json_path):
    json_full_path =os.path.join(json_path, image_path.split('/')[-1].split('.')[0] + ".json")
    part_json = get_part_json(json_full_path)
    human_body_coords =get_coords(part_json['human_body'])
    img = Image.open(image_path)
    return img.crop(human_body_coords[0])
    
    
def get_part_json(json_file_path):
    '''
    Get part dictionary from json path
    '''
    part_json = {}
    with open(json_file_path, 'r') as f:
        boxes = json.load(f)['shapes']
        for box in boxes:
            part_json[box["label"]]=[]
        for box in boxes:
            part_json[box["label"]].append(box["points"])

    for key in part_json:#! 빈 애들은 None으로 처리해서 없다고 판단.
        if not part_json[key]:
            part_json[key] = None

    return part_json
def get_coords(part):
    extracted_coordinates = []
    if part is None:
        return None
    elif len(part) == 1:
        # print(part[0][0])
        xmin, ymin = list(map(int,part[0][0]))
        xmax, ymax = list(map(int,part[0][1]))
        return [[xmin,ymin,xmax,ymax]]#아래 2일경우와 통일하기 위해 이중 리스트로 
    elif len(part) == 2:
        #! Eye, Ear, hand, foot -> These have 2 part, return list
        for a in part: 
            # print(a)
            xmin, ymin = list(map(int,a[0]))
            xmax, ymax = list(map(int,a[1]))
            extracted_coordinates.append([xmin,ymin,xmax,ymax])
        return extracted_coordinates
    else:
        for a in part: 
            # print(a)
            xmin, ymin = list(map(int,a[0]))
            xmax, ymax = list(map(int,a[1]))
            extracted_coordinates.append([xmin,ymin,xmax,ymax])
        return extracted_coordinates