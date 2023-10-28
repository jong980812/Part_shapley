import os
import PIL
from PIL import Image
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
from torchvision import datasets, transforms, models
from shapley.transform import ThresholdTransform,AddNoise,DetachWhite
from einops import rearrange
from itertools import product
import math
import torchvision.models as models
model=models.efficientnet_b1(pretrained=True,progress=False)
model.classifier[1] = torch.nn.Linear(1280, 2)
import torchvision
# model=torchvision.models.resnet18()
# in_feat=model.fc.in_features
# model.fc=torch.nn.Linear(in_feat,2)
data_path='/data/datasets/asd/All_5split/01/val/TD/'
# data_path='/data/datasets/ai_hub_sketch_4way/01/val/m_w'
# data_path='/data/datasets/ai_hub/ai_hub_sketch_mw/01/val/w/'
import random
weight='/data/jong980812/project/mae/result_ver2/All_5split/binary_240/OUT/02/checkpoint-29.pth'
checkpoint = torch.load(weight, map_location='cpu')
print("Load pre-trained checkpoint from: %s" % weight)
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
msg = model.load_state_dict(checkpoint_model, strict=False)
def set_conv_padding_mode(model, padding_mode='replicate'):
  for name, layer in model.named_modules():
      if isinstance(layer, torch.nn.Conv2d):
          layer.padding_mode = padding_mode
set_conv_padding_mode(model,padding_mode='replicate')
model.eval()
def get_shapley_matrix(all_ordered_pair, correct_output):
    shapley_values = torch.zeros_like(all_ordered_pair, dtype=torch.float32)

    # 각 ordered pair에 대한 값을 가져와 shapley_values에 저장
    for a,ordered_pairs in enumerate(all_ordered_pair):
        for i, ordered_pair in enumerate(ordered_pairs):
            # ordered_pair를 인덱스로 사용하여 correct_output에서 값을 가져옴
            indices = ordered_pair  # ordered_pair를 텐서로 변환
            # print(indices)
            values1 = correct_output[int(indices[0])]
            values2 = correct_output[int(indices[1])]  # correct_output에서 해당 위치의 값 가져오기
            # print(values1,values2)
            shapley_values[a,i] = torch.cat([values1.unsqueeze(0),values2.unsqueeze(0)],dim=0)
    return shapley_values
def binary_to_decimal(binary_tuple):
    decimal_value = 0
    binary_length = len(binary_tuple)

    for i, bit in enumerate(binary_tuple):
        decimal_value += bit * (2 ** (binary_length - i - 1))

    return decimal_value
def decimal_to_binary(decimal_value, num_bits):
    binary_tuple = []
    
    for i in range(num_bits):
        bit = (decimal_value >> (num_bits - i - 1)) & 1
        binary_tuple.append(bit)
    
    return tuple(binary_tuple)
def count_ones(binary_tuple):
    count = 0
    for bit in binary_tuple:
        if bit == 1:
            count += 1
    return count
def get_ordered_pair():

    n = 6  # digit의 개수
    digits = [0, 1]  # 각 digit의 가능한 값

    # 경우의 수 생성
    part_combinations = list(product(digits, repeat=n))


    index_to_insert = 1  # 두 번째 위치에 추가하려면 인덱스 1을 사용합니다.
    all_ordered_pair=[]
    for index in range(7):
        ordered_pair=[] 
        index_to_insert = index
        for combi in part_combinations:
            insert_value = [0,1]
            new_combi_0= combi[:index_to_insert] + (insert_value[0],) + combi[index_to_insert:]
            new_combi_1= combi[:index_to_insert] + (insert_value[1],) + combi[index_to_insert:]
            ordered_pair.append([binary_to_decimal(new_combi_0),binary_to_decimal(new_combi_1)])
        all_ordered_pair.append(ordered_pair)
    all_ordered_pair=torch.Tensor(all_ordered_pair)
    num_part = (all_ordered_pair.shape[0])
    num_case = (all_ordered_pair.shape[1])
    weights = torch.zeros((num_part,num_case))
    for i in range(num_part):
        for j in range(num_case):
            # all_ordered_pair의 값 가져오기
            value = int(all_ordered_pair[i, j, 1])
            
            # 이진수로 변환
            binary_value = decimal_to_binary(value, 7)
            
            # 1의 개수 세기
            num_ones = binary_value.count(1)
            
            # num * (7 combination num) 계산
            combination = math.comb(num_part,num_ones)
            weight = num_ones * combination
            
            # 결과를 weights에 저장
            weights[i, j] = weight
    return all_ordered_pair, weights
class shapley_part(Dataset):
    def __init__(self, data_folder, json_folder,part, binary_thresholding=None, transform=None):
        self.json_folder = json_folder
        self.data_folder = data_folder
        self.binary_thresholding=binary_thresholding
        self.transform = transform
        self.part = part
        self.num_part = len(part)
        self.image_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        self.json_paths = [image_path.split('/')[-1].split('.')[0] + ".json" for image_path in self.image_paths] #! Get json path from image paths.
        print(self.image_paths)
    def get_part_json(self, json_file_path, part_name):
        '''
        Get part dictionary from json path
        '''
        part_json = {}
        
        for part in part_name:
            part_json[part] = []
        with open(json_file_path, 'r') as f:
            boxes = json.load(f)['shapes']
            for box in boxes:
                part_json[box["label"]].append(box["points"])
    
        for key in part_json:#! 빈 애들은 None으로 처리해서 없다고 판단.
            if not part_json[key]:
                part_json[key] = None

        return part_json
    def get_coords(self, part):
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
            exit(0)
    def get_white_image(self,size):
        return Image.new("RGB", size, (255, 255, 255))
    # def get_empty_face(self,img, part_imgs, part_json):
    #     '''
    #     empty_face is face detached 'eye','nose','mouth','ear'
    #     '''
    #     head_json = part_json['head']
    #     head_coords = self.get_coords(head_json)
    #     head = part_imgs['head'][0]#!
    #     white_image = self.get_white_image(img.size)
    #     white_image.paste(head,head_coords[0])
    #     for part in ['eye','nose','mouth','ear']:
    #         if part_json[part] is not None:
    #           part_coords= self.get_coords(part_json[part])
    #           part_img = part_imgs[part]
    #           if part in ['eye','ear']:   
    #               white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
    #               white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
    #           else:
    #               white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
                  
    #     return white_image 
    def get_empty_face(self,img, part_imgs, part_json):
        '''
        empty_face is face detached 'eye','nose','mouth','ear'
        '''
        head_json = part_json['head']
        head_coords = self.get_coords(head_json)
        head = part_imgs['head'][0]#!
        white_image = self.get_white_image(img.size)
        white_image.paste(head,head_coords[0])
        for part in ['eye','nose','mouth','ear']:
            if part_json[part] is not None:
              part_coords= self.get_coords(part_json[part])
              part_img = part_imgs[part]
              if part in ['eye','ear']:   
                  white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
                  white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
              else:
                  white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
        # white_image.show()
        return white_image
    def get_empty_lower_body(self,img, part_imgs, part_json):
        '''
        empty_lower_body detacched foot
        '''
        lower_body_json = part_json['lower_body']
        lower_body_coords = self.get_coords(lower_body_json)
        lower_body = part_imgs['lower_body'][0]#!
        white_image = self.get_white_image(img.size)
        white_image.paste(lower_body,lower_body_coords[0])
        if part_json["foot"] is not None:
            part_coords= self.get_coords(part_json["foot"])
            part_img = part_imgs["foot"] 
            white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
            white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
        
        return white_image.crop(lower_body_coords[0])
    def get_empty_upper_body(self,img, part_imgs, part_json):
        '''
        empty_lower_body detacched foot
        '''
        upper_body_json = part_json['upper_body']
        upper_body_coords = self.get_coords(upper_body_json)
        upper_body = part_imgs['upper_body'][0]#!
        white_image = self.get_white_image(img.size)
        white_image.paste(upper_body,upper_body_coords[0])
        if part_json["hand"] is not None:
            part_coords= self.get_coords(part_json["hand"])
            part_img = part_imgs["hand"] 
            white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
            white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
        # white_image.crop(upper_body_coords[0]).show()
        return white_image.crop(upper_body_coords[0])
    
    def create_new_images(self,img, binary_combination, part_imgs,part_json):
        #! Making New images
        original_img = img
        empty_face_active, eye_active, nose_active, ear_active, mouth_active, hand_active, foot_active = binary_combination
        # New white image

        new_image = self.get_white_image(original_img.size)
        if empty_face_active:
            new_image.paste(part_imgs["empty_face"][0],(0,0))
        # print(part_json['lower_body'][0])
        # print(part_imgs["empty_lower_body"][0].size,self.get_coords(part_json['lower_body'])[0] )
        new_image.paste(part_imgs["empty_lower_body"][0], self.get_coords(part_json['lower_body'])[0])  # 원하는 위치에 붙임
        new_image.paste(part_imgs["empty_upper_body"][0], self.get_coords(part_json['upper_body'])[0])  # 원하는 위치에 붙임
        # 각 파트 이미지를 읽어와서 새로운 이미지에 붙임
        if eye_active and (part_json["eye"] is not None):
            new_image.paste(part_imgs["eye"][0], self.get_coords(part_json['eye'])[0])  # 원하는 위치에 붙임
            new_image.paste(part_imgs["eye"][1], self.get_coords(part_json['eye'])[1])  # 원하는 위치에 붙임 
        if nose_active and (part_json["nose"] is not None):
            new_image.paste(part_imgs["nose"][0], self.get_coords(part_json['nose'])[0])  # 원하는 위치에 붙임 
        if ear_active and (part_json["ear"] is not None):
            new_image.paste(part_imgs["ear"][0], self.get_coords(part_json['ear'])[0])  # 원하는 위치에 붙임 
            new_image.paste(part_imgs["ear"][1], self.get_coords(part_json['ear'])[1])  # 원하는 위치에 붙임 
        if mouth_active and (part_json["mouth"] is not None):
            new_image.paste(part_imgs["mouth"][0], self.get_coords(part_json['mouth'])[0])  # 원하는 위치에 붙임 
        if hand_active and (part_json["hand"] is not None):
            new_image.paste(part_imgs["hand"][0], self.get_coords(part_json['hand'])[0])  # 원하는 위치에 붙임 
            new_image.paste(part_imgs["hand"][1], self.get_coords(part_json['hand'])[1])  # 원하는 위치에 붙임 
        if foot_active and (part_json["foot"] is not None):
            new_image.paste(part_imgs["foot"][0], self.get_coords(part_json['foot'])[0])  # 원하는 위치에 붙임 
            new_image.paste(part_imgs["foot"][1], self.get_coords(part_json['foot'])[1])  # 원하는 위치에 붙임 
        # 다른 파트들에 대해서도 같은 방식으로 처리
        return new_image
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print(img_path)
        label = 0 if (img_path.split('/')[-1].split('.')[0].split('-')[0])=='A' else 1
        image = Image.open(img_path)
        part_name = self.part#["head", "eye", "nose", "ear", "mouth", "hand", "foot", "upper_body", "lower_body"]
        # if self.binary_thresholding:
        #     image = image.convert("L")#! Convert grayscale
        #     image = image.point(lambda p: p > self.binary_thresholding and 255)
        part_json = self.get_part_json(os.path.join(self.json_folder,self.json_paths[idx]),part_name=part_name)
        part_imgs = {}
        for part in part_name:#모든 part를 다시 dict으로 리턴하기위함.
            part_imgs[part]=[]
            # print(part)
            coords = self.get_coords(part_json[part])
            # print(coords)
            if coords is None:
                part_imgs[part].append(None)    
                
            elif len(coords) ==1:
                part_imgs[part].append(image.crop(coords[0]))    
            elif len(coords) == 2:
                part_imgs[part].append(image.crop(coords[0]))    
                part_imgs[part].append(image.crop(coords[1]))    
        empty_face = self.get_empty_face(image,part_imgs,part_json)
        # empty_face.show()
        empty_upper_body = self.get_empty_upper_body(image,part_imgs,part_json)
        empty_lower_body = self.get_empty_lower_body(image,part_imgs,part_json)
        part_imgs['empty_face']=[empty_face]
        part_imgs['empty_lower_body']=[empty_lower_body]
        part_imgs['empty_upper_body']=[empty_upper_body]
        part_combinations = list(itertools.product([0, 1], repeat=7))
        new_imgs = []
        for combination in part_combinations:
            # print(combination)
            new_img=self.create_new_images(img=image,binary_combination=combination, part_imgs=part_imgs,part_json=part_json)
            if self.transform:
                new_img=self.transform(new_img)
            new_imgs.append(new_img.unsqueeze(0))
        new_imgs = torch.cat(new_imgs,dim=0)
        image = self.transform(image)
        image_3ch = image.expand(3,-1,-1)
        return new_imgs,image_3ch,label 
    
    




if __name__=="__main__":
    transform= transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),ThresholdTransform(240)])
    part_name = ["head", "eye", "nose", "ear", "mouth", "hand", "foot", "upper_body", "lower_body"]
    dataset = shapley_part('/data/jong980812/project/mae/util/shapley/TD','/data/jong980812/project/mae/util/shapley/TD',part_name,240,transform=transform)
    data_loader=DataLoader(dataset,5,num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ordered_pair,weights = get_ordered_pair()
    part_number = all_ordered_pair.shape[0]
    part_count = {i: 0 for i in range(part_number)}
    num_correct = 0
    for new_imgs,original_image,label in data_loader:
        # print(new_imgs.shape)
        input_data = new_imgs
        # print('complete')
        batch_size = input_data.shape[0]
        input_data = rearrange(input_data,  'b t c h w -> (b t) c h w')
        
        
        model.to(device)
        input_data = input_data.to(device)
        original_image = original_image.to(device)
        label = label.to(device)
        model.eval()
        with torch.no_grad():
            prediction = model(original_image)
            output=model(input_data)
        output = rearrange(output, '(b t) o -> b t o', b=batch_size) # batch_size, 128, output(2)
        prediction = prediction.argmax(1)
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
        
