from torch.utils.data import Dataset
import os
import PIL
from PIL import Image
import numpy as np
import json
import itertools
import torch
import random 
random.seed(777)

def get_dataset_information(args):
    information = dict()
    information['task'] = args.data_path.split('/')[3].split('_')[-1] 
    information['class_names']= os.listdir(args.data_path)
    return information


class Shapley_part(Dataset):
    def __init__(self, data_folder, json_folder, task, transform=None):
        self.json_folder = json_folder
        self.data_folder = data_folder
        self.task = task
        self.transform = transform
        self.image_paths = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        self.json_paths = [image_path.split('/')[-1].split('.')[0] + ".json" for image_path in self.image_paths] #! Get json path from image paths.
        # print(self.image_paths)
    def __repr__(self) -> str:
        target_foler= self.data_folder
        target_task = self.task
        number = len(self.image_paths)
        return f'Shapley Part Dataset class\nTarget Folder:{target_foler}\nTarget Task:{target_task}\nData Num:{number}'
    def get_part_json(self, json_file_path):
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
            for a in part: 
                # print(a)
                xmin, ymin = list(map(int,a[0]))
                xmax, ymax = list(map(int,a[1]))
                extracted_coordinates.append([xmin,ymin,xmax,ymax])
            return extracted_coordinates
    def get_white_image(self,size):
        return Image.new("RGB", size, (255, 255, 255))
    def get_only_hair(self,img,part_imgs,part_json):
        head_coords = self.get_coords(part_json['head'])
        head = part_imgs['head'][0]#!
        hair_coords = self.get_coords(part_json['hair'])
        hair = part_imgs['hair'][0]#!
        face_coords = self.get_coords(part_json['face'])
        face = part_imgs['face'][0]
        neck_coords = self.get_coords(part_json['neck'])
        neck = part_imgs['neck'][0]
        white_image = self.get_white_image(img.size)
        white_image.paste(hair,hair_coords[0])
        white_image.paste(self.get_white_image(face.size),face_coords[0])
        white_image.paste(self.get_white_image(neck.size),neck_coords[0])
        return white_image.crop(hair_coords[0]), [[hair_coords[0][0],hair_coords[0][1]],[hair_coords[0][2],hair_coords[0][3]]]
    def get_only_face(self,img,part_imgs,part_json):
        face_coords = self.get_coords(part_json['face'])
        face = part_imgs['face'][0]

        white_image = self.get_white_image(img.size)
        white_image.paste(face,face_coords[0])

        for part in ['eye','nose','mouth','ear']:
                    if part_json[part] is not None:
                        part_coords= self.get_coords(part_json[part])
                        part_img = part_imgs[part]
                        for i in range(len(part_img)):
                            white_image.paste(self.get_white_image(part_img[i].size),part_coords[i])
        return white_image.crop(face_coords[0]), [[face_coords[0][0],face_coords[0][1]],[face_coords[0][2],face_coords[0][3]]]
    def get_empty_face(self,img, part_imgs, part_json):
        '''
        empty_face is face detached  'eye','nose','mouth','ear' from head+hair
        '''
        head_coords = self.get_coords(part_json['head'])
        head = part_imgs['head'][0]#!
        
        hair_coords = self.get_coords(part_json['hair'])
        hair = part_imgs['hair'][0]#!
        white_image = self.get_white_image(img.size)
        white_image.paste(head,head_coords[0])
        white_image.paste(hair,hair_coords[0])
        for part in ['eye','nose','mouth','ear']:
            if part_json[part] is not None:
                part_coords= self.get_coords(part_json[part])
                part_img = part_imgs[part]
                for i in range(len(part_img)):
                    white_image.paste(self.get_white_image(part_img[i].size),part_coords[i])
                  
            #   if part in ['eye','ear']:   
            #       white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
            #       white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
            #   else:
            #       white_image.paste(self.get_white_image(part_img[0].size),part_coords[0])
        # white_image.show()
        xmins = [head_coords[0][0],hair_coords[0][0]]
        ymins = [head_coords[0][1],hair_coords[0][1]]
        xmaxs = [head_coords[0][2],hair_coords[0][2]]
        ymaxs = [head_coords[0][3],hair_coords[0][3]]
        empty_face_coords = [min(xmins),min(ymins), max(xmaxs), max(ymaxs)]
            
        return white_image.crop(empty_face_coords), [[min(xmins),min(ymins)],[max(xmaxs),max(ymaxs)]]
    def get_empty_upper_body(self,img, part_imgs, part_json):
        '''
        empty_lower_body detacched foot
        upper body를 목 포함한 걸로 규정하고, 
        1. 백지에 몸통 붙이기
        2. 양 팔 붙이고 손떼기
        3. 목 붙이기
        4. 새로운 upperbody 좌표 규정.
        '''
        white_image = Image.new("RGB", img.size, (255, 255, 255))
        upper_body_else_arm_json = part_json['upper_body_else_arm']
        upper_body_else_arm_coords = self.get_coords(upper_body_else_arm_json)
        upper_body_else_arm = part_imgs['upper_body_else_arm'][0]#!
        if part_json["neck"] is not None:
            neck_json = part_json['neck']
            neck_coords = self.get_coords(neck_json)
            neck = part_imgs['neck'][0]#!
            white_image.paste(neck,neck_coords[0])
        neck_json = part_json['neck']
        neck_coords = self.get_coords(neck_json)
        neck = part_imgs['neck'][0]#!
        arm_json = part_json['arm']
        arm_coords = self.get_coords(arm_json)
        arm_imgs= part_imgs['arm']#!
        for i in range(len(arm_imgs)):
            white_image.paste(arm_imgs[i],arm_coords[i])
        
        white_image.paste(upper_body_else_arm,upper_body_else_arm_coords[0])


        # white_image.paste(left_hand,hand_coords[0])
        # white_image.paste(right_hand,hand_coords[1])
        if part_json["hand"] is not None:
            part_coords= self.get_coords(part_json["hand"])
            part_img = part_imgs["hand"]
            for i in range(len(part_img)):
                white_image.paste(Image.new("RGB", part_img[i].size, (255, 255, 255)),part_coords[i])
                # white_image.paste(Image.new("RGB", part_img[1].size, (255, 255, 255)),part_coords[1])
        xmins = [arm_coords[i][0] for i in range(len(arm_imgs))]
        ymins = [arm_coords[i][1] for i in range(len(arm_imgs))]
        xmaxs = [arm_coords[i][2] for i in range(len(arm_imgs))]
        ymaxs = [upper_body_else_arm_coords[0][3]]#[arm_coords[i][3] for i in range(len(arm_imgs))]
        if part_json['neck'] is not None:
            ymins.append(neck_coords[0][1])
        upper_body_coords = [min(xmins),min(ymins),max(xmaxs),max(ymaxs)]
        
        return white_image.crop(upper_body_coords), [[upper_body_coords[0],upper_body_coords[1]],[upper_body_coords[2],upper_body_coords[3]]]

    def get_empty_lower_body(self,img, part_imgs, part_json):
        '''
        empty_lower_body detacched foot
        leg 두개를 빈 도화지에 붙이고 발을 뗀뒤 empty lower body로 규정.
        '''
        white_image = self.get_white_image(img.size)
        
        leg_json = part_json['leg']
        leg_coords = self.get_coords(leg_json)
        leg_img= part_imgs['leg'] 
        
        pocket_json =  part_json['pocket']
        pocket_coords = self.get_coords(pocket_json)
        if len(leg_coords)!=2:
            print('a')
        for i in range(len(leg_coords)):
            white_image.paste(leg_img[i],leg_coords[i])
            # white_image.paste(leg_img[i],leg_coords[i])
        # for i,pocket in enumerate(part_imgs['pocket']):
        #     # pocket.show()
        #     white_image.paste(pocket,pocket_coords[i])
        # white_image.show()
        if part_json["foot"] is not None:
            part_coords= self.get_coords(part_json["foot"])
            part_img = part_imgs["foot"] 
            for i in range(len(part_img)):
                white_image.paste(Image.new("RGB", part_img[i].size, (255, 255, 255)),part_coords[i])
            # white_image.paste(self.get_white_image(part_img[1].size),part_coords[1])
        # white_image.show()
        # white_image.crop(upper_body_coords[0]).show()
        # upper_body_ymax= self.get_coords(part_json["upper_body_else_arm"])[0][3] 
        # pocket_ymin = min([pocket_coords[i][1] for i in range(len(part_imgs['pocket']))])
        # leg_ymin = min(leg_coords[0][1],leg_coords[1][1])
        xmins = [leg_coords[i][0] for i in range(len(leg_coords))]
        # ymins = [pocket_ymin] if (upper_body_ymax<=pocket_ymin) and (leg_ymin>=pocket_ymin) else [leg_coords[0][1],leg_coords[1][1]]
        ymins = [leg_coords[i][1] for i in range(len(leg_coords))]
        xmaxs = [leg_coords[i][2] for i in range(len(leg_coords))]
        ymaxs = [leg_coords[i][3] for i in range(len(leg_coords))]

        lower_body_coords = [min(xmins),min(ymins),max(xmaxs),max(ymaxs)]
        # white_image.crop(lower_body_coords).show()
        return white_image.crop(lower_body_coords),[[min(xmins),min(ymins)],[max(xmaxs),max(ymaxs)]]
    
    def create_new_images(self,img, binary_combination, part_imgs,part_json,sex):
        #! Making New images
        original_img = img
        hair_active, eye_active, nose_active, ear_active, mouth_active, hand_active, foot_active = binary_combination
        new_image = img.copy()
        #! Original image에서 Lower body, Upperbody빼고 모두 없앰.
        for i in range(len(part_imgs['empty_face'])):
            new_image.paste(self.get_white_image(part_imgs['empty_face'][i].size),self.get_coords(part_json['empty_face'])[i])
        for i in range(len(part_imgs['eye'])):
            new_image.paste(self.get_white_image(part_imgs['eye'][i].size),self.get_coords(part_json['eye'])[i])
        for i in range(len(part_imgs['nose'])):
            new_image.paste(self.get_white_image(part_imgs['nose'][i].size),self.get_coords(part_json['nose'])[i])
        for i in range(len(part_imgs['ear'])):
            new_image.paste(self.get_white_image(part_imgs['ear'][i].size),self.get_coords(part_json['ear'])[i])
        for i in range(len(part_imgs['mouth'])):
            new_image.paste(self.get_white_image(part_imgs['mouth'][i].size),self.get_coords(part_json['mouth'])[i])
        for i in range(len(part_imgs['hand'])):
            new_image.paste(self.get_white_image(part_imgs['hand'][i].size),self.get_coords(part_json['hand'])[i])
        for i in range(len(part_imgs['foot'])):
            new_image.paste(self.get_white_image(part_imgs['foot'][i].size),self.get_coords(part_json['foot'])[i]) 
        for i in range(len(part_imgs['hand'])):
            new_image.paste(self.get_white_image(part_imgs['hand'][i].size),self.get_coords(part_json['hand'])[i]) 
        for i in range(len(part_json['sneakers'])):
            new_image.paste(self.get_white_image(part_imgs['sneakers'][i].size),self.get_coords(part_json['sneakers'])[i])
        if part_json['sneakers'] is not None:
            # print(part_json['sneakers'])
            for i in range(len(part_json['sneakers'])):
                new_image.paste(self.get_white_image(part_imgs['sneakers'][i].size),self.get_coords(part_json['sneakers'])[i])
        if sex == 0 and (part_json['man_shoes'] is not None):
            for i in range(len(part_json['man_shoes'])):
                new_image.paste(self.get_white_image(part_imgs['man_shoes'][i].size),self.get_coords(part_json['man_shoes'])[i])
        elif sex == 1 and (part_json['woman_shoes'] is not None):
            for i in range(len(part_json['woman_shoes'])):
                new_image.paste(self.get_white_image(part_imgs['woman_shoes'][i].size),self.get_coords(part_json['woman_shoes'])[i])
        new_image.paste(part_imgs["empty_upper_body"][0], self.get_coords(part_json['empty_upper_body'])[0])  # 원하는 위치에 붙임
        new_image.paste(part_imgs["only_face"][0], self.get_coords(part_json['only_face'])[0])  # 원하는 위치에 붙임

          
        #!######
        
        if hair_active:
            # new_image.paste(part_imgs["empty_face"][0],self.get_coords(part_json['empty_face'])[0])
            new_image.paste(part_imgs["only_hair"][0],self.get_coords(part_json['only_hair'])[0])
            new_image.paste(part_imgs["only_face"][0], self.get_coords(part_json['only_face'])[0])  # 원하는 위치에 붙임
            new_image.paste(part_imgs["neck"][0], self.get_coords(part_json['neck'])[0])  # 원하는 위치에 붙임
            
        # 각 파트 이미지를 읽어와서 새로운 이미지에 붙임
        if eye_active and (part_json["eye"] is not None):
            for i in range(len(part_imgs["eye"])):
                new_image.paste(part_imgs["eye"][i], self.get_coords(part_json['eye'])[i])  # 원하는 위치에 붙임
        if nose_active and (part_json["nose"] is not None):
            for i in range(len(part_imgs["nose"])):
                new_image.paste(part_imgs["nose"][i], self.get_coords(part_json['nose'])[i])  # 원하는 위치에 붙임
        if ear_active and (part_json["ear"] is not None):
            for i in range(len(part_imgs["ear"])):
                new_image.paste(part_imgs["ear"][i], self.get_coords(part_json['ear'])[i])  # 원하는 위치에 붙임
        if mouth_active and (part_json["mouth"] is not None):
            for i in range(len(part_imgs["mouth"])):
                new_image.paste(part_imgs["mouth"][i], self.get_coords(part_json['mouth'])[i])  # 원하는 위치에 붙임
        if hand_active and (part_json["hand"] is not None):
            for i in range(len(part_imgs["hand"])):
                new_image.paste(part_imgs["hand"][i], self.get_coords(part_json['hand'])[i])  # 원하는 위치에 붙임
        if foot_active and (part_json["foot"] is not None):
            for i in range(len(part_imgs["foot"])):
                new_image.paste(part_imgs["foot"][i], self.get_coords(part_json['foot'])[i])  # 원하는 위치에 붙임
        # 다른 파트들에 대해서도 같은 방식으로 처리
        return new_image.crop( self.get_coords(part_json['human_body'])[0])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # print(img_path)
        sex = 0 if (img_path.split('/')[-1].split('.')[0].split('_')[0])=='m' else 1
        if self.task == 'mw':
            label = 0 if (img_path.split('/')[-2])=='m' else 1
        elif self.task == 'drawer':
            label = 0 if (img_path.split('/')[-2])=='m' else 1
        elif self.task == '4way':
            labe_list ={'m_m': 0, 'm_w': 1, 'w_m': 2, 'w_w': 3}
            label = labe_list[(img_path.split('/')[-2])]
        elif self.task == 'age':
            label = int(img_path.split('/')[-2])
            
        image = Image.open(img_path)
        part_name = ["human_body","face","head","hair", "neck","eye", "nose", "ear", "mouth","pocket","arm","hand", "leg","foot", "sneakers","upper_body_else_arm"]#원하는 파트
        part_name.append('man_shoes') if sex==0 else part_name.append('woman_shoes')
        
        # if self.binary_thresholding:
            # image = image.convert("L")#! Convert grayscale
            # image = image.point(lambda p: p > self.binary_thresholding and 255)
        part_json = self.get_part_json(os.path.join(self.json_folder,self.json_paths[idx]))#! 존재하는 모든 part에 대해서 불러옴.
        part_imgs = {}
        for part in part_name:#모든 part를 잘라서 다시 dict으로 리턴하기위함.
            part_imgs[part]=[]
            coords = self.get_coords(part_json[part])
            if coords is None:
                part_imgs[part].append(None)    
            # elif len(coords) ==1:
            #     part_imgs[part].append(image.crop(coords[0]))    
            # elif len(coords) == 2:
            #     part_imgs[part].append(image.crop(coords[0]))    
            #     part_imgs[part].append(image.crop(coords[1]))
            else:
                for i in range(len(coords)):
                    part_imgs[part].append(image.crop(coords[i]))    
        empty_face , empty_face_coords= self.get_empty_face(image,part_imgs,part_json)
        # empty_face.show()
        empty_upper_body, empty_upper_body_coords = self.get_empty_upper_body(image,part_imgs,part_json)
        empty_lower_body, empty_lower_body_coords= self.get_empty_lower_body(image,part_imgs,part_json)
        only_hair, only_hair_coords = self.get_only_hair(image,part_imgs,part_json)
        only_face, only_face_coords = self.get_only_face(image,part_imgs,part_json)
        part_imgs['empty_face']=[empty_face]
        part_json['empty_face']=[empty_face_coords]
        part_imgs['empty_lower_body']=[empty_lower_body]
        part_json['empty_lower_body']=[empty_lower_body_coords]
        part_imgs['empty_upper_body']=[empty_upper_body]
        part_json['empty_upper_body']=[empty_upper_body_coords]#좌표 바뀌어서 넣어줘야함.
        part_imgs['only_hair']=[only_hair]
        part_json['only_hair']=[only_hair_coords]        
        part_imgs['only_face']=[only_face]
        part_json['only_face']=[only_face_coords]
        original_image=image.crop( self.get_coords(part_json['human_body'])[0])
        
        part_combinations = list(itertools.product([0, 1], repeat=7))
        new_imgs = []
        # print(part_json)
        for combination in part_combinations:
            # print(combination)
            new_img=self.create_new_images(img=image,binary_combination=combination, part_imgs=part_imgs,part_json=part_json,sex=sex)
            if self.transform:
                new_img=self.transform(new_img)#.expand(3,-1,-1)
            new_imgs.append(new_img.unsqueeze(0))
        new_imgs = torch.cat(new_imgs,dim=0)
        # image = self.transform(image)
        # image_3ch = image.expand(3,-1,-1)
        return new_imgs, self.transform(original_image), label ,img_path