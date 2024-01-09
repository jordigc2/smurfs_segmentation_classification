import warnings

import json
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from pycocotools.coco import COCO
from torchvision import transforms

import random

class SmurfsDataset(data.Dataset):
    def __init__(self, data_path, json_file_name, img_ids, preprocess_fn=None) -> None:
    
        with open(data_path + json_file_name, "r") as json_file:
            self.smurfs_data = json.load(json_file)
        
        self.data_path = data_path
        self.coco = COCO(data_path + json_file_name)

        self.preprocess_fn = preprocess_fn

        self.img_annotId_dict = defaultdict(list)        
        self.categ_dict = {} 
        self.ann_dict = {}
        self.imgs_dict = {}
        

        self.color_list = ["yellow", "blue", "red","green"]
        self.color_code = [[255,255,0], [0,0,255], [255,0,0], [0,255,0]]

        for ann in self.smurfs_data['annotations']: 
            if ann['image_id'] in img_ids:
                self.img_annotId_dict[ann['image_id']].append(ann['id']) 
                #print("ann:", type(ann["id"]))
                self.ann_dict[ann['id']]=ann
        for img in self.smurfs_data['images']:
            if img['id'] in img_ids:
                self.imgs_dict[img['id']] = img
        for cat in self.smurfs_data['categories']:
            self.categ_dict[cat['id']] = cat

    def __getitem__(self, index):
        
        imgs_ids = self.get_imgs_ids()

        img_id = imgs_ids[index]
        img_h = self.imgs_dict[img_id]['height']
        img_w = self.imgs_dict[img_id]['width']
        ann_ids = self.get_ann_by_imgs_id([img_id])

        annotations = self.coco.loadAnns(ann_ids)
        
        multi_class_binary_mask = np.zeros((img_h, img_w, len(self.categ_dict)))

        for ann in annotations:
            mask_ann = self.coco.annToMask(ann)
            multi_class_binary_mask[:,:, ann['category_id']] = \
                np.logical_or(multi_class_binary_mask[:,:, ann['category_id']],
                              mask_ann)

        img = Image.open(self.get_img_path_by_id(img_id=img_id))

        sample = {"image": img, "target": multi_class_binary_mask}

        sample = self.transform(sample)

        return sample  
    
    def __len__(self):
        return len(self.imgs_dict)

    def transform(self, sample):
        img, mask = sample['image'], sample['target']  

        # Suppress the specific warning during image conversion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = img.resize((736,736)).convert('RGB')

        mask_chn_list = []
        for chn in range(mask.shape[2]):
            mask_chn = mask[:,:,chn]
            mask_pil = Image.fromarray(mask_chn).resize((736,736))
            mask_resized = np.array(mask_pil)
            mask_resized[mask_resized>0] = int(1)
            mask_resized[mask_resized<=0] = int(0)
            mask_chn_list.append(mask_resized)

        mask_aug = torch.from_numpy(np.array(mask_chn_list))

        img_aug = img.copy()

        if img.size[0] > 1:
            #Applying data aumentation to generalize:
            # (hflip, rot, colorjitter)
            if torch.rand(1) > 0.5:
                img_aug = transforms.functional.hflip(img_aug)
                mask_aug = torch.flip(mask_aug, dims=[2])

            if torch.rand(1) > 0.5:
                range_rot = random.randint(0,10)                

                img_aug = transforms.functional.rotate(img_aug, range_rot)
                mask_aug = transforms.functional.rotate(mask_aug,range_rot)

            if torch.rand(1) > 0.5:
                transform = \
                        transforms.ColorJitter(brightness=(0.5,1.5),
                       contrast=(1),saturation=(0.5,1.5))
                img_aug = transform(img_aug)
                                
        if self.preprocess_fn is not None:
            img_pre = self.preprocess_fn(np.array(img_aug))
            # img_pre = img_aug.copy()
        else:
            img_pre = img_aug.copy()

        trans_img = transforms.ToTensor()(img_pre)
        img_tensor = transforms.ToTensor()(img_aug)

        return {"image": trans_img, "target": mask_aug, "image_pil": img_tensor}

    def get_imgs_ids(self):
        return list(self.imgs_dict.keys())

    def get_img_path_by_id(self, img_id):
        return self.data_path + self.imgs_dict[img_id]['file_name']
    
    def get_cat_by_id(self, cat_ids):
        return [self.categ_dict[cat_id] for cat_id in cat_ids]
    
    def get_ann_by_imgs_id(self, img_ids):
        return [annotId for img_id in img_ids for annotId in self.img_annotId_dict[img_id]]
    
    def get_ann_by_id(self, ann_ids):
        return [self.ann_dict[ann_id] for ann_id in ann_ids]
    
    def set_annot_cat_by_id(self, cat_id, annot_id):
        self.ann_dict[annot_id]['category_id'] = cat_id

    def set_preprocessing_fn(self, preprocess_fn):
        self.preprocess_fn = preprocess_fn

    def update_json_cat_by_ann_id(self, cat_id, annot_id):
        self.smurfs_data['annotations'][annot_id]['category_id'] = cat_id
        save_file = open(self.data_path + "updated_result.json", "w")  
        json.dump(self.smurfs_data, save_file)
        save_file.close()
    

    def visualize_annot_by_imgid(self, index):

        imgs_ids = self.get_imgs_ids()

        img_id = imgs_ids[index]

        img = Image.open(self.get_img_path_by_id(img_id=img_id))
        
        plt.figure("img_id: " + str(img_id))

        for ann_id in self.get_ann_by_imgs_id([img_id]):

            annot = self.ann_dict[ann_id]
            
            bbox = annot['bbox']
            x, y, w, h = [int(b) for b in bbox]
            class_id = annot["category_id"]
            class_name = self.get_cat_by_id([class_id])[0]["name"]
            color_ = self.color_list[class_id]
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_, facecolor='none')

            t_box=plt.text(x, y, class_name,  color='red', fontsize=10)
            t_box=plt.text(x, y+100, ann_id,  color='blue', fontsize=10)
            t_box.set_bbox(dict(boxstyle='square, pad=0',facecolor='white', alpha=0.6, edgecolor='blue'))
            plt.gca().add_patch(rect)
        plt.imshow(img)         

    def generate_multi_class_mask(self, mask, is_tensor=False):
        mask_h = mask.shape[1]
        mask_w = mask.shape[2]

        if not is_tensor:
            rgb_mask = np.zeros((mask_h, mask_w,3))
            for cat in self.categ_dict:
                rgb_mask[:,:,0]+= (mask[cat,:,:]*self.color_code[cat][0]) 
                rgb_mask[:,:,1]+= (mask[cat,:,:]*self.color_code[cat][1]) 
                rgb_mask[:,:,2]+= (mask[cat,:,:]*self.color_code[cat][2]) 
                rgb_mask[rgb_mask > 255] = 255
        else:
            rgb_mask = np.zeros((3,mask_h, mask_w))
            for cat in self.categ_dict:
                rgb_mask[0,:,:]+= (mask[cat,:,:]*self.color_code[cat][0]//255) 
                rgb_mask[1,:,:]+= (mask[cat,:,:]*self.color_code[cat][1]//255) 
                rgb_mask[2,:,:]+= (mask[cat,:,:]*self.color_code[cat][2]//255)
                rgb_mask[rgb_mask > 1] = 1

            rgb_mask = torch.from_numpy(rgb_mask)

        return rgb_mask