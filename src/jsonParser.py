import json
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class JSONParserCOCO:
    def __init__(self, data_path, json_file_name) -> None:
    
        with open(data_path + json_file_name, "r") as json_file:
            self.smurfs_data = json.load(json_file)
        
        self.data_path = data_path

        self.img_annotId_dict = defaultdict(list)        
        self.categ_dict = {} 
        self.ann_dict = {}
        self.imgs_dict = {}

        self.color_list = ["red", "blue", "orange","green"]

        for ann in self.smurfs_data['annotations']: 
            self.img_annotId_dict[ann['image_id']].append(ann['id']) 
            #print("ann:", type(ann["id"]))
            self.ann_dict[ann['id']]=ann
        for img in self.smurfs_data['images']:
            self.imgs_dict[img['id']] = img
        for cat in self.smurfs_data['categories']:
            self.categ_dict[cat['id']] = cat

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

    def update_json_cat_by_ann_id(self, cat_id, annot_id):
        self.smurfs_data['annotations'][annot_id]['category_id'] = cat_id
        save_file = open(self.data_path + "updated_result.json", "w")  
        json.dump(self.smurfs_data, save_file)
        save_file.close()
    

    def visualize_annot_by_imgid(self, img_id):
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
 
        


# if __name__== "__main__":
#     data_path = "/data/smurfs_coco_format/"
#     json_file_name = "result.json"

#     dataset = JSONParserCOCO(data_path, json_file_name)
#     dataset.visualize_annot_by_imgid(1)