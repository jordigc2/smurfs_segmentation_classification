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
    def __init__(self, data_path, json_file_name, img_ids, preprocess_fn=None, seg_only=True) -> None:
        """
        Constructor for the SmurfsDataset class.

        Args:
            data_path (str): Path to the dataset.
            json_file_name (str): Name of the JSON file containing dataset annotations.
            img_ids (list): List of image IDs to include in the dataset.
            preprocess_fn (callable): Preprocessing function for images (optional).
            seg_only (bool): Flag indicating if the model was for segmentation only or not.
        """

        with open(data_path + json_file_name, "r") as json_file:
            self.smurfs_data = json.load(json_file)

        self.data_path = data_path
        self.seg_only = seg_only
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
                self.ann_dict[ann['id']]=ann
        for img in self.smurfs_data['images']:
            if img['id'] in img_ids:
                self.imgs_dict[img['id']] = img
        for cat in self.smurfs_data['categories']:
            self.categ_dict[cat['id']] = cat

    def get_imgs_ids(self):
        """
        Get the list of image IDs in the dataset.

        Returns:
            list: List of image IDs.
        """
        return list(self.imgs_dict.keys())

    def get_img_path_by_id(self, img_id):
        """
        Get the file path of an image based on its ID.

        Args:
            img_id (int): Image ID.

        Returns:
            str: File path of the image.
        """
        return self.data_path + self.imgs_dict[img_id]['file_name']

    def get_cat_by_id(self, cat_ids):
        """
        Get category information based on category IDs.

        Args:
            cat_ids (list): List of category IDs.

        Returns:
            list: List of category information dictionaries.
        """
        return [self.categ_dict[cat_id] for cat_id in cat_ids]

    def get_ann_by_imgs_id(self, img_ids):
        """
        Get annotation IDs based on a list of image IDs.

        Args:
            img_ids (list): List of image IDs.

        Returns:
            list: List of annotation IDs.
        """
        return [annotId for img_id in img_ids for annotId in self.img_annotId_dict[img_id]]

    def get_ann_by_id(self, ann_ids):
        """
        Get annotation information based on annotation IDs.

        Args:
            ann_ids (list): List of annotation IDs.

        Returns:
            list: List of annotation information dictionaries.
        """
        return [self.ann_dict[ann_id] for ann_id in ann_ids]

    def set_annot_cat_by_id(self, cat_id, annot_id):
        """
        Set the category ID for a specific annotation ID.

        Args:
            cat_id (int): Category ID.
            annot_id (int): Annotation ID.
        """
        self.ann_dict[annot_id]['category_id'] = cat_id

    def set_preprocessing_fn(self, preprocess_fn):
        """
        Set the preprocessing function for images.

        Args:
            preprocess_fn (callable): Preprocessing function for images.
        """
        self.preprocess_fn = preprocess_fn

    def __getitem__(self, index):
        """
        Get the item (image and annotations) at the specified index and applies several transforms
        to the images, segmentatoin masks and bounding boxes for data augmentation.

        Args:
            index (int): Index of the item.

        Returns:
            dict: A dictionary containing the transformed image and annotations.
        """

        imgs_ids = self.get_imgs_ids()

        img_id = imgs_ids[index]
        img_h = self.imgs_dict[img_id]['height']
        img_w = self.imgs_dict[img_id]['width']
        ann_ids = self.get_ann_by_imgs_id([img_id])

        annotations = self.coco.loadAnns(ann_ids)

        multi_class_binary_mask = np.zeros((img_h, img_w, len(self.categ_dict)))

        boxes = []
        labels = []
        areas = []
        is_crowds = []
        for ann in annotations:
            mask_ann = self.coco.annToMask(ann)
            multi_class_binary_mask[:,:, ann['category_id']] = \
                np.logical_or(multi_class_binary_mask[:,:, ann['category_id']],
                              mask_ann)

            xmin = ann['bbox'][0]
            ymin = ann['bbox'][1]
            xmax = ann['bbox'][2]
            ymax = ann['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(ann['category_id']+1)

            areas.append(ann['area'])

            is_crowds.append(ann['iscrowd'])

        # Annotation is in dictionary format
        annotation = {}
        annotation["boxes"] = boxes
        annotation["labels"] = labels
        # annotation["image_id"] = img_id
        # annotation["area"] = areas
        # annotation["iscrowd"] = is_crowds
        annotation["segmentation"] = multi_class_binary_mask


        img = Image.open(self.get_img_path_by_id(img_id=img_id))

        sample = self.transform({"image": img, "target": annotation})

        return sample

    def __len__(self):
        return len(self.imgs_dict)

    def transform(self, sample):
        """
        Apply transformations to the input image and annotations.

        Args:
            sample (dict): A dictionary containing the image and annotations.

        Returns:
            dict: A dictionary containing the transformed image and annotations.
        """
        ann = sample['target']
        img, mask = sample['image'], ann['segmentation']

        # Suppress the specific warning during image conversion
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img = img.resize((736,736)).convert('RGB')

        bboxes = self.resize_bboxes(sample)

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
                bboxes = self.hflip_bboxes(bboxes)

            if torch.rand(1) > 0.5 and self.seg_only:
                range_rot = random.randint(0,10)

                img_aug = transforms.functional.rotate(img_aug, range_rot)
                mask_aug = transforms.functional.rotate(mask_aug,range_rot)

            if torch.rand(1) > 0.5:
                transform = \
                        transforms.ColorJitter(brightness=(0.5,1.5),
                       contrast=(1),saturation=(0.5,1.5))
                img_aug = transform(img_aug)

        roi_masks, roi_imgs = self.extract_sub_segment(bboxes, mask_aug.numpy(), img_aug)

        if self.preprocess_fn is not None:
            # img_pre = self.preprocess_fn(np.array(img_aug))
            img_pre = img_aug.copy()
        else:
            img_pre = img_aug.copy()

        trans_img = transforms.ToTensor()(img_pre)
        img_tensor = transforms.ToTensor()(img_aug)

        ann['roi_masks'] = roi_masks
        ann['roi_imgs'] = roi_imgs
        ann['boxes'] = torch.from_numpy(bboxes)
        ann['labels'] = torch.Tensor(ann['labels'])
        ann['segmentation'] = mask_aug
        if self.seg_only:
            ann = {'segmentation': mask_aug}
            return {"image": trans_img, "target": ann, "image_pil": img_tensor}

        return {"image": trans_img, "target": ann, "image_pil": img_tensor}

    def resize_bboxes(self, sample, new_img_size=(736, 736)):
        """
        Resize bounding boxes based on the new image size.

        Args:
            sample (dict): A dictionary containing the image and target annotations.
            new_img_size (tuple): New image size in the format (width, height).

        Returns:
            numpy.ndarray: Resized bounding boxes.
        """
        img_w, img_h = sample['image'].size[0], sample['image'].size[1]
        bboxes = np.array(sample['target']['boxes'])

        bboxes[:, 0] = bboxes[:, 0] * (new_img_size[0] / img_w)
        bboxes[:, 1] = bboxes[:, 1] * (new_img_size[1] / img_h)
        bboxes[:, 2] = bboxes[:, 2] * (new_img_size[0] / img_w)
        bboxes[:, 3] = bboxes[:, 3] * (new_img_size[1] / img_h)

        return bboxes

    def hflip_bboxes(self, bboxes, img_size=(736,736)):
        """
        Horizontally flip bounding boxes.

        Args:
            bboxes (numpy.ndarray): Bounding boxes in the format [x_min, y_min, width, height].
            img_size (tuple): Image size in the format (width, height).

        Returns:
            numpy.ndarray: Horizontally flipped bounding boxes.
        """
        bboxes[:, 0] = img_size[0] - bboxes[:, 0] - bboxes[:, 2]
        return bboxes


    def update_json_cat_by_ann_id(self, cat_id, annot_id):
        """
        Update the category ID in the JSON file based on annotation ID.

        Args:
            cat_id (int): New category ID.
            annot_id (int): Annotation ID.
        """
        self.smurfs_data['annotations'][annot_id]['category_id'] = cat_id
        save_file = open(self.data_path + "updated_result.json", "w")
        json.dump(self.smurfs_data, save_file)
        save_file.close()


    def visualize_annot_by_imgid(self, index):
        """
        Visualize annotations for a specific image.

        Args:
            index (int): Index of the image in the dataset.
        """

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

    def extract_sub_segment(self, bboxes, mask, img):
        """
        Extract sub-segments based on bounding boxes from the mask and image.

        Args:
            bboxes (numpy.ndarray): Bounding boxes in the format [x_min, y_min, width, height].
            mask (numpy.ndarray): Binary mask.
            img (PIL.Image.Image or numpy.ndarray): Original image.

        Returns:
            tuple: Tuple containing sub-segment masks (torch.Tensor) and sub-segment images (torch.Tensor).
        """
        sub_seg_mask = []
        sub_seg_img = []
        img = np.asarray(img)

        for bbox in bboxes:
            x0,y0 = int(bbox[1]), int(bbox[0])
            x1,y1 = int(bbox[3]+x0), int(bbox[2]+y0)

            roi_mask = mask[:,x0:x1, y0:y1]
            roi_img = img[x0:x1, y0:y1,:]

            mask_bin = np.zeros((160,160), dtype=np.uint8)
            for chn in range(roi_mask.shape[0]):
                mask_chn = roi_mask[chn,:,:]
                mask_pil = Image.fromarray(mask_chn).resize((160,160))
                mask_resized = np.array(mask_pil, dtype=np.uint8)
                mask_resized[mask_resized>0] = int(1)
                mask_resized[mask_resized<=0] = int(0)
                mask_bin = np.bitwise_or(mask_bin, mask_resized)

            roi_mask = mask_bin
            roi_img = Image.fromarray(roi_img).resize((160,160))

            sub_seg_mask.append(roi_mask)
            sub_seg_img.append(transforms.ToTensor()(roi_img))

        sub_masks_tensor = torch.from_numpy(np.array(sub_seg_mask))
        sub_imgs_tensor = torch.stack(sub_seg_img, dim=0)

        return sub_masks_tensor, sub_imgs_tensor

    def generate_multi_class_mask(self, mask, is_tensor=False):
        """
        Generate a multi-class mask by combining binary masks for each class.

        Args:
            mask (numpy.ndarray or torch.Tensor): Binary masks for each class.
            is_tensor (bool): Whether the input mask is a torch.Tensor.

        Returns:
            numpy.ndarray or torch.Tensor: Multi-class mask.
        """
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