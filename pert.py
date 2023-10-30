from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
import random
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import pickle
import random

class Perturbation:
    def __init__(self,data, model, layer, pert, mt, cls, layercheck):
        self.datapath = []
        folders = [folder for folder in os.listdir(data) if os.path.isdir(os.path.join(data, folder))]
        folders.sort()
        for i in range(len(folders)):
            self.datapath.append(os.path.join(data, folders[i])) 
        self.modelpath = model
        self.targetlayer = layer
        self.task = pert
        self.model_type = mt
        self.cls = cls
        self.layer_check = layercheck

        self.model = []
        self.conf_tot_all_images = {}
        self.IoU_tot_all_images = {}
        self.conf_initial = {}
        self.image_list = {}
        self.output_folder = "output"




    def run(self,cls):
        self.cls = cls
        self._instance()
        self.acquire_data(self.datapath[self.cls])
        self.compute()
        self.save_data()


    def _instance(self):
        if self.model_type:
            VISpath = os.path.join(self.modelpath, "VisibleModel/weights/best.pt")
            IRpath = os.path.join(self.modelpath, "IRModel/weights/best.pt")
            self.model.append(YOLO(VISpath))
            self.model.append(YOLO(IRpath))
        else:
            path = os.path.join(self.modelpath, "VisibleModel/weights/best.pt")
            self.model.append(YOLO(path))
            


    def perturbation(self, img, mask, percentile, bbox, cls, model):
        results = model(img, verbose=False, max_det=1)
        boxes = results[0].boxes
        original = results[0].plot()
        
        # conf = float(boxes.conf.cpu().numpy())

        if(self.task == "remove"):
            perturbed = img * mask
        if(self.task == "mean"):
            perturbed, _ = self.replacepixel(img, mask, percentile, self.task)
        if(self.task == "random"):
            perturbed, _ = self.replacepixel(img, mask, percentile, self.task)
        results = model([perturbed], max_det=1, classes = cls)
        if len(results[0].boxes) != 0:
            iou = self.calculate_iou(bbox,results[0].boxes.xyxy[0].cpu().numpy())
            new = results[0].plot()
            boxes = results[0].boxes
            conf_incr = float(boxes.conf.cpu().numpy())
            out = np.stack((original, new), axis=0)
            return out, conf_incr, iou
        else:
            return None, None, None

    def replacepixel(self, img, mask, percentile_value, task):
        mask = mask *255
        mask = mask.astype(np.uint8)
        image = Image.fromarray(img)
        mask = Image.fromarray(mask)
        mask_intensity = np.percentile(mask, percentile_value)

        pixel_data = image.load()
        mask_data = mask.load()
        pixel_data = image.load()
        mask_data = mask.load()
        modified_mask = mask.copy()
        modified_mask_data = modified_mask.load()
        width, height = image.size
        mask_intensity_values = [(mask_r + mask_g + mask_b) / 3 for mask_r, mask_g, mask_b in mask.getdata()]
        
        if (task == 'mean'):
            mean_r = np.mean(img[:, :, 0])
            mean_g = np.mean(img[:, :, 1])
            mean_b = np.mean(img[:, :, 2])
        
        for x in range(width):
            for y in range(height):
                mask_intensity_value = mask_intensity_values[y * width + x]

                if mask_intensity_value <= mask_intensity:
                    if(task == 'random'):
                        new_r = random.randint(0, 255)
                        new_g = random.randint(0, 255)
                        new_b = random.randint(0, 255)
                        modified_mask_data[x, y] = (new_r, new_g, new_b)
                        pixel_data[x, y] = (new_r, new_g, new_b) #random
                    
                    elif(task == 'mean'):
                        pixel_data[x, y] = (int(mean_r), int(mean_g), int(mean_b)) #mean
                    
                    elif(task == "remove"):
                        r, g, b = pixel_data[x, y]
                        mask_r, mask_g, mask_b = mask_data[x, y]
                        new_r = r * (mask_r / 255)
                        new_g = g * (mask_g / 255)
                        new_b = b * (mask_b / 255)
                        modified_mask_data[x, y] = (int(new_r), int(new_g), int(new_b))
                        pixel_data[x, y] = (int(new_r), int(new_g), int(new_b))

        modified_image = np.array(image)
        modified_mask = np.array(modified_mask)
        return modified_image, modified_mask



    def calculate_iou(self,box1, box2):

        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2

        area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

        x1_intersection = max(x1_box1, x1_box2)
        y1_intersection = max(y1_box1, y1_box2)
        x2_intersection = min(x2_box1, x2_box2)
        y2_intersection = min(y2_box1, y2_box2)

        if x1_intersection < x2_intersection and y1_intersection < y2_intersection:
            area_intersection = (x2_intersection - x1_intersection) * (y2_intersection - y1_intersection)
        else:
            area_intersection = 0.0

        iou = area_intersection / (area_box1 + area_box2 - area_intersection)

        return iou



    def test_layers(self):
        self._instance()
        path = self.get_path()
        if path is not None:
            img, rgb_img = self.get_image(path)
        plt.figure(figsize=(15, 8))
        results = self.model[0](rgb_img, verbose=False)
        u = 0
        for i in range(16,1,-1):
            target_layers = [self.model[0].model.model[-(i)]]
            cam = EigenCAM(self.model[0], target_layers)
            grayscale_cam = cam(rgb_img)[0, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            ax = plt.subplot(3, 5, u + 1)
            u += 1
            plt.imshow(cam_image)
            plt.text(0.5, -0.1, f'Layer -{i}', transform=ax.transAxes,
                    fontsize=12, ha='center', va='center', color='red')
            plt.axis("off")
        plt.suptitle("Layer check", fontsize=23, fontweight='bold', backgroundcolor='lightgray')
        filename = "layer_check"
        self.save_plot(filename) 
        plt.show()
            
    def get_path(self):
        path = self.datapath[0]

        with os.scandir(path) as entries:
            image_files = [entry.path for entry in entries if entry.is_file() and "Cam1" in entry.name]
        if image_files:
            return random.choice(image_files)
        else:
            return None

    def save_plot(self, filename):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        plt.savefig(os.path.join(self.output_folder, filename))


    def parse_detections(self, detections):
        self.COLORS = np.random.uniform(0, 255, size=(len(self.model[0].names), 3))
        boxes, colors, names = [], [], []
        for i in range(len(detections["xmin"])):
            confidence = float(detections["confidence"][i])
            if confidence < 0.2:
                continue
            xmin = int(float(detections["xmin"][i]))
            ymin = int(float(detections["ymin"][i]))
            xmax = int(float(detections["xmax"][i]))
            ymax = int(float(detections["ymax"][i]))
            name = detections["name"][i]
            category = int(detections["class"][i])
            color = self.COLORS[category]
            boxes.append((xmin, ymin, xmax, ymax))
            colors.append(color)
            names.append(name)
        return boxes, colors, names
        
    def printout(outputs):
        plt.figure(figsize=(20, 20))
        for i in range(len(outputs)):
            ax = plt.subplot(1, 4, i + 1)
            plt.imshow(outputs[i] /255, cmap="gray")
            plt.axis("off")
        plt.show()
    
    def get_image(self, image_path):
        img = np.array(Image.open(image_path))
        img = cv2.resize(img, (640, 640))
        rgb_img = img.copy()
        img = img.astype(np.float32) / 255.0
        return img, rgb_img
    

    def invert_grayscale_image(image):
        max_pixel_value = np.max(image)
        inverted_image = max_pixel_value - image
        return inverted_image


    def acquire_data(self,path):
        for filename in tqdm(os.listdir(path), desc="Acquiring images"):
            if filename.endswith(('.jpg')):
                file_path = os.path.join(path , filename)
                image = np.array(Image.open(file_path))
                self.image_list[filename]=image

    def compute(self):
        i=0
        for i in tqdm(range(len(self.image_list)), desc="Processing images"):
            conf_tot = []
            IoU_tot = []
            if "Cam1" in list(self.image_list.keys())[i] or "Cam2" in list(self.image_list.keys())[i]:
                model = self.model[0]
            else:
                model = self.model[1]
 
            layers = [model.model.model[-self.targetlayer]]
            name = list(self.image_list.keys())[i]

            image = list(self.image_list.values())[i]
            gray, inv, image, box, classes, conf, flag = self.acquire(image, model, layers)
            self.conf_initial[name] = conf

            if flag:
                if self.task != "remove":
                    for i in range (0,101,5):
                        out, conf_inc, iou = self.perturbation(image, gray, i, self.task, box, classes, model)
                        if out is not None:
                            conf_tot.append(conf_inc)
                            IoU_tot.append(iou)

                        else:
                            conf_tot.append(-1)
                            IoU_tot.append(0)
                else:
                    out, conf_inc, iou = self.perturbation(image, gray, 50, self.task, box, classes, model)
                    if out is not None:
                        conf_tot.append(conf_inc)
                        IoU_tot.append(iou)
            # heatmap[name] = (gray)
            self.conf_tot_all_images[name] = (conf_tot)
            self.IoU_tot_all_images[name] = (IoU_tot)


    def save_data(self):
        folders = []
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        name = ["initial_conf","conf_tot","iou_tot"]
        outs = [self.conf_initial, self.conf_tot_all_images, self.IoU_tot_all_images]
        
        for i in range (len(self.datapath)):
            split_parts = self.datapath[i].split('/')
            last_part = split_parts[-1]
            folders.append(last_part.split('_'))
        cls = folders[self.cls][0]
        
        for i in range(3):
            filename = cls + "_" + name[i] +".pkl"
            filepath = os.path.join(self.output_folder, filename)
            with open(filepath, 'wb') as file:
                pickle.dump(outs[i], file)



    def acquire(self, img, model, layers):
        img = cv2.resize(img, (640, 640))
        rgb_img = img.copy()
        img = img.astype(np.float32) / 255.0
        results = model(rgb_img, verbose=False, max_det= 1)
        if  len(results[0].boxes) == 0:
            return None, None, None, None, None, None, False
        else:
            boxes = results[0].boxes
            columns = ['xmin', 'ymin',	'xmax', 'ymax',	'confidence', 'class', 'name']
            bbox = boxes.xyxy[0].cpu().numpy()
            conf = float(boxes.conf.cpu().numpy())
            cls = int(boxes.cls[0])
            name = (results[0].names[int(boxes.cls[0])])
            data = np.array([bbox[0],bbox[1],bbox[2],bbox[3], conf, cls, name ])
            detections = pd.DataFrame([data], columns=columns)
            boxes, _, _ = self.parse_detections(detections)
            cam = EigenCAM(model, layers)
            grayscale_cam = np.squeeze(cam(rgb_img), axis=0)
            inverted_image = self.invert_grayscale_image(grayscale_cam)
            inv = np.stack([inverted_image] * 3, axis=2)
            g_scale = np.stack([grayscale_cam] * 3, axis=2)
            return g_scale, inv,  rgb_img, bbox, cls, conf, True
    
    





