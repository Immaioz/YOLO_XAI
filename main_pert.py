from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image
import random
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import time


class Perturbation:
    
    def perturbation(self, img, mask, percentile, task, bbox, cls, model):
        results = model(img, verbose=False, max_det=1)
        boxes = results[0].boxes
        original = results[0].plot()
        conf = float(boxes.conf.cpu().numpy())
        # data = results[0].boxes.data.cpu().numpy()
        # print(data)
        if(task == "remove"):
            perturbed = img * mask
            #perturbed, _ = replacepixel(img, mask, percentile, task)
        if(task == "mean"):
            perturbed, _ = self.replacepixel(img, mask, percentile, task)
        if(task == "random"):
            perturbed, _ = self.replacepixel(img, mask, percentile, task)
        results = model([perturbed], max_det=1, classes = cls)
        if len(results[0].boxes) != 0:
            iou = self.calculate_iou(bbox,results[0].boxes.xyxy[0].cpu().numpy())
            # data = results[0].boxes.data.cpu().numpy()
            # print(data)
            new = results[0].plot()
            boxes = results[0].boxes
            conf_incr =  float(boxes.conf.cpu().numpy())
            out = np.stack((original, new), axis=0)
            return out, conf_incr, iou
        else:
            return None, None, None

    def replacepixel(self, img, mask, percentile_value, task):
        mask = mask *255
        mask = mask.astype(np.uint8)
        image = Image.fromarray(img)
        mask = Image.fromarray(mask)
        # percentile_value = 100 - percentile_value
        mask_intensity = np.percentile(mask, percentile_value)
        # Get pixel access objects
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

        # Convert the modified Pillow Image back to a numpy array
        modified_image = np.array(image)
        modified_mask = np.array(modified_mask)
        return modified_image, modified_mask



    def calculate_iou(self,box1, box2):
        # Convert XYXY format to (x1, y1, x2, y2) format
        x1_box1, y1_box1, x2_box1, y2_box1 = box1
        x1_box2, y1_box2, x2_box2, y2_box2 = box2

        # Calculate the area of each bounding box
        area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
        area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

        # Calculate the coordinates of the intersection box
        x1_intersection = max(x1_box1, x1_box2)
        y1_intersection = max(y1_box1, y1_box2)
        x2_intersection = min(x2_box1, x2_box2)
        y2_intersection = min(y2_box1, y2_box2)

        # Calculate the area of the intersection box
        if x1_intersection < x2_intersection and y1_intersection < y2_intersection:
            area_intersection = (x2_intersection - x1_intersection) * (y2_intersection - y1_intersection)
        else:
            area_intersection = 0.0

        # Calculate IoU
        iou = area_intersection / (area_box1 + area_box2 - area_intersection)

        return iou





    def test_layers(self, img, model, rgb_img, cam, grayscale_cam):
        plt.figure(figsize=(15, 8))
        u = 0
        for i in range(16,1,-1):
            target_layers = [model.model.model[-(i)]]
            cam = EigenCAM(model, target_layers)
            grayscale_cam = cam(rgb_img)[0, :]
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            #print(cam_image.shape)
            ax = plt.subplot(3, 5, u + 1)
            u += 1
            #print(u)
            plt.imshow(cam_image)
            plt.text(0.5, -0.1, f'Layer -{i}', transform=ax.transAxes,
                    fontsize=12, ha='center', va='center', color='red')
            plt.axis("off")





    def parse_detections(self, detections):
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
            color = COLORS[category]

            boxes.append((xmin, ymin, xmax, ymax))
            colors.append(color)
            names.append(name)
        return boxes, colors, names
    
    def draw_detections(boxes, colors, names, img):
        for box, color, name in zip(boxes, colors, names):
            xmin, ymin, xmax, ymax = box
            cv2.rectangle(
                img,
                (xmin, ymin),
                (xmax, ymax),
                color,
                2)

            cv2.putText(img, name, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        lineType=cv2.LINE_AA)
        return img
    
    def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        image_with_bounding_boxes = self.draw_detections(boxes, colors, names, eigencam_image_renormalized)
        return image_with_bounding_boxes
    
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
        return img
    
    def camtest(image_path):
        img = self.get_image(image_path)
        rgb_img = img.copy()
        img = img.astype(np.float32) / 255.0
        results = model([rgb_img], verbose=False)
        boxes = results[0].boxes
        columns = ['xmin', 'ymin',	'xmax', 'ymax',	'confidence', 'class', 'name']
        bbox = boxes.xyxy[0].cpu().numpy()
        conf = float(boxes.conf.cpu().numpy())
        cls = int(boxes.cls[0])
        name = (results[0].names[int(boxes.cls[0])])
        print("Confidence:", conf)
        print("Class: ", name)
        data = np.array([bbox[0],bbox[1],bbox[2],bbox[3], conf, cls, name ])
        detections = pd.DataFrame([data], columns=columns)
        boxes, colors, names = self.parse_detections(detections)
        detection = results[0].plot()
        cam = EigenCAM(model, target_layers)
        grayscale_cam = cam(rgb_img)[0, :, :]
        g_scale = np.stack([grayscale_cam] * 3, axis=2)
        g_scale = np.array(g_scale) *255
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
        renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)
        return detection, cam_image, g_scale, renormalized_cam_image





    def invert_grayscale_image(image):
        # Calculate the maximum pixel value for the grayscale image (usually 255 for 8-bit images)
        max_pixel_value = np.max(image)

        # Invert the image by subtracting each pixel value from the maximum
        inverted_image = max_pixel_value - image

        return inverted_image





def acquire(img, model, layers):
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
        boxes, colors, names = parse_detections(detections)
        cam = EigenCAM(model, layers)
        grayscale_cam = np.squeeze(cam(rgb_img), axis=0)
        inverted_image = invert_grayscale_image(grayscale_cam)
        inv = np.stack([inverted_image] * 3, axis=2)
        g_scale = np.stack([grayscale_cam] * 3, axis=2)
        return g_scale, inv,  rgb_img, bbox, cls, conf, True
    
    
