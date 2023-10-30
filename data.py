import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import os

class Utils: 
    def __init__(self, path, cls, out):
        self.data_path = path
        self.cls = cls
        self.out = out
        self.dict = {}
        self.data = ["Confidence", "Intersection Over Union", "Initial Confidence"]
        self.output_folder = "output"

    def _instance(self):
        if self.cls != 7:
            self.dict = self.acquire(self.cls)
        else:
            for i in range(6):
                name = "class"+str(i)
                self.dict[name] = self.acquire(i)

    def acquire(self, x):
        dicts = {}
        files = []
        data = ["initial_conf", "conf_tot", "iou_tot"]
        for i in range(len(data)):
            files.append(self.data_path + "0" + str(x) + "_" + str(data[i]) + ".pkl")
        for i,file in enumerate(files):
            with open(file, 'rb') as dict:
                loaded_dict = pickle.load(dict)
                dicts[data[i]] = loaded_dict
        return dicts
    
    def remove_none(self, dict):
        sub_dict = dict["initial_conf"]
        for key in sub_dict:
            if sub_dict[key] is None:
                sub_dict[key] = 0
        return sub_dict
    
    def sum_norm(self,dict):
        sum_values = []
        for index in range(len(next(iter(dict.values())))):
            index_sum = 0
            for key, value_list in dict.items():
                if not dict[key]:
                    index_sum += 0
                else:
                    index_sum += value_list[index]
            sum_values.append(index_sum)
        min_value = min(sum_values)
        max_value = max(sum_values)
        normalized_values = [(x - min_value) / (max_value - min_value) for x in sum_values]
        return normalized_values

    def baseline_conf(self, dict):
        dict_correct = self.remove_none(dict)
        return sum(dict_correct.values()) / len(dict_correct)
    
    def find_min(self,dict):
        dict_correct = self.remove_none(dict)
        filtered_values = [value for value in dict_correct.values() if value != 0]
        if filtered_values:
            min_value = min(filtered_values)
            return [key for key, value in dict_correct.items() if value == min_value]
    
    def sort_remove(self, dict):
        dict_correct = self.remove_none(dict)
        return {k: v for k, v in sorted(dict_correct.items(), key=lambda item: item[1]) if v != 0}

    def plot_initial(self):
        correct_dict = self.remove_none(self.dict)
        keys = range(len(list(correct_dict.keys())))
        values = list(correct_dict.values())
        name = "Class " + str(self.cls)
        plt.figure(figsize=(15, 10)) 
        plt.bar(keys, values)
        plt.xlabel('Images')
        plt.ylabel('Confidence')
        plt.title('Initial Confidence score for each image in ' + name)
        filename = "initial_conf"
        self.save_plot(filename) 
        plt.show()

    def plot(self, flag):
        step = []
        conf = self.sum_norm(self.dict["conf_tot"])
        iou = self.sum_norm(self.dict["iou_tot"])
        baseline = self.baseline_conf(self.dict)
        name = "Class " + str(self.cls)
        for i in range(0, 101, 5):
            step.append(i)
        if flag:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7)) 
            ax1.plot(step, conf, label= "Confidence", marker='o', linestyle='-')
            ax1.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title("Confidence " + name)
            ax2.plot(step, iou, label= "IoU",marker='o', linestyle='-')
            ax2.set_title("IoU " + name)
            ax2.grid(True)
            filename = "iou_conf_" + name
        else:
            plt.figure(figsize=(15, 10))  
            filename = "initial_conf_" + name
            plt.plot(step, conf, label= "Confidence", marker='o', linestyle='-')
            plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
            plt.legend()
            plt.title(name)
            plt.grid(True)
        plt.tight_layout()
        filename = "initial_conf"
        self.save_plot(filename) 
        plt.show()
    
    def plot_example(self, ex):
        example_inital = self.dict["initial_conf"][ex]
        example = self.dict["conf_tot"][ex]
        for i in range(len(example)):
            if example[i] == -1:
                example[i] = 0
        conf_tot = [x - example_inital for x in example]
        steps = []
        for i in range(0, 101, 5):
            steps.append(i)
        plt.plot(steps, example, label = "Modified", marker='o', linestyle='-')
        plt.axhline(y=example_inital, color='r', linestyle='--', label='Baseline')
        plt.legend()
        plt.xlabel('Pixel perturbation')
        plt.ylabel('Confidence')
        plt.title('Confidence vs. Pixel perturbation')
        plt.grid(True)
        # filename = "initial_conf"
        # self.save_plot(filename) 
        plt.show()

    def plt_tot(self):
        step = []
        for u in range(0, 101, 5):
            step.append(u)

        plt.figure(figsize=(20, 10))
        for i in range(6):
            x = i
            cls = self.acquire(i)
            name = self.data[self.out]
            if self.out == 0:
                var = self.sum_norm(cls["conf_tot"])
                baseline = self.baseline_conf(cls)
            elif self.out == 1:
                var = self.sum_norm(cls["iou_tot"])
            elif self.out == 2:
                var = self.remove_none(cls)
                keys = range(len(list(var.keys())))
                values = list(var.values())
    
            ax = plt.subplot(2, 3, i + 1)
            cla = "Class " + str(x)
            
            if self.out == 2:
                plt.bar(keys, values)
            else:
                plt.plot(step,var, label= name)
            if self.out == 0:
                plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
                plt.legend()

            plt.xlabel('Pixel perturbation')
            plt.ylabel(name)
            title = name + " vs. Pixel perturbation " + cla
            plt.title(title)
            plt.grid(True) 
            plt.axis("on")
            ttitle = name + " variation with Uniform distribution perturbation"
            plt.suptitle(ttitle, fontsize=23, fontweight='bold', backgroundcolor='lightgray')
        filename = "total_plot_" + name + "_.png"
        self.save_plot(filename)
        plt.show()

    def save_plot(self, filename):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        plt.savefig(os.path.join(self.output_folder, filename))
        
    def save_csv(self):
        dict_conf = {
            "class0" : [[],[]],
            "class1" : [[],[]],
            "class2" : [[],[]],
            "class3" : [[],[]],
            "class4" : [[],[]],
            "class5" : [[],[]]
        }
        name = ["conf","iou"]
        pert = self.perturbation[self.type]
        for i in range(6):
            data = self.acquire(i)
            conf = self.sum_norm(data["conf_tot"])
            iou = self.sum_norm(data["iou_tot"])
            cls = "class"+str(i)
            dict_conf[cls][0] = conf
            dict_conf[cls][1] = iou

        for x in range(2):
            filename = str(name[x]) + ".csv"
            path = os.path.join(self.output_folder, filename)
            with open(path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = list(dict_conf.keys())  
                writer.writerow(header)
                for i in range(len(list(dict_conf.values())[0][0])):
                    row = []
                    for key, value in dict_conf.items():
                        row.append(value[x][i])
                    writer.writerow(row)




