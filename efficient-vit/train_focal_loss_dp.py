from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F
from random import random, randint, choice
from vit_pytorch import ViT
import numpy as np
from torch.optim import lr_scheduler
import os
import json
from os import cpu_count
from multiprocessing.pool import Pool
from functools import partial
from multiprocessing import Manager
from progress.bar import ChargingBar
# from efficient_sovit_v2 import EfficientViT
# from efficient_sovit_v3 import EfficientViT
# from efficient_sovit import EfficientViT
from multi_sovit_linear import EfficientViT
# from efficient_vit import EfficientViT
import uuid
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score
import cv2
from transforms.albu import IsotropicResize
import glob
import pandas as pd
from tqdm import tqdm
from utils import get_method, check_correct, resize, shuffle_dataset, get_n_params
from sklearn.utils.class_weight import compute_class_weight 
from torch.optim.lr_scheduler import LambdaLR
import collections
from deepfakes_dataset import DeepFakesDataset
import math
import yaml
import argparse
import matplotlib.pyplot as plt


#从test程序码中获取
MODELS_DIR = "models"
# BASE_DIR = "../../deep_fakes"



# BASE_DIR="H:\\dataset_test"
BASE_DIR="/share/home/zhangdz/mjw"
# VALIDATION_LABELS_PATH = os.path.join(BASE_DIR, "dfdc_test_labels.csv")
VALIDATION_LABELS_PATH = os.path.join(BASE_DIR, "dataset/dfdc_test_labels.csv")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
MODELS_PATH = "models"



# BASE_DIR = '../../deep_fakes/'
# DATA_DIR = os.path.join(BASE_DIR, "dataset")
# TRAINING_DIR = os.path.join(DATA_DIR, "training_set")
# VALIDATION_DIR = os.path.join(DATA_DIR, "validation_set")
# TEST_DIR = os.path.join(DATA_DIR, "test_set")
# MODELS_PATH = "models"
# METADATA_PATH = os.path.join(BASE_DIR, "data/metadata") # Folder containing all training metadata for DFDC dataset
# VALIDATION_LABELS_PATH = os.path.join(DATA_DIR, "dfdc_val_labels.csv")

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


def read_frames(video_path, train_dataset, validation_dataset,opt,config):
    
    # Get the video label based on dataset selected
    method = get_method(video_path, DATA_DIR)
    if TRAINING_DIR in video_path:  #training中的label来源于json文件
        # if "Original" in video_path:
        #     label = 0.
        # elif "DFDC" in video_path:
        #     for json_path in glob.glob(os.path.join(METADATA_PATH, "*.json")):
        #         with open(json_path, "r") as f:
        #             metadata = json.load(f)
        #         video_folder_name = os.path.basename(video_path)
        #         video_key = video_folder_name + ".mp4"
        #         if video_key in metadata.keys():
        #             item = metadata[video_key]
        #             label = item.get("label", None)
        #             if label == "FAKE":
        #                 label = 1.
        #             else:
        #                 label = 0.
        #             break
        #         else:
        #             label = None
        # else:
        #     label = 1.
        # if label == None:
        #     print("NOT FOUND", video_path)
        if "Original" in video_path:
            label = 0.
        elif "DFDC" in video_path:
            val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
            video_folder_name = os.path.basename(video_path)
            video_key = video_folder_name + ".mp4"
            label = val_df.loc[val_df['filename'] == video_key]['label'].values[0]
        else:
            label = 1.
    else:         #validation中的label来源于csv
        if "Original" in video_path:
            label = 0.
        elif "DFDC" in video_path:
            val_df = pd.DataFrame(pd.read_csv(VALIDATION_LABELS_PATH))
            video_folder_name = os.path.basename(video_path)
            video_key = video_folder_name + ".mp4"
            label = val_df.loc[val_df['filename'] == video_key]['label'].values[0]
        else:
            label = 1.

    # Calculate the interval to extract the frames
    frames_number = len(os.listdir(video_path))
    if label == 0:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_real']),1) # Compensate unbalancing
    else:
        min_video_frames = max(int(config['training']['frames-per-video'] * config['training']['rebalancing_fake']),1)

    
    if VALIDATION_DIR in video_path:
        min_video_frames = int(max(min_video_frames/8, 2))
    frames_interval = int(frames_number / min_video_frames)
    frames_paths = os.listdir(video_path)
    frames_paths_dict = {}

    # Group the faces with the same index, reduce probabiity to skip some faces in the same video
    for path in frames_paths:
        for i in range(0,1):
            if "_" + str(i) in path:
                if i not in frames_paths_dict.keys():
                    frames_paths_dict[i] = [path]
                else:
                    frames_paths_dict[i].append(path)

    # Select only the frames at a certain interval
    if frames_interval > 0:
        for key in frames_paths_dict.keys():
            if len(frames_paths_dict) > frames_interval:
                frames_paths_dict[key] = frames_paths_dict[key][::frames_interval]
            
            frames_paths_dict[key] = frames_paths_dict[key][:min_video_frames]

    # Select N frames from the collected ones
    for key in frames_paths_dict.keys():
        for index, frame_image in enumerate(frames_paths_dict[key]):
            #image = transform(np.asarray(cv2.imread(os.path.join(video_path, frame_image))))
            image = cv2.imread(os.path.join(video_path, frame_image))
            image=cv2.resize(image,(224,224)) #预处理，将训练图像换成224x224规格，从而让之后的训练能够正确进行
            if image is not None:
                if TRAINING_DIR in video_path:
                    train_dataset.append((image, label)) #train_dataset是一个组，将image、label作为其中存在的元素列添加进这个组中
                else:
                    validation_dataset.append((image, label))

# Main body
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none).')
    parser.add_argument('--dataset', type=str, default='All', 
                        help="Which dataset to use (Deepfakes|Face2Face|FaceShifter|FaceSwap|NeuralTextures|All)")
    parser.add_argument('--max_videos', type=int, default=-1, 
                        help="Maximum number of videos to use for training (default: all).")
    parser.add_argument('--config', type=str, 
                        help="Which configuration to use. See into 'config' folder.")
    parser.add_argument('--efficient_net', type=int, default=0, 
                        help="Which EfficientNet version to use (0 or 7, default: 0)")
    parser.add_argument('--patience', type=int, default=5,
                        help="How many epochs wait before stopping for validation loss not improving.")




    opt = parser.parse_args()
    print(opt)

    with open(opt.config, 'r') as ymlfile:
        config = yaml.safe_load(ymlfile)
 
    if opt.efficient_net == 0:
        channels = 1280
    else:
        channels = 2560
        
    model = EfficientViT(config=config, channels=channels, selected_efficient_net = opt.efficient_net)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight-decay'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step-size'], gamma=config['training']['gamma'])
    starting_epoch = 0
    if os.path.exists(opt.resume):
        model.load_state_dict(torch.load(opt.resume))
        starting_epoch = int(opt.resume.split("checkpoint")[1].split("_")[0]) + 1 # The checkpoint's file name format should be "checkpoint_EPOCH"
    else:
        print("No checkpoint loaded.")


    print("Model Parameters:", get_n_params(model))
    

   
    #READ DATASET
    if opt.dataset != "All" and opt.dataset != "DFDC":
        folders = ["Original", opt.dataset]
    else:
        # folders = ["Original", "DFDC", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]
        folders = [ "DFDC"]

    sets = [TRAINING_DIR, VALIDATION_DIR]

    paths = []
    for dataset in sets:
        for folder in folders:
            subfolder = os.path.join(dataset, folder)
            for index, video_folder_name in enumerate(os.listdir(subfolder)):
                if index == opt.max_videos:
                    break

                if os.path.isdir(os.path.join(subfolder, video_folder_name)):
                    paths.append(os.path.join(subfolder, video_folder_name))
                
                

    mgr = Manager()
    train_dataset = mgr.list() #    创建了一个支持多进程共享的列表对象，可以在多个进程中对这个列表进行读写操作，实现数据的共享。
    validation_dataset = mgr.list()


#pool存在是为了并行处理的实现，这段代码作用为并行地将视频文件用read_frames处理
    with Pool(processes=opt.workers) as p:
        with tqdm(total=len(paths)) as pbar:
            for v in p.imap_unordered(partial(read_frames, train_dataset=train_dataset, validation_dataset=validation_dataset,opt=opt,config=config),paths):
                pbar.update()
    
    train_samples = len(train_dataset)
    train_dataset = shuffle_dataset(train_dataset)
    validation_samples = len(validation_dataset)
    validation_dataset = shuffle_dataset(validation_dataset) #对dataset进行乱序操作，增加随机性

    # Print some useful statistics
    print("Train images:", len(train_dataset), "Validation images:", len(validation_dataset))
    print("__TRAINING STATS__")
    train_counters = collections.Counter(image[1] for image in train_dataset)
    print(train_counters)
    print(train_counters[0])
    print(train_counters[1])
    class_weights = train_counters[0] / train_counters[1]
    print("Weights", class_weights)

    print("__VALIDATION STATS__")
    val_counters = collections.Counter(image[1] for image in validation_dataset)
    print(val_counters)
    print("___________________")

    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights]))
    loss_fn=BCEFocalLoss()

    # Create the data loaders
    validation_labels = np.asarray([row[1] for row in validation_dataset])
    labels = np.asarray([row[1] for row in train_dataset])

    # for row in train_dataset:
    #     print(row[0].shape)


    train_dataset = DeepFakesDataset(np.asarray([row[0] for row in train_dataset]), labels, config['model']['image-size'])
    dl = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                 batch_sampler=None, num_workers=opt.workers, collate_fn=None,
                                 pin_memory=False, drop_last=False, timeout=0,
                                 worker_init_fn=None, prefetch_factor=2,
                                 persistent_workers=False)
    del train_dataset

    validation_dataset = DeepFakesDataset(np.asarray([row[0] for row in validation_dataset]), validation_labels, config['model']['image-size'], mode='validation')
    val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=config['training']['bs'], shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=0, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)
    del validation_dataset

    model = nn.DataParallel(model.cuda(), device_ids=[0,1])
    counter = 0
    not_improved_loss = 0
    previous_loss = math.inf
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for t in range(starting_epoch, opt.num_epochs + 1):
        if not_improved_loss == opt.patience:
            break
        counter = 0

        total_loss = 0
        total_val_loss = 0
        
        bar = ChargingBar('EPOCH #' + str(t), max=(len(dl)*config['training']['bs'])+len(val_dl))
        train_correct = 0
        positive = 0
        negative = 0
        for index, (images, labels) in enumerate(dl):
            images = np.transpose(images, (0, 3, 1, 2))
            labels = labels.unsqueeze(1)
            images = images.cuda()
            
            y_pred = model(images)
            y_pred = y_pred.cpu()
            loss = loss_fn(y_pred, labels.float())
        
            corrects, positive_class, negative_class = check_correct(y_pred, labels)  
            train_correct += corrects
            positive += positive_class
            negative += negative_class
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            counter += 1
            total_loss += round(loss.item(), 2)
            
            if index%1200 == 0: # Intermediate metrics print
                print("\nLoss: ", total_loss/counter, "Accuracy: ",train_correct/(counter*config['training']['bs']) ,"Train 0s: ", negative, "Train 1s:", positive)


            for i in range(config['training']['bs']):
                bar.next()

        val_correct = 0
        val_positive = 0
        val_negative = 0
        val_counter = 0


        train_correct /= train_samples
        total_loss /= counter
        for index, (val_images, val_labels) in enumerate(val_dl):
    
            val_images = np.transpose(val_images, (0, 3, 1, 2))
            
            val_images = val_images.cuda()
            val_labels = val_labels.unsqueeze(1)
            val_pred = model(val_images)
            val_pred = val_pred.cpu()
            val_labels = val_labels.long()
            val_loss = loss_fn(val_pred, val_labels.float())
            total_val_loss += round(val_loss.item(), 2)
            corrects, positive_class, negative_class = check_correct(val_pred, val_labels)
            val_correct += corrects
            val_positive += positive_class
            val_counter += 1
            val_negative += negative_class
            bar.next()
            
        scheduler.step()
        bar.finish()


        total_val_loss /= val_counter
        val_correct /= validation_samples
        if previous_loss <= total_val_loss:
            print("Validation loss did not improved")
            not_improved_loss += 1
        else:
            not_improved_loss = 0
        
        previous_loss = total_val_loss
        print("#" + str(t) + "/" + str(opt.num_epochs) + " loss:" +
            str(total_loss) + " accuracy:" + str(train_correct) +" val_loss:" + str(total_val_loss) + " val_accuracy:" + str(val_correct) + " val_0s:" + str(val_negative) + "/" + str(np.count_nonzero(validation_labels == 0)) + " val_1s:" + str(val_positive) + "/" + str(np.count_nonzero(validation_labels == 1)))


        if not os.path.exists(MODELS_PATH):
            os.makedirs(MODELS_PATH)
        torch.save(model.module.state_dict(), os.path.join(MODELS_PATH,
                                                    "efficientnetB" + str(opt.efficient_net) + "_checkpoint" + str(
                                                        t) + "_" + opt.dataset))



        train_losses.append(round(float(str(total_loss)),3))
        val_losses.append(round(float(str(total_val_loss)),3))
        train_accuracies.append(round(float(str(train_correct)),3))
        val_accuracies.append(round(float(str(val_correct)),3))


    plt.figure(figsize=(12, 4))


    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.yticks([])
    plt.ylim(0,1)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.yticks([])  # 不显示纵轴刻度标签
    plt.ylim(0, 1)

    # 保存图形到本地文件
    plt.savefig('training_validation_plot.png')
        
