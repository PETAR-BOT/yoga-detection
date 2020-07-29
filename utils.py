import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from pytorch_openpose import model
from pytorch_openpose import util
from pytorch_openpose.body import Body

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image

def get_keypoints(subset, candidate):
    """
    Returns keypoints for the first person in subset

    Parameters
    ----------
    subset : np.ndarray
        indices of candidates in each subset, total score, and number of parts

    candidate : np.ndarray
        all of the candidates

    Returns
    -------
    keypoints : np.ndarray
        coordinates of keypoints (-1 if it was not found)
    """
    parts = np.asarray(subset[0][:-2], dtype=np.int)
    keypoints = candidate[:, :2][parts]
    #print(keypoints)
    keypoints = keypoints - np.amin(keypoints, axis=0)
    #print(keypoints)
    keypoints = keypoints / np.amax(keypoints, axis=0)
    #print(keypoints)
    keypoints[parts == -1] = -1 * np.ones(2)

    return keypoints

def classify_pose(path):
    """ returns the yoga pose classification from
        the retrained resnet34 model

    Parameters
    ----------
        path: string
            path to image

    Returns
    -------
        classification: string
            yoga pose classification
    """
    idx_to_class = {0: "bridge", 1: "childs", 2: "downwarddog", 3: "mountain", 4: "plank", 
    5: "seatedforwardbend", 6: "tree", 7:"trianglepose", 8: "warrior1", 9:"warrior2"}

    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)


    device = torch.device('cpu')
    state_dict = torch.load('model/resnet34-pose.pth', map_location=device)
    model.load_state_dict(state_dict)

    test_image = 'images/mountain.png'
    oriImg = cv2.imread(test_image)
    oriImg = oriImg[:,:, ::-1] # RGB 

    normalize = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_pil = Image.open(path).convert("RGB")
    img_tensor = normalize(img_pil).float()
    img_tensor = img_tensor.unsqueeze_(0)

    model.eval()
    with torch.no_grad():
        fc_out = model(img_tensor)
        output = fc_out.detach().numpy()
        return idx_to_class[output.argmax()]

def show_vectors(path):
    """ returns the image with the vectors drawn on

    Parameters
    ----------
        path: String
            path to image
    
    Returns
    -------
        plot: matplotlib.pyplot
            img with vectors
    """
    # From demo.py file
    body_estimation = Body('model/body_pose_model.pth')
    oriImg = cv2.imread(path)
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()