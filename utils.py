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
import PoseDatabase

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
    candidate, subset, oriImg = get_candidate_subset(path)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()

def get_candidate_subset(path):
    """ helper method that returns the original image, 
        candidate, and subset for key vectors

    Parameters
    ----------
        path: String
            path to image

    Returns
    -------
        tuple - (candidate, subset, oriImg)
            candidate and subset are used to creat key vectors
            oriImg is the loaded image
    """
    body_estimation = Body('model/body_pose_model.pth')
    oriImg = cv2.imread(path)
    candidate, subset = body_estimation(oriImg)
    return (candidate, subset, oriImg)

def get_similarity(path, pose, db_path='vectorDB.pkl'):
    """ gets the similarity between DB's pose and 
        your image's pose
    
    Parameters
    ----------
        path: string
            path to image
        pose: string
            predicted pose from resnet
        db_path: string 
            path to vector database

    Returns
    -------
        similarity: float
            cosine similarity between angle vectors of the key vector
            from DB and the key vector from img
    """
    candidate, subset, oriImg = get_candidate_subset(path)
    vector = compute_angles_vector(get_keypoints(subset, candidate))

    database = PoseDatabase.PoseDatabase()
    database.load_database(db_path)
    pose_vec = database.key_vectors[pose]
    lrf_pose_vec = np.hstack((pose_vec[:, 0].reshape(18,1), 1 - pose_vec[:, 1].reshape(18,1)))

    similarity = min(cosine_sim(vector, compute_angles_vector(pose_vec)), cosine_sim(vector, compute_angles_vector(lrf_pose_vec)))
    #similarity = min(np.linalg.norm(pose_vec - vector), np.linalg.norm(np.hstack((pose_vec[:, 0].reshape(18,1), 1 - pose_vec[:, 1].reshape(18,1))) - vector))
    return similarity

def compute_angles_vector(key_vec):
    """ helper method that computes the angle
        vector for a key vector, it uses the first and
        second key points as the fixed side and calculates
        the angle to all other 16 key points
    
    Parameters
    ----------
        key_vec: np.ndarray, shape-(18,2)
            key vector that stores all the key points
    
    Returns
    -------
        angles: np.ndarray, shape-(16,)
            angles between fixed side and other key points
    """
    fixed_side = key_vec[0,:] - key_vec[1,:]

    angles = np.array([angle(fixed_side, key_vec[0, :] - key_vec[i, :]) if key_vec[i, 0] != -1 else -1 for i in range(2, 18)])
    return angles

def angle(a, b):
    """ helper method that finds angle between two vectors

    Parameters
    ----------
        a: np.ndarray, shape-(1,2)
        b: np.ndarray, shape-(1,2)
    
    Return
    ------
        angle: float
    """
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(cosine_angle)

def cosine_sim(d1, d2):
    """ calculates the cosine similarity

    Parameters
    ----------
        d1: np.ndarray, shape-(16,)
        d2: np.ndarray, shape-(16,)
            two vectors containing the angles between key points

    Returns
    -------
        cosine_sim: float
    """
    return (np.dot(d1, d2)) / (np.linalg.norm(d1)* np.linalg.norm(d2))

def create_img_from_cam():
    """ opens web cam to capture picture + adds a 10 second timer + saves img
        for classification + determining similarity
    """
    cap = cv2.VideoCapture(0)
 
    while True:
        ret, img = cap.read()
        cv2.imshow('Pose Detection',img)
        k = cv2.waitKey(125)
        j = 100
        if k == ord('q'):
            while j>=10:
                ret, img = cap.read()
                if j%10 == 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img,str(j//10),(250,250), font, 7,(255,0,0),10,cv2.LINE_AA)
                cv2.imshow('Pose Detection',img)
                cv2.waitKey(125)
                j = j-1
            else:
                ret, img = cap.read()
                cv2.imshow('Pose Detection',img)
                cv2.waitKey(1000)
                cv2.imwrite('images/camera.jpg',img)
        elif k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

