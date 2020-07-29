import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import PoseDatabase
import utils

from pytorch_openpose import model, util, body

database = PoseDatabase.PoseDatabase()
body_estimation = body.Body('model/body_pose_model.pth')

classes = ["bridge","childs", "downwarddog", "mountain", "plank", 
    "seatedforwardbend", "tree", "trianglepose", "warrior1", "warrior2"]

for i in classes:
    print(i)
    oriImg = cv2.imread('images/' + i + '.jpg')  # B,G,R order
    candidate, subset = body_estimation(oriImg)
    database.add_vector(i, utils.get_keypoints(subset, candidate))

database.save_database('vectorDB.pkl')