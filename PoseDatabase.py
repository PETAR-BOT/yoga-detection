import pickle
from pathlib import Path
import numpy as np
from collections import Counter

class PoseDatabase:
    """ a database that stores the yoga poses key vectors
    """

    def __init__(self):
        """ initializes DB
        """
        # maps yoga pose to vector
        self.key_vectors = {}
    
    def add_vector(self, pose, vector):
        """ adds a pose to the database

        Parameters
        ----------
            pose: string
                name of the yoga pose
            vector: np.ndarray shape-(18, 2) ??
                normalized vector of the pose
        """
        # assuming the pose is not in db
        if self.key_vectors[pose] is None:
            self.key_vectors[pose] = vector
        else:
            return "pose is already in database"
        
    def replace_pose_vector(self, pose, vector):
        """ updates a pose (because we only want one)

        Parameters
        ----------
            pose: string
                name of the yoga pose
            vector: np.ndarray shape-(18, 2) ??
                normalized vector of the pose
        """
        if self.key_vectors[pose] is not None:
            self.key_vectors[pose] = vector
        else:
            return "pose is not in database, use add_vector method"
        
    def find_match(self, vector, threshold=1.):
        """ queries the database to find matching pose

        Parameters
        ----------
            vector: np.ndarray shape-(18, 2)
                normalized vector of a pose
            threshold: float
                threshold for a successful classification

        Returns
        -------
            pose: String
                name of the pose
        """
        distances = Counter({pose: min(np.linalg.norm(vec, vector), np.linalg.norm(np.fliplr(vec), vector)) for pose, vec in self.key_vectors})
        (pose2, d2), (pose1, d1) = distances.most_common()[-2:]
        if d2 >= threshold*d1:
            return pose1
        else:
            return "did not meet classification threshold"

    def load_database(self, path):
        """ takes in the path of the database, and sets the loaded database

        Parameters
        ----------
            path: String
                The path of the database
            
        """
        path = Path(path)
        with open(path, mode="rb") as opened_file:
            self.key_vectors = pickle.load(opened_file)

    def save_database(self, filename):
        """ pickles dict to filename

        Parameters
        ----------
            filename: String
                the name of the file
        """
        with open(filename, mode="wb") as opened_file:
            return pickle.dump(self.key_vectors, opened_file)