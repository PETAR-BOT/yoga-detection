# yoga-detection
Detecting and classifying yoga poses using transfer learning and OpenPose.

# Quick Start
Clone the repository, navigate to the directory, and run the interface.py file.

# How it works
Our project is two pronged: yoga classification and determining the similarity between yoga poses.

Yoga classification is done through computer vision, using the pretrained Resnet34 framework to train on a dataset found online. This dataset contains 10 yoga classes such as downward dog, mountain, and tree poses. We achieved a 97% accuracy on the testing data.

To compare whether a pose is bad or good, we used the Pytorch implementation of the OpenPose library, which has human pose models that detect up to 18 body parts such as both ears, elbows, shoulders, and knees. Openpose returns the coordinates of each body part and we normalize the coordinates. We calculated the angles between each of the 18 key points by using two key points as a fixed side, namely the nose and neck key points. We determined the angle formed between the fixed side and all the other body parts and stored the angles in a vector. By finding the cosine similarity between different angle vectors, we determined whether a pose is comparable.

# Resources
Colab notebook used to train the Resnet34 framework: https://colab.research.google.com/drive/1W7c1KjxoP3C0TYQiLtZ8KGTsQ_RbbFTd?usp=sharing

Training Data: https://drive.google.com/drive/folders/1vsBgnEf2NcQNvlCnIZr0P1m2_XXLhLlQ?usp=sharing

Testing Data: https://drive.google.com/drive/folders/1WW7n4C01nBgBGNsQBLiRrsTB7h0WFIOB?usp=sharing
