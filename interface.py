import utils

print('\n\nWelcome to the Yoga pose estimator')
print('----------------------------------\n')
path = input('Please enter the path of your image: ')

print("Yoga pose: ", utils.classify_pose(path))
utils.show_vectors(path)