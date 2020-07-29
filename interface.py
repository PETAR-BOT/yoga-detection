import utils

print('\n\nWelcome to the Yoga pose estimator')
print('----------------------------------\n')

print('Please enter 1 or 2: ')
print('1. Yoga pose from web camera')
choice = int(input('2. Yoga picture from computer\n'))
if choice == 1:
    print('Press q to start the count down timer and esc to exit once the picture is taken')
    utils.create_img_from_cam()
    path = 'images/camera.jpg'
elif choice == 2:
    path = input('Please enter the path of your image: ')
else:
    print('Choice is invalid')

pose = utils.classify_pose(path)
print("Yoga pose: ", pose)
print("Similarity: ", utils.get_similarity(path, pose))
utils.show_vectors(path)