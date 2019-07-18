from embedding import getRep
import cv
import pickle
import glob
from tqdm import tqdm
import time

# Generate Embedding for all the images present in train directory
images = glob.glob('data/train/*.JPG')
embedding_dict = {}
pbar = tqdm(total=len(images))
for imagePath in images:
    bgrImg = cv2.imread(imgPath)
    embedding = getRep(bgrImg, imgPath)
    fileName = imagePath.split('/')[-1].split('.')[0]
    embedding_dict[fileName] = embedding
    pbar.update(1)
pbar.close()

# Save image embedding to Disk
f = open("embedding_" + str(int(time.time())) + ".pkl","wb")
pickle.dump(dict,f)
f.close()