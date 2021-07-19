import torch
import pandas
import cv2
from os import listdir, remove
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # getting the model from torch hub

folder = "./data/"
imglist = listdir(folder)
deleted = 0
i = 0
for v in imglist: # convert first frame in vids to image
    if (".mp4" in v):
        f = cv2.VideoCapture(folder + v)
        rval, frame = f.read()
        cv2.imwrite(folder + str(v) + ".jpg", frame)
        imglist[i] = str(v) + ".jpg"
    i+= 1

imgs = [folder + f for f in imglist]  # batch of images

# Inference
results = model(imgs)

for i, (im, pred) in enumerate(zip(results.imgs, results.pred)):
    imgX = im.shape[0]
    imgY = im.shape[1]
    
    people = 0
    largestX = 1
    largestY = 1
    for c in pred[:, -1].unique():
        n = (pred[:, -1] == c).sum()  # detections per class
        if results.names[int(c)] == 'person':
            people = int(n)
    for *box, conf, cls in reversed(pred):
        if results.names[int(cls)] == "person":
            width = int(box[2]) - int(box[0])
            height = int(box[3]) - int(box[1])
            if (width * height) > (largestX * largestY):
                largestX = width
                largestY = height

    # area comparison vs image size
    dimvsbb = (largestX * largestY) / (imgX * imgY)
    print("dim vs bounding box: " + str(dimvsbb) + " (need 0.1)")
    print("for " + imglist[i] + ", found " + str(people))
    if people == 0 or people > 3 or dimvsbb < 0.1:
        print("deleting " + imglist[i] + "...")
        try:
            remove(folder + imglist[i])
            if (".mp4" in imglist[i]):
                mp4file = imglist[i].replace(".jpg", "")
                remove(folder + mp4file)
            print("deleted " + imglist[i])
            deleted += 1
        except:
            print("failed to delete " + imglist[i])
    elif (".mp4" in imglist[i]): # remove .jpg frame if video passed
        try:
            remove(folder + imglist[i])
        except:
            pass
        
print("deleted " + str(deleted) + " items")