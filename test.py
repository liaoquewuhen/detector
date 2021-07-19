from mnist1d import detection,load_model
import numpy as np
from PIL import Image

img = Image.open(r"D:\py\textDectectionDocker\mnist_5.jpg")
im_nd = np.array(img)
net = load_model()
print(int(detection(net=net,img=im_nd)))
