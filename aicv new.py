# ================= PRACTICAL 1 — U-Net Segmentation =================
import cv2
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)
model.eval()

img = cv2.imread('a.jpeg')
img = cv2.resize(img, (128,128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

t = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)/255.0

with torch.no_grad():
    out = model(t)
    out = torch.sigmoid(out)
    mask = out[0,0].numpy()

plt.subplot(1,2,1); plt.title('Image'); plt.imshow(img)
plt.subplot(1,2,2); plt.title('Mask'); plt.imshow(mask, cmap='gray')
plt.show()


# ================= PRACTICAL 1a — Linear Regression =================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.array([1000,1500,2000,2500,3000]).reshape(-1,1)
y = np.array([150000,250000,350000,450000,550000])

m = LinearRegression()
m.fit(x,y)

print(m.intercept_)
print(m.coef_[0])
print(m.predict([[2200]]))

plt.scatter(x,y)
plt.plot(x,m.predict(x))
plt.scatter(2200,m.predict([[2200]]))
plt.show()


# ================= PRACTICAL 1b — Logistic Regression =================
import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[120,45],[85,30],[150,50],[70,25],[95,35],[180,55]])
y = np.array([1,0,1,0,0,1])

m = LogisticRegression()
m.fit(x,y)

print(m.predict([[100,40]]))
print(m.predict_proba([[100,40]]))


# ================= PRACTICAL 1c — Decision Tree =================
import numpy as np
from sklearn.tree import DecisionTreeClassifier, tree
import matplotlib.pyplot as plt

x = np.array([[25,0],[30,1],[45,2],[35,1],[50,0]])
y = np.array([0,1,1,1,0])

m = DecisionTreeClassifier(max_depth=3, random_state=42)
m.fit(x,y)

print(m.predict([[40,2]]))

tree.plot_tree(m, feature_names=['age','income'], class_names=['no','yes'], filled=True)
plt.show()


# ================= PRACTICAL 1d — KNN =================
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x = np.array([[2,5],[4,6],[5,7],[1,4],[3,5],[6,8]])
y = np.array([0,1,1,0,0,1])

m = KNeighborsClassifier(n_neighbors=3)
m.fit(x,y)

print(m.predict([[4,5]]))
print(m.predict_proba([[4,5]]))


# ================= PRACTICAL 2 — Logistic Regression Single Feature =================
import numpy as np
from sklearn.linear_model import LogisticRegression

x = np.array([[2],[4],[6],[8],[10]])
y = np.array([0,0,1,1,1])

m = LogisticRegression()
m.fit(x,y)

print(m.predict([[5]]))
print(m.predict_proba([[5]]))


# ================= PRACTICAL 3 — KMeans =================
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

x = np.array([[15,39],[16,81],[17,6],[18,7],[19,40],[20,80]])

km = KMeans(n_clusters=3)
km.fit(x)

lbl = km.labels_
cen = km.cluster_centers_

plt.scatter(x[:,0], x[:,1], c=lbl)
plt.scatter(cen[:,0], cen[:,1], marker='X', s=200)
plt.show()


# ================= PRACTICAL 4 — OpenCV Basics =================
import cv2

img = cv2.imread('a1.jpeg')
cv2.imshow('original', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)

crop = img[50:200,100:200]
cv2.imshow('crop', crop)

(h,w) = img.shape[:2]
M = cv2.getRotationMatrix2D((w/2,h/2),270,1.0)
rot = cv2.warpAffine(img, M, (w,h))
cv2.imshow('rot', rot)

rsz = cv2.resize(img,(100,100))
cv2.imshow('resize', rsz)

blr = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('blur', blr)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ================= PRACTICAL 5 — YOLO =================
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

img = cv2.imread('a.jpeg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

res = model(rgb)
out = res[0].plot()

cv2.imshow('det', out)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ================= PRACTICAL 6 — Haar Face Detection =================
import cv2

cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('a1.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cas.detectMultiScale(gray,1.1,7,(30,30))

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ================= PRACTICAL 7 — ORB Matching =================
import cv2

img1 = cv2.imread('a1.jpeg',0)
img2 = cv2.imread('a2.jpeg',0)

orb = cv2.ORB_create()

kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key=lambda x:x.distance)

res = cv2.drawMatches(img1,kp1,img2,kp2,matches[:5],None,flags=2)

cv2.imshow('ORB', res)
cv2.waitKey(0)
cv2.destroyAllWindows()