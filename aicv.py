//practical 1
import cv2
import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# load model
model = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     # Downloads pre-trained features
    in_channels=3,                  
    classes=1                       
)
model.eval()

# read image
img = cv2.imread('a.jpeg')
img = cv2.resize(img, (128, 128))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# convert to tensor
img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0

# prediction
with torch.no_grad():
    mask = model(img_tensor)
    mask = torch.sigmoid(mask)        # apply sigmoid here if not in model
    mask_np = mask[0, 0].numpy()

# visualization
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(img)

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(mask_np, cmap="gray")
        
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

//practical 1a
#step 1- import library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x=np.array([1000,1500,2000,2500,3000]).reshape(-1,1) #x is a feature, for squarefoot. Only 1 target
y=np.array([150000,250000,350000,450000,550000]) # our target
#train the model
model= LinearRegression()
model.fit(x,y) #feature then pattern, giving data pattern to model to predict
#find m,c intercept and coeff
print("Intercept: ",model.intercept_)
print("slope: ",model.coef_[0])
#predict the price for 2200
predicted_price= model.predict([[2200]])
print("Predicted price: ",predicted_price)
#visualization
plt.scatter(x,y,color="blue",label="actualdata")

#can plot redicted price too
plt.plot(x,model.predict(x),color="red",label="regression")

#mark the predicted price
plt.scatter(2200,predicted_price,color="yellow",label="predicted price")
plt.xlabel("square_footage")
plt.ylabel("Price")
plt.legend()
plt.show()
 
//practical 1b
import numpy as np
from sklearn.linear_model import LogisticRegression
x=np.array([[120,45],[85,30],[150,50],[70,25],[95,35],[180,55]]) #glucose and age
y=np.array([1,0,1,0,0,1]) #yes or no
model=LogisticRegression()
model.fit(x,y)
prediction= model.predict([[100,40]])
probability=model.predict_proba([[100,40]])
print(prediction) #0
print(probability) #0.9,0.09
 

//practical 1c
#prac 1c Decision tree algorithm

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

x=np.array([[25,0],[30,1],[45,2],[35,1],[50,0]])
y=np.array([0,1,1,1,0])

model=DecisionTreeClassifier(max_depth=3,random_state=42) 
model.fit(x,y)

prediction=model.predict([[40,2]])
print("prediction: ",prediction) #1

#visualisation

plt.figure(figsize=(10,6)) #paper size
tree.plot_tree(model,feature_names=["age","income"],class_names=["no","yes"],filled=True)
plt.show()


//practical 1d
#prac 1d KNN

import numpy as np
from sklearn.neighbors  import KNeighborsClassifier

x=np.array([[2,5],[4,6],[5,7],[1,4],[3,5],[6,8] ])
y=np.array([0,1,1,0,0,1])

model=KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)

prediction= model.predict([[4,5]])
probability= model.predict_proba([[4,5]])


print(prediction) 
print(probability) 
#prac 1d KNN

import numpy as np
from sklearn.neighbors  import KNeighborsClassifier

x=np.array([[2,5],[4,6],[5,7],[1,4],[3,5],[6,8] ])
y=np.array([0,1,1,0,0,1])

model=KNeighborsClassifier(n_neighbors=3)
model.fit(x,y)

prediction= model.predict([[4,5]])
probability= model.predict_proba([[4,5]])


print(prediction) 
print(probability) 


//practical 2
import  numpy as np  
from sklearn.linear_model import LogisticRegression
x=np.array([[2],[4],[6],[8],[10]])
y = np.array([0, 0, 1, 1, 1])
model=LogisticRegression()
model.fit(x,y)
prediction=model.predict([[5]])
prop=model.predict_proba([[5]])
print(prediction)
print(prop)


//practical 3
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
 #step 2 import variables
x=np.array([[15,39],[16,81],[17,6],[18,7],[19,40],[20,80]])
kmeans=KMeans(n_clusters=3)
kmeans.fit(x)
labels=kmeans.labels_
print(labels)
plt.scatter(x[:,0], x[:,1], c=labels, cmap='rainbow')
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    color='blue', marker='X', s=200, label='centroid'
)
plt.xlabel("Annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()


//practical 4
import cv2
image=cv2.imread("a1.jpeg")
cv2.imshow("first",image)

gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("second",gray)

crop=image[50:200,100:200]
cv2.imshow("crop",crop)

(h,w)=image.shape[:2]
center=(w/2,h/2)
M=cv2.getRotationMatrix2D(center,270,1.0)
rotated=cv2.warpAffine(image,M,(w,h))
cv2.imshow("rotated",rotated)

resize=cv2.resize(image,(100,100))
cv2.imshow("resize",resize)

blur=cv2.GaussianBlur(image,(8,8),0)
cv2.imshow("blur",blur)

cv2.waitKey(0)
cv2.destroyAllWindows()


//practical 5
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
model = YOLO("yolov8n.pt")
img_path = r"C:\Users\Lenovo\OneDrive\Pictures\Saved Pictures\txt.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = model(img_rgb)
annotated_img = results[0].plot()
cv2.imshow("detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()


//practical 6
import cv2
import numpy as np

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

img = cv2.imread("C:\\Users\\Lenovo\\SEM 6\\AICV\\a1.jpeg")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=7,
    minSize=(30, 30)
)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detected faces", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


//practical 7
# detect and match image using ORB
import cv2
import numpy as np

img1 = cv2.imread("a1.jpeg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("a2.jpeg", cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

result = cv2.drawMatches(img1, kp1,img2, kp2,matches[:5],None,flags=2)

cv2.imshow("ORB Matches", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

//practical 8
//Some Yolopt file and stuff