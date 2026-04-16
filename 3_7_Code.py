# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:42:22 2026

@author: willi
"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io



plt.rcParams['figure.figsize'] = [7, 7]
plt.rcParams.update({'font.size': 18})




folder = r"C:\Users\willi\Downloads\my_faces"

face_list = []

for filename in os.listdir(folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(folder, filename)
        
        img = Image.open(path).convert("L")
        img = img.resize((192, 168))
        
        img_arr = np.array(img, dtype=np.float64)
        
        img_arr = np.array(img, dtype=np.float64)
        
        img_vec = img_arr.T.reshape(-1, 1, order='F')
        face_list.append(img_vec)

faces = np.hstack(face_list)





nfaces = faces.shape[1]

print(faces.shape[1])

## Function Definitions

def shrink(X,tau):
    Y = np.abs(X)-tau
    return np.sign(X) * np.maximum(Y,np.zeros_like(Y))
def SVT(X,tau):
    U,S,VT = np.linalg.svd(X,full_matrices=0)
    out = U @ np.diag(shrink(S,tau)) @ VT
    return out
def RPCA(X):
    n1,n2 = X.shape
    mu = n1*n2/(4*np.sum(np.abs(X.reshape(-1))))
    lambd = 1/np.sqrt(np.maximum(n1,n2))
    thresh = 10**(-7) * np.linalg.norm(X)
    
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    L = np.zeros_like(X)
    count = 0
    while (np.linalg.norm(X-L-S) > thresh) and (count < 500):
        L = SVT(X-S+(1/mu)*Y,1/mu)
        S = shrink(X-L+(1/mu)*Y,lambd/mu)
        Y = Y + mu*(X-L-S)
        count += 1
        if count % 10 == 0:
            err = np.linalg.norm(X - L - S)
            print(f"Iter {count}, error = {err:.4f}")
    return L,S

X = faces


L,S = RPCA(X)

k = 0

fig, axs = plt.subplots(2, 2)
axs = axs.reshape(-1)

imgX = np.reshape(X[:, k-1], (192, 168), order='F')
imgL = np.reshape(L[:, k-1], (192, 168), order='F')
imgS = np.reshape(S[:, k-1], (192, 168), order='F')

# 🔥 ROTATE ONLY HERE (try k=1 or k=3)
imgX = np.rot90(imgX, k=-1)
imgL = np.rot90(imgL, k=-1)
imgS = np.rot90(imgS, k=-1)

axs[0].imshow(imgX, cmap='gray')
axs[0].set_title('X')

axs[1].imshow(imgL, cmap='gray')
axs[1].set_title('L')

axs[2].imshow(imgS, cmap='gray')
axs[2].set_title('S')

for ax in axs:
    ax.axis('off')

plt.tight_layout()
plt.show()
