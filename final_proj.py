#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from utils import *
import os
from scipy import signal
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz
from numpy import *


# In[33]:





# ## Part 1 Toy Problem (20 pts)

# In[34]:





# In[35]:





# In[36]:





# ## Preparation

# In[2]:


# Feel free to change image
background_img = cv2.cvtColor(cv2.imread('painting.JPG'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
plt.figure()
plt.imshow(background_img)


# In[3]:


# Feel free to change image
object_img = cv2.cvtColor(cv2.imread('painting.JPG'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
mask_coords = specify_mask(object_img)


# In[307]:


xs = mask_coords[0]
ys = mask_coords[1]
xs = [int(i) for i in xs]
ys = [int(i) for i in ys]

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
print(xs)
print(ys)
im_src = cv2.imread("painting.JPG")
# focal=1.4*(max(im_src.shape[0],im_src.shape[1]))

#constants
focal=1000
camera_height=1
camera_pos=(0,1,0)
def saveBackWallImg(img,xs,ys):
    back_wall_image=np.copy(im_src[ys[0]:ys[2],xs[0]:xs[2]])
    back_wall_image=cv2.resize(back_wall_image, (256,256), interpolation = cv2.INTER_AREA)
    cv2.imwrite('backwall.jpeg',back_wall_image)
    return

def getBoxHeight(ys):
    return (ys[2]-ys[0])/float(ys[2]-ys[1])

#https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
def perp(a) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2):
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1


####################################################
####################################################
##### P0 ################################# P1 ######
####################################################
####################################################
################## P2 ############# P3 #############
####################################################
####################################################
####################################################
####################################################
####################################################
################## P4 ############# P5 #############
####################################################
####################################################
##### P6 ################################# P7 ######
def getBoundingBox3DCoords(xs,ys,pxl_to_d_ratio,box_height,focal):
    P0=((xs[0]-xs[1])*pxl_to_d_ratio,box_height,0)
    P1=((xs[2]-xs[1])*pxl_to_d_ratio,box_height,0)
    P2=((xs[0]-xs[1])*pxl_to_d_ratio,box_height,-focal/float(ys[2]-ys[1]))
    P3=((xs[2]-xs[1])*pxl_to_d_ratio,box_height,-focal/float(ys[2]-ys[1]))
    P4=((xs[0]-xs[1])*pxl_to_d_ratio,0,-focal/float(ys[2]-ys[1]))
    P5=((xs[2]-xs[1])*pxl_to_d_ratio,0,-focal/float(ys[2]-ys[1]))
    P6=((xs[0]-xs[1])*pxl_to_d_ratio,0,0)
    P7=((xs[2]-xs[1])*pxl_to_d_ratio,0,0)
    return [P0,P1,P2,P3,P4,P5,P6,P7]

def getHomographyOfRightWall(img,xs,ys):
    w=img.shape[1]
    h=img.shape[0]
    
    #two points that determine P3 --> P1 line
    P0 = np.array([xs[1],abs(h-ys[1])])
    P1 = np.array([xs[2],abs(h-ys[0])])

    #two points that determine P7 --> P1 line
    P2 = np.array([w,0])
    P3 = np.array([w,h])
    
    #pixel coordinate of P3, P5
    P00=np.array([ys[0],xs[2]])
    P10=np.array([ys[2],xs[2]])
    
    intersectionP=seg_intersect(P0,P1,P2,P3)
    intersectionP[1]=h-intersectionP[1]
    #pixel coordinate of P1
    P01=np.array([intersectionP[1],intersectionP[0]])
    
    #two points that determine P5 --> P7 line
    P0 = np.array([xs[1],abs(h-ys[1])])
    P1 = np.array([xs[2],abs(h-ys[2])])

    intersectionP=seg_intersect(P0,P1,P2,P3)
    intersectionP[1]=h-intersectionP[1]
    #pixel coordinate of P7
    P11=np.array([intersectionP[1],intersectionP[0]])
    
#     print(P00)
#     print(P01)
#     print(P10)
#     print(P11)
    
    uvCoords=np.array([[0.,0.],[256.,0.],[0.,256.],[256.,256.]])
    picCoords=np.array([[P00[1],P00[0]],[P01[1],P01[0]],[P10[1],P10[0]],[P11[1],P11[0]]])
    H,status=cv2.findHomography(picCoords,uvCoords)
    return H

def getHomographyOfLeftWall(img,xs,ys):
    w=img.shape[1]
    h=img.shape[0]
    
    #two points that determine P2 --> P0 line
    P0 = np.array([xs[1],abs(h-ys[1])])
    P1 = np.array([xs[0],abs(h-ys[0])])

    #two points that determine P6 --> P0 line
    P2 = np.array([0,0])
    P3 = np.array([0,h])
    
    #pixel coordinate of P2, P4
    P01=np.array([ys[0],xs[0]])
    P11=np.array([ys[2],xs[0]])
    
    intersectionP=seg_intersect(P0,P1,P2,P3)
    intersectionP[1]=h-intersectionP[1]
    #pixel coordinate of P1
    P00=np.array([intersectionP[1],intersectionP[0]])
    
    #two points that determine P4 --> P6 line
    P0 = np.array([xs[1],abs(h-ys[1])])
    P1 = np.array([xs[0],abs(h-ys[2])])

    intersectionP=seg_intersect(P0,P1,P2,P3)
    intersectionP[1]=h-intersectionP[1]
    #pixel coordinate of P6
    P10=np.array([intersectionP[1],intersectionP[0]])
    
    print(P00)
    print(P01)
    print(P10)
    print(P11)
    
    uvCoords=np.array([[0.,0.],[256.,0.],[0.,256.],[256.,256.]])
    picCoords=np.array([[P00[1],P00[0]],[P01[1],P01[0]],[P10[1],P10[0]],[P11[1],P11[0]]])
    H,status=cv2.findHomography(picCoords,uvCoords)
    return H
    
box_height=getBoxHeight(ys)
pxl_to_d_ratio=float(box_height)/(ys[2]-ys[0])

"""
P0 --> bottom left
P1 --> bottom right
P2 --> top left
P3 --> top right

nV --> normal vector
"""
def saveObjPlane(P0,P1,P2,P3,planeName):
    nV=["0.000000","0.000000","0.000000"]
    if(planeName=="backwall"):
        nV[2]="1.000000"
    elif(planeName=="rightwall"):
        nV[0]="-1.000000"
    elif(planeName=="leftwall"):
        nV[0]="1.000000"
    elif(planeName=="bottomwall"):
        nV[1]="1.000000"
    elif(planeName=="topwall"):
        nV[1]="-1.000000"
    
    vertexStrings=[' '.join(["mtllib",planeName+".mtl\no Plane"]),
                   ' '.join(["v",str(P0[0]),str(P0[1]),str(P0[2])]),
                   ' '.join(["v",str(P1[0]),str(P1[1]),str(P1[2])]),
                   ' '.join(["v",str(P2[0]),str(P2[1]),str(P2[2])]),
                   ' '.join(["v",str(P3[0]),str(P3[1]),str(P3[2])]),
                   ' '.join(["vt","0.000000","0.000000"]),
                   ' '.join(["vt","1.000000","0.000000"]),
                   ' '.join(["vt","0.000000","1.000000"]),
                   ' '.join(["vt","1.000000","1.000000"]),
                   ' '.join(["vn",nV[0],nV[1],nV[2]]),
                   ' '.join(["usemtl",planeName+"Material"]),
                   ' '.join(["s","off"]),
                   ' '.join(["f","1/1/1","2/2/1","4/4/1","3/3/1"]),
                  ]
    vertexStrings='\n'.join(vertexStrings)
    mltStrings=[' '.join(["newmtl",planeName+"Material"]),
                   ' '.join(["map_Kd",planeName+".jpeg"]),
                   ' '.join(["map_Ka",planeName+".jpeg"]),
                  ]
    mtlStrings='\n'.join(mltStrings)
    file = open(planeName+'.obj', 'w')
    file.write(vertexStrings)
    file.close()
    file = open(planeName+'.mtl', 'w')
    file.write(mtlStrings)
    file.close()
    
coords3D=getBoundingBox3DCoords(xs,ys,pxl_to_d_ratio,box_height,focal)
saveObjPlane(coords3D[4],coords3D[5],coords3D[2],coords3D[3],'backwall')
saveObjPlane(coords3D[6],coords3D[4],coords3D[0],coords3D[2],'leftwall')
saveObjPlane(coords3D[5],coords3D[7],coords3D[3],coords3D[1],'rightwall')
saveObjPlane(coords3D[6],coords3D[7],coords3D[4],coords3D[5],'bottomwall')
saveObjPlane(coords3D[2],coords3D[3],coords3D[0],coords3D[1],'topwall')

saveBackWallImg(im_src,xs,ys)

im_src = cv2.imread("painting.JPG")
H=getHomographyOfRightWall(im_src,xs,ys)
im_dst = cv2.cvtColor(cv2.warpPerspective(im_src,H, (256,256)),cv2.COLOR_BGR2RGB)
cv2.imwrite('rightwall.jpeg',cv2.cvtColor(im_dst,cv2.COLOR_RGB2BGR))
plt.figure()
plt.imshow(im_dst)

print("****************")
im_src = cv2.imread("painting.JPG")
H=getHomographyOfLeftWall(im_src,xs,ys)
print(H)
H_t=np.array([[373.],[723.],[1.]])
a=H.dot(H_t)
a=a[:]/a[2]
print(a)
im_dst = cv2.cvtColor(cv2.warpPerspective(im_src,H, (256,256)),cv2.COLOR_BGR2RGB)
cv2.imwrite('leftwall.jpeg',cv2.cvtColor(im_dst,cv2.COLOR_RGB2BGR))
plt.figure()
plt.imshow(im_dst)


# In[200]:



# A=((xs[0]-xs[1])*pxl_to_d_ratio,0,-focal/float(ys[2]-ys[1]))
# B=((xs[2]-xs[1])*pxl_to_d_ratio,0,-focal/float(ys[2]-ys[1]))
# C=((xs[0]-xs[1])*pxl_to_d_ratio,box_height,-focal/float(ys[2]-ys[1]))
# D=((xs[2]-xs[1])*pxl_to_d_ratio,box_height,-focal/float(ys[2]-ys[1]))



    


# print(A)
# print(B)
# print(C)
# print(D)

# #bottom plane coordinate
# BA=((xs[0]-xs[1])*pxl_to_d_ratio,0,0)
# BB=((xs[2]-xs[1])*pxl_to_d_ratio,0,0)
# BC=((xs[0]-xs[1])*pxl_to_d_ratio,0,-focal/float(ys[2]-ys[1]))
# BD=((xs[2]-xs[1])*pxl_to_d_ratio,0,-focal/float(ys[2]-ys[1]))

# print(BA)
# print(BB)
# print(BC)
# print(BD)

# #bottom plane 2D coord
# BA2=(0,0)



# print(object_img.shape)
w=object_img.shape[1]
h=object_img.shape[0]

p1 = np.array([xs[1],abs(h-ys[1])])
p2 = np.array([xs[2],abs(h-ys[0])])

p3 = np.array([w,0])
p4 = np.array([w,h])


P00=np.array([ys[0],xs[2]])
P10=np.array([ys[2],xs[2]])



intersectionP=seg_intersect( p1,p2, p3,p4)
# print(intersectionP)
intersectionP[1]=h-intersectionP[1]
P01=np.array([intersectionP[1],intersectionP[0]])

p1 = np.array([xs[1],abs(h-ys[1])])
p2 = np.array([xs[2],abs(h-ys[2])])

intersectionP=seg_intersect( p1,p2, p3,p4)
# print(intersectionP)
intersectionP[1]=h-intersectionP[1]
P11=np.array([intersectionP[1],intersectionP[0]])

uvCoords=np.array([[0.,0.],[256.,0.],[0.,256.],[256.,256.]])
picCoords=np.array([[P00[1],P00[0]],[P01[1],P01[0]],[P10[1],P10[0]],[P11[1],P11[0]]])
print(uvCoords)
print(picCoords)


# In[271]:


def computeHomography(pts1, pts2,normalization_func=None):
    '''
    Compute homography that maps from pts1 to pts2 using SVD
     
    Input: pts1 and pts2 are 3xN matrices for N points in homogeneous
    coordinates. 
    
    Output: H is a 3x3 matrix, such that pts2~=H*pts1
    '''
    numOfRows=2*len(pts1)
    numOfCols=9
    A=np.zeros((numOfRows,numOfCols))
    
    for i in range(0,len(pts1)):
        (aX,aY) = pts1[i,:]
        (bX,bY) = pts2[i,:]
        
        A[2*i] = np.array([-aX,-aY,-1,0,0,0,bX*aX,bX*aY,bX])
        A[2*i+1] = np.array([0,0,0,-aX,-aY,-1,bY*aX,bY*aY,bY])
    
    U,s,v = np.linalg.svd(A)
    
    H = np.eye(3)
    
    H = np.reshape(v[-1], (3,3))
    H = np.divide(H,H[2,2])
    return H


# In[225]:


# H=computeHomography(picCoords,uvCoords,None)
H,status=cv2.findHomography(picCoords,uvCoords)
# H =computeHomography(picCoords,uvCoords)
# H_t=np.array([[1,0,500],[0,1,100],[0,0,1]])
print(H)
im_src = cv2.imread("painting.JPG")
# new_obj_img=cv2.cvtColor(cv2.imread("painting.JPG"), cv2.COLOR_BGR2RGB) / 255.0
im_dst = cv2.cvtColor(cv2.warpPerspective(im_src,H, (256,256)),cv2.COLOR_BGR2RGB)
plt.figure()
# # blendedOutput= cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.imshow(im_dst)
cv2.imwrite('right_wall.jpeg',cv2.cvtColor(im_dst,cv2.COLOR_RGB2BGR))

H_t=np.array([[1707.],[2399.],[1.]])
a=H.dot(H_t)
print(a)
a=a[:]/a[2]
print(a)
# a[:]/a[2]
# print(new_obj_img)
# print(H)
# H_t=np.array([[1,0,abs(100)],[0,1,abs(100)],[0,0,1]])
# output=cv2.warpPerspective(new_obj_img,H_t.dot(H), (256, 256))
# print(output)



# In[74]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
bottom_center = specify_bottom_center(background_img)


# In[75]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
cropped_object, object_mask = align_source(object_img, mask, background_img, bottom_center)


# ## Part 2 Poisson Blending (50 pts)

# In[53]:


def computeIm2var(mask):
    (H, W) = np.shape(mask)
    im2var = np.zeros((H,W),dtype=int)
    v = 0
    for y  in range(0,H):
        for x in range(1,W):
            if (mask[y, x]==True):
                im2var[y,x] = v
                v = v+1
    print('Finished computing im2var')
    return im2var

def poisson_blend_helper(fg,bkg,mask,yMin,yMax,xMin,xMax):
    """
    The implementation for gradient domain processing is not complicated, but it is easy to make a mistake, so let's start with a toy example. Reconstruct this image from its gradient values, plus one pixel intensity. Denote the intensity of the source image at (x, y) as s(x,y) and the value to solve for as v(x,y). For each pixel, then, we have two objectives:
    1. minimize (v(x+1,y)-v(x,y) - (s(x+1,y)-s(x,y)))^2
    2. minimize (v(x,y+1)-v(x,y) - (s(x,y+1)-s(x,y)))^2
    Note that these could be solved while adding any constant value to v, so we will add one more objective:
    3. minimize (v(1,1)-s(1,1))^2
    """
    im_h, im_w, im_d = bkg[yMin:yMax,xMin:xMax].shape
    print(im_h)
    print(im_w)
    im2var = computeIm2var(mask)
    e=0
    totalE=(4*im_h*im_w)+1
    print('Creating A and b matrix')
    A=lil_matrix((totalE,im_h*im_w), dtype=np.float32)
    b=lil_matrix((totalE,3), dtype=np.float32)
    print('Finished creating A and b matrix')
    print(im2var)
    print(A.shape)
    print(b.shape)
    for y in range(yMin,yMax):
        for x in range(xMin,xMax):
            if(mask[y,x]==True):
                if(mask[y,x+1]==True):   
                    A[e,im2var[y,x]]=1
                    A[e,im2var[y,x+1]] = -1
                    b[e] = fg[y,x] - fg[y,x+1]
                else:
                    A[e,im2var[y,x]] = 1
                    b[e] = bkg[y,x+1]
                e = e + 1;

                if(mask[y+1,x]==True):
                    A[e,im2var[y,x]]=1
                    A[e,im2var[y+1,x]] = -1
                    b[e] = fg[y,x] - fg[y+1,x]
                else:
                    A[e,im2var[y,x]] = 1
                    b[e] = bkg[y+1,x]
                e = e + 1;

                A[e,im2var[y,x]]=1
                if(mask[y-1,x]==True):
                    A[e,im2var[y,x]]=1
                    A[e,im2var[y-1,x]] = -1
                    b[e] = fg[y,x] - fg[y-1,x]
                else:
                    A[e,im2var[y,x]] = 1
                    b[e] = bkg[y-1,x]
                e = e + 1;

                A[e,im2var[y,x]]=1
                if(mask[y,x-1]==True):
                    A[e,im2var[y,x]]=1
                    A[e,im2var[y,x-1]] = -1
                    b[e] = fg[y,x] -fg[y,x-1]
                else:
                    A[e,im2var[y,x]] = 1
                    b[e] = bkg[y,x-1]
                e = e + 1;
    
    res=np.zeros((im_h,im_w,3),dtype=np.float32)
    bMatrix=b.tocsr()
    
    for i in range(0,3):     
        v=linalg.lsqr(A.tocsr(), bMatrix.getcol(i).toarray().reshape((-1,)))
        print(v[0])
        vInd=0
        for y in range(yMin,yMax):
            for x in range(xMin,xMax):
                if(mask[y,x]==True):
                    res[y-yMin,x-xMin,i]=(v[0][vInd])
                    vInd=vInd+1
    return res

def poisson_blend(cropped_object, object_mask, background_img):
    """
    :param cropped_object: numpy.ndarray One you get from align_source
    :param object_mask: numpy.ndarray One you get from align_source
    :param background_img: numpy.ndarray 
    """
    print(cropped_object.shape)
    print(object_mask.shape)
    print(background_img.shape)
    res=np.zeros(background_img.shape)
    
    top_left_mask_coord=None
    bottom_right_mask_coord=None
    
    yMin=object_mask.shape[0]
    xMin=object_mask.shape[1]
    yMax=0
    xMax=0
    
    for y in range(0,object_mask.shape[0]):
        for x in range(0,object_mask.shape[1]):
            if(object_mask[y,x]==True):
                yMax=max(yMax,y)
                xMax=max(xMax,x)
                yMin=min(yMin,y)
                xMin=min(xMin,x)
    yMin=yMin-5
    xMin=xMin-5
    yMax=yMax+5
    xMax=xMax+5
    res=background_img
#     cropped_object[yMin:yMax,xMin:xMax]
#     background_img[yMin:yMax,xMin:xMax]
#     object_mask[yMin:yMax,xMin:xMax]
    poissonRes=poisson_blend_helper(cropped_object,background_img,object_mask,yMin,yMax,xMin,xMax)
    for i in range(0,3):
        res[yMin:yMax,xMin:xMax,i]=np.where(object_mask[yMin:yMax,xMin:xMax]==True,poissonRes[:,:,i],background_img[yMin:yMax,xMin:xMax,i])
    return res


# In[77]:


im_blend = poisson_blend(cropped_object, object_mask, background_img)
if im_blend.any():
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    plt.imshow(im_blend)


# In[78]:


res=np.zeros(im_blend.shape)
res[:,:,0]=im_blend[:,:,2]*255
res[:,:,1]=im_blend[:,:,1]*255
res[:,:,2]=im_blend[:,:,0]*255
cv2.imwrite('pen.jpeg', res) 


# In[47]:


# Feel free to change image
background_img = cv2.cvtColor(cv2.imread('samples/desert.JPG'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
plt.figure()
plt.imshow(background_img)


# In[49]:


# Feel free to change image
object_img = cv2.cvtColor(cv2.imread('samples/butterfly2.jpg'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
mask_coords = specify_mask(object_img)


# In[50]:


xs = mask_coords[0]
ys = mask_coords[1]
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure()
mask = get_mask(ys, xs, object_img)


# In[51]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
bottom_center = specify_bottom_center(background_img)


# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
cropped_object, object_mask = align_source(object_img, mask, background_img, bottom_center)


# In[53]:


im_blend = poisson_blend(cropped_object, object_mask, background_img)
if im_blend.any():
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    plt.imshow(im_blend)


# In[54]:


res=np.zeros(im_blend.shape)
res[:,:,0]=im_blend[:,:,2]*255
res[:,:,1]=im_blend[:,:,1]*255
res[:,:,2]=im_blend[:,:,0]*255
cv2.imwrite('butterfly.jpeg', res) 


# In[87]:


# Feel free to change image
background_img = cv2.cvtColor(cv2.imread('samples/black_sand_beach.jpg'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
plt.figure()
plt.imshow(background_img)


# In[89]:


# Feel free to change image
object_img = cv2.cvtColor(cv2.imread('samples/dc3.png'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
mask_coords = specify_mask(object_img)


# In[90]:


xs = mask_coords[0]
ys = mask_coords[1]
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure()
mask = get_mask(ys, xs, object_img)


# In[91]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
bottom_center = specify_bottom_center(background_img)


# In[92]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
cropped_object, object_mask = align_source(object_img, mask, background_img, bottom_center)


# In[93]:


im_blend = poisson_blend(cropped_object, object_mask, background_img)
if im_blend.any():
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    plt.imshow(im_blend)


# In[94]:


res=np.zeros(im_blend.shape)
res[:,:,0]=im_blend[:,:,2]*255
res[:,:,1]=im_blend[:,:,1]*255
res[:,:,2]=im_blend[:,:,0]*255
cv2.imwrite('batman.jpeg', res) 


# In[26]:


# Feel free to change image
background_img = cv2.cvtColor(cv2.imread('samples/swimming_pool.jpg'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
plt.figure()
plt.imshow(background_img)


# In[28]:


# Feel free to change image
object_img = cv2.cvtColor(cv2.imread('samples/dogbed.jpg'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
mask_coords = specify_mask(object_img)


# In[29]:


xs = mask_coords[0]
ys = mask_coords[1]
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure()
mask = get_mask(ys, xs, object_img)


# In[30]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
bottom_center = specify_bottom_center(background_img)


# In[31]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
cropped_object, object_mask = align_source(object_img, mask, background_img, bottom_center)


# In[32]:


im_blend = poisson_blend(cropped_object, object_mask, background_img)
if im_blend.any():
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    plt.imshow(im_blend)


# In[33]:


res=np.zeros(im_blend.shape)
res[:,:,0]=im_blend[:,:,2]*255
res[:,:,1]=im_blend[:,:,1]*255
res[:,:,2]=im_blend[:,:,0]*255
cv2.imwrite('pool_dog2.jpeg', res) 


# In[96]:


# Feel free to change image
background_img = cv2.cvtColor(cv2.imread('samples/american_flag.jpg'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
plt.figure()
plt.imshow(background_img)


# In[99]:


# Feel free to change image
object_img = cv2.cvtColor(cv2.imread('samples/lincoln.jpg'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
mask_coords = specify_mask(object_img)


# In[100]:


xs = mask_coords[0]
ys = mask_coords[1]
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure()
mask = get_mask(ys, xs, object_img)


# In[101]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
bottom_center = specify_bottom_center(background_img)


# In[104]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
cropped_object, object_mask = align_source(object_img, mask, background_img, bottom_center)
cv2.imwrite('cropped_lincoln_flag.jpeg', background_img) 


# ## Part 3 Mixed Gradients (20 pts)

# In[51]:


def mix_blend_helper(fg,bkg,mask,yMin,yMax,xMin,xMax):
    """
    The implementation for gradient domain processing is not complicated, but it is easy to make a mistake, so let's start with a toy example. Reconstruct this image from its gradient values, plus one pixel intensity. Denote the intensity of the source image at (x, y) as s(x,y) and the value to solve for as v(x,y). For each pixel, then, we have two objectives:
    1. minimize (v(x+1,y)-v(x,y) - (s(x+1,y)-s(x,y)))^2
    2. minimize (v(x,y+1)-v(x,y) - (s(x,y+1)-s(x,y)))^2
    Note that these could be solved while adding any constant value to v, so we will add one more objective:
    3. minimize (v(1,1)-s(1,1))^2
    """
    im_h, im_w, im_d = bkg[yMin:yMax,xMin:xMax].shape
    print(im_h)
    print(im_w)
    im2var = computeIm2var(mask)
    e=0
    totalE=(4*im_h*im_w)+1
    print('Creating A and b matrix')
    A=lil_matrix((totalE,im_h*im_w), dtype=np.float32)
    b=lil_matrix((totalE,3), dtype=np.float32)
    print('Finished creating A and b matrix')
    print(im2var)
    print(A.shape)
    print(b.shape)
    for y in range(yMin,yMax):
        for x in range(xMin,xMax):
            if(mask[y,x]==True):
                if(mask[y,x+1]==True):   
                    A[e,im2var[y,x]]=1
                    A[e,im2var[y,x+1]] = -1
                    fgGradient=fg[y,x] - fg[y,x+1]
                    bkgGradient=bkg[y,x] - bkg[y,x+1]
                    for i in range(0,3):
                        if abs(fgGradient[i])>abs(bkgGradient[i]):
                            b[e,i]=fgGradient[i]
                        else:
                            b[e,i]=bkgGradient[i]
                else:
                    A[e,im2var[y,x]] = 1
                    b[e] = bkg[y,x+1]
                e = e + 1;

                if(mask[y+1,x]==True):
                    A[e,im2var[y,x]]=1
                    A[e,im2var[y+1,x]] = -1
                    fgGradient=fg[y,x] - fg[y+1,x]
                    bkgGradient=bkg[y,x] - bkg[y+1,x]
                    for i in range(0,3):
                        if abs(fgGradient[i])>abs(bkgGradient[i]):
                            b[e,i]=fgGradient[i]
                        else:
                            b[e,i]=bkgGradient[i]
                else:
                    A[e,im2var[y,x]] = 1
                    b[e] = bkg[y+1,x]
                e = e + 1;

                A[e,im2var[y,x]]=1
                if(mask[y-1,x]==True):
                    A[e,im2var[y,x]]=1
                    A[e,im2var[y-1,x]] = -1
                    fgGradient=fg[y,x] - fg[y-1,x]
                    bkgGradient=bkg[y,x] - bkg[y-1,x]
                    for i in range(0,3):
                        if abs(fgGradient[i])>abs(bkgGradient[i]):
                            b[e,i]=fgGradient[i]
                        else:
                            b[e,i]=bkgGradient[i]
                else:
                    A[e,im2var[y,x]] = 1
                    b[e] = bkg[y-1,x]
                e = e + 1;

                A[e,im2var[y,x]]=1
                if(mask[y,x-1]==True):
                    A[e,im2var[y,x]]=1
                    A[e,im2var[y,x-1]] = -1
                    fgGradient=fg[y,x] - fg[y-1,x]
                    bkgGradient=bkg[y,x] - bkg[y-1,x]
                    for i in range(0,3):
                        if abs(fgGradient[i])>abs(bkgGradient[i]):
                            b[e,i]=fgGradient[i]
                        else:
                            b[e,i]=bkgGradient[i]
                else:
                    A[e,im2var[y,x]] = 1
                    b[e] = bkg[y,x-1]
                e = e + 1;
    
#     A[e,im2var[0,0]] = 1
#     b[e] = bkg[0,0]
    
    res=np.zeros((im_h,im_w,3),dtype=np.float32)
    bMatrix=b.tocsr()
    
    for i in range(0,3):     
        v=linalg.lsqr(A.tocsr(), bMatrix.getcol(i).toarray().reshape((-1,)))
        print(v[0])
        vInd=0
        for y in range(yMin,yMax):
            for x in range(xMin,xMax):
                if(mask[y,x]==True):
#                     print(v[0][vInd])
                    res[y-yMin,x-xMin,i]=(v[0][vInd])
                    vInd=vInd+1
#     plt.figure()
#     plt.imshow(res)
    return res

def mix_blend(cropped_object, object_mask, background_img):
    """
    :param cropped_object: numpy.ndarray One you get from align_source
    :param object_mask: numpy.ndarray One you get from align_source
    :param background_img: numpy.ndarray 
    """
    print(cropped_object.shape)
    print(object_mask.shape)
    print(background_img.shape)
    res=np.zeros(background_img.shape)
    
    top_left_mask_coord=None
    bottom_right_mask_coord=None
    
    yMin=object_mask.shape[0]
    xMin=object_mask.shape[1]
    yMax=0
    xMax=0
    
    for y in range(0,object_mask.shape[0]):
        for x in range(0,object_mask.shape[1]):
            if(object_mask[y,x]==True):
                yMax=max(yMax,y)
                xMax=max(xMax,x)
                yMin=min(yMin,y)
                xMin=min(xMin,x)
    yMin=yMin-5
    xMin=xMin-5
    yMax=yMax+5
    xMax=xMax+5
    res=background_img
#     cropped_object[yMin:yMax,xMin:xMax]
#     background_img[yMin:yMax,xMin:xMax]
#     object_mask[yMin:yMax,xMin:xMax]
    poissonRes=mix_blend_helper(cropped_object,background_img,object_mask,yMin,yMax,xMin,xMax)
    for i in range(0,3):
        res[yMin:yMax,xMin:xMax,i]=np.where(object_mask[yMin:yMax,xMin:xMax]==True,poissonRes[:,:,i],background_img[yMin:yMax,xMin:xMax,i])
    return res
 


# In[106]:


im_mix = mix_blend(cropped_object, object_mask, background_img)
if im_mix.any():
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    plt.imshow(im_mix)


# In[84]:


res=np.zeros(im_mix.shape)
res[:,:,0]=im_mix[:,:,2]*255
res[:,:,1]=im_mix[:,:,1]*255
res[:,:,2]=im_mix[:,:,0]*255
cv2.imwrite('flag.jpeg', res) 


# In[59]:


# Feel free to change image
background_img = cv2.cvtColor(cv2.imread('samples/chocolate_bar.png'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
plt.figure()
plt.imshow(background_img)


# In[56]:


# Feel free to change image
object_img = cv2.cvtColor(cv2.imread('samples/writing.jpg'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
mask_coords = specify_mask(object_img)


# In[57]:


xs = mask_coords[0]
ys = mask_coords[1]
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure()
mask = get_mask(ys, xs, object_img)


# In[60]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
bottom_center = specify_bottom_center(background_img)


# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
cropped_object, object_mask = align_source(object_img, mask, background_img, bottom_center)
cv2.imwrite('cropped_lincoln_flag.jpeg', background_img) 


# In[62]:


im_mix = mix_blend(cropped_object, object_mask, background_img)
if im_mix.any():
    get_ipython().run_line_magic('matplotlib', 'inline')
    import matplotlib.pyplot as plt
    plt.imshow(im_mix)


# In[63]:


res=np.zeros(im_mix.shape)
res[:,:,0]=im_mix[:,:,2]*255
res[:,:,1]=im_mix[:,:,1]*255
res[:,:,2]=im_mix[:,:,0]*255
cv2.imwrite('chocolate_bar_f.jpeg', res) 


# # Bells & Whistles (Extra Points)

# ## Color2Gray (20 pts)

# In[2]:


def color2gray(img):
    """
    The implementation for gradient domain processing is not complicated, but it is easy to make a mistake, so let's start with a toy example. Reconstruct this image from its gradient values, plus one pixel intensity. Denote the intensity of the source image at (x, y) as s(x,y) and the value to solve for as v(x,y). For each pixel, then, we have two objectives:
    1. minimize (v(x+1,y)-v(x,y) - (s(x+1,y)-s(x,y)))^2
    2. minimize (v(x,y+1)-v(x,y) - (s(x,y+1)-s(x,y)))^2
    Note that these could be solved while adding any constant value to v, so we will add one more objective:
    3. minimize (v(1,1)-s(1,1))^2
    
    :param toy_img: numpy.ndarray
    """
    
    imgGrey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img=img/255.0
    imgGrey=imgGrey/255.0
    bkg=img
    im_h, im_w = imgGrey.shape
    im2var = np.arange(im_h * im_w).reshape(im_h, im_w)
    e=0
    totalE=(4*im_h*im_w)+1
    A=lil_matrix((totalE,im_h*im_w), dtype=np.float32)
    b=lil_matrix((totalE,1), dtype=np.float32)
    print(np.shape(A))
    for y in range(0,im_h):
        for x in range(0,im_w):
            if(x!=im_w-1):               
                A[e,im2var[y,x+1]] = -1
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x] - bkg[y,x+1])
                b[e]=(bkg[y,x] - bkg[y,x+1])[np.where(vals == np.amax(vals))[0][0]]
            else:
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x])
                b[e]=(bkg[y,x])[np.where(vals == np.amax(vals))[0][0]]
            e = e + 1;

            if(y!=im_h-1):               
                A[e,im2var[y+1,x]] = -1
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x] - bkg[y+1,x])
                b[e]=(bkg[y,x] - bkg[y+1,x])[np.where(vals == np.amax(vals))[0][0]]
            else:
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x])
                b[e]=(bkg[y,x])[np.where(vals == np.amax(vals))[0][0]]
            e = e + 1;
            
            if(x!=0):               
                A[e,im2var[y,x-1]] = -1
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x] - bkg[y,x-1])
                b[e]=(bkg[y,x] - bkg[y,x-1])[np.where(vals == np.amax(vals))[0][0]]
            else:
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x])
                b[e]=(bkg[y,x])[np.where(vals == np.amax(vals))[0][0]]
            e = e + 1;

            if(y!=0):               
                A[e,im2var[y-1,x]] = -1
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x] - bkg[y-1,x])
                b[e]=(bkg[y,x] - bkg[y-1,x])[np.where(vals == np.amax(vals))[0][0]]
            else:
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x])
                b[e]=(bkg[y,x])[np.where(vals == np.amax(vals))[0][0]]
            e = e + 1;

    A[e][im2var[0][0]] = 1
    vals=abs(bkg[0,0])
    b[e]=(bkg[0,0])[np.where(vals == np.amax(vals))[0][0]]

    v=linalg.lsqr(A.tocsr(), b.tocsr().toarray().reshape((-1,)))
    return np.reshape(v[0], (im_h,im_w))


# In[3]:


color_blind4 = cv2.cvtColor(cv2.imread('samples/colorBlind4.png'), cv2.COLOR_BGR2RGB)
# color_blind4 = cv2.cvtColor(color_blind4, cv2.COLOR_BGR2GRAY).astype('double') / 255.0
res=color2gray(color_blind4)
plt.imshow(res,cmap='gray')


# In[8]:


res1=np.zeros(res.shape)
res1=res*255
cv2.imwrite('4gray.jpeg', res1) 

cv2.imwrite('4originalgrey.jpeg',cv2.cvtColor(color_blind4, cv2.COLOR_BGR2GRAY).astype('double'))


# In[12]:


color_blind8 = cv2.cvtColor(cv2.imread('samples/colorBlind8.png'), cv2.COLOR_BGR2RGB)
new_color_blind8=np.copy(color_blind8)
# color_blind4 = cv2.cvtColor(color_blind4, cv2.COLOR_BGR2GRAY).astype('double') / 255.0
res=color2gray(color_blind8)
plt.imshow(res, cmap="gray")


# In[13]:


res1=np.zeros(res.shape)
res1=res*255
cv2.imwrite('8gray.jpeg', res1) 

cv2.imwrite('8originalgrey.jpeg',cv2.cvtColor(new_color_blind8, cv2.COLOR_BGR2GRAY).astype('double'))


# ## Laplacian pyramid blending (20 pts)

# In[178]:


def gaussian_kernel(sigma, kernel_half_size):
    '''
    Inputs:
        sigma = standard deviation for the gaussian kernel
        kernel_half_size = recommended to be at least 3*sigma
    
    Output:
        Returns a 2D Gaussian kernel matrix
    '''
    window_size = kernel_half_size*2+1
    gaussian_kernel_1d = signal.gaussian(window_size, std=sigma).reshape(window_size, 1)
    gaussian_kernel_2d = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)
    gaussian_kernel_2d /= np.sum(gaussian_kernel_2d) # make sure it sums to one

    return gaussian_kernel_2d

def applyLowPassFilter(img, sigma, kernal_half_size):
    kernal = gaussian_kernel(sigma, kernal_half_size)
    return cv2.filter2D(img,-1,kernal)

def getGaussianPyramid(img,lvls):
    (H,W) = np.shape(img)
    copy_img=img.copy()
    blur_img=applyLowPassFilter(copy_img,100,25)
    res=[]
    for i in range(0,lvls):
        res.append(blur_img)
        blur_img=blur_img.copy()
        blur_img=cv2.pyrDown(blur_img)
    return res

def getIdentityPyramid(img,lvls):
    (H,W) = np.shape(img)
    copy_img=img.copy()
    res=[]
    for i in range(0,lvls):
        res.append(copy_img)
        copy_img=copy_img.copy()
        copy_img=cv2.pyrDown(copy_img)
    return res

def laplacian_blend(img1, img2, mask):
    res=np.zeros(np.shape(img1),dtype=float)
    lvls=3
    for c in range(0,3):
        gImg1Pyramids=getGaussianPyramid(img1[:,:,c],lvls)
        gImg2Pyramids=getGaussianPyramid(img2[:,:,c],lvls)
        iImg1Pyramids=getIdentityPyramid(img1[:,:,c],lvls)
        iImg2Pyramids=getIdentityPyramid(img2[:,:,c],lvls)
        maskImgPyramids=getGaussianPyramid(mask,lvls)
        lImg1Pyramids=[]
        lImg2Pyramids=[]
        
        for i in range(0,lvls):
            lImg1Pyramids.append(iImg1Pyramids[i]-gImg1Pyramids[i])
            
        for i in range(0,lvls):
            lImg2Pyramids.append(iImg2Pyramids[i]-gImg2Pyramids[i])
            
        for i in range(0,lvls):
            if((i%2)==0):
                if(lImg1Pyramids[i].shape[0]!=img1.shape[0]) or (lImg1Pyramids[i].shape[1]!=img1.shape[1]):
                    for x in range(0,i-1):
                        lImg1Pyramids[i]=cv2.pyrUp(iImg1Pyramids[i],dstsize =(lImg1Pyramids[i].shape[1]*(2), lImg1Pyramids[i].shape[0]*(2)))
                        lImg2Pyramids[i]=cv2.pyrUp(iImg2Pyramids[i],dstsize =(lImg2Pyramids[i].shape[1]*(2), lImg2Pyramids[i].shape[0]*(2)))
                        maskImgPyramids[i]=cv2.pyrUp(maskImgPyramids[i],dstsize =(maskImgPyramids[i].shape[1]*(2), maskImgPyramids[i].shape[0]*(2)))
                    lImg1Pyramids[i]=lImg1Pyramids[i][0:int(img1.shape[0]/2),0:int(img1.shape[1]/2)]
                    lImg2Pyramids[i]=lImg2Pyramids[i][0:int(img1.shape[0]/2),0:int(img1.shape[1]/2)]
                    maskImgPyramids[i]=maskImgPyramids[i][0:int(img1.shape[0]/2),0:int(img1.shape[1]/2)]
                    lImg1Pyramids[i]=cv2.pyrUp(lImg1Pyramids[i],dstsize =(img1.shape[1], img1.shape[0]))
                    lImg2Pyramids[i]=cv2.pyrUp(lImg2Pyramids[i],dstsize =(img1.shape[1], img1.shape[0]))
                    maskImgPyramids[i]=cv2.pyrUp(maskImgPyramids[i],dstsize =(img1.shape[1], img1.shape[0]))
                res[:,:,c]=lImg1Pyramids[i]*maskImgPyramids[i]+lImg2Pyramids[i]*(1-maskImgPyramids[i])
            else:
                if(gImg1Pyramids[i].shape[0]!=img1.shape[0]) or (gImg1Pyramids[i].shape[1]!=img1.shape[1]):
                    for x in range(0,i-1):
                        gImg1Pyramids[i]=cv2.pyrUp(gImg1Pyramids[i],dstsize =(gImg1Pyramids[i].shape[1]*(2), gImg1Pyramids[i].shape[0]*(2)))
                        gImg2Pyramids[i]=cv2.pyrUp(iImg2Pyramids[i],dstsize =(gImg2Pyramids[i].shape[1]*(2), gImg2Pyramids[i].shape[0]*(2)))
                        maskImgPyramids[i]=cv2.pyrUp(maskImgPyramids[i],dstsize =(maskImgPyramids[i].shape[1]*(2), maskImgPyramids[i].shape[0]*(2)))
                    gImg1Pyramids[i]=gImg1Pyramids[i][0:int(img1.shape[0]/2),0:int(img1.shape[1]/2)]
                    gImg2Pyramids[i]=gImg2Pyramids[i][0:int(img1.shape[0]/2),0:int(img1.shape[1]/2)]
                    maskImgPyramids[i]=maskImgPyramids[i][0:int(img1.shape[0]/2),0:int(img1.shape[1]/2)]
                    gImg1Pyramids[i]=cv2.pyrUp(gImg1Pyramids[i],dstsize =(img1.shape[1], img1.shape[0]))
                    gImg2Pyramids[i]=cv2.pyrUp(gImg2Pyramids[i],dstsize =(img1.shape[1], img1.shape[0]))
                    maskImgPyramids[i]=cv2.pyrUp(maskImgPyramids[i],dstsize =(img1.shape[1], img1.shape[0]))
                res[:,:,c]=gImg1Pyramids[i]*maskImgPyramids[i]+gImg2Pyramids[i]*(1-maskImgPyramids[i])
    return res


# In[190]:


apple = cv2.cvtColor(cv2.imread('samples/apple.JPG'), cv2.COLOR_BGR2RGB)/255.0
orange = cv2.cvtColor(cv2.imread('samples/orange.JPG'), cv2.COLOR_BGR2RGB)/255.0
mask = cv2.cvtColor(cv2.imread('samples/mask.png'), cv2.COLOR_BGR2RGB)
mask= cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype('double') / 255.0
apple = apple[0:484,0:474,:]
orange = orange[0:484,0:474,:]
mask = mask[0:484,0:474]
result = laplacian_blend(apple,orange,mask)
plt.imshow(result)


# In[191]:


res=np.zeros(result.shape)
res[:,:,0]=result[:,:,2]*255
res[:,:,1]=result[:,:,1]*255
res[:,:,2]=result[:,:,0]*255
cv2.imwrite('apple_orange_blend.jpeg', res) 


# In[192]:


apple = cv2.cvtColor(cv2.imread('samples/pumpkin.jpg'), cv2.COLOR_BGR2RGB)/255.0
orange = cv2.cvtColor(cv2.imread('samples/watermellon.png'), cv2.COLOR_BGR2RGB)/255.0
mask = cv2.cvtColor(cv2.imread('samples/pumpkin_watermellon_mask.png'), cv2.COLOR_BGR2RGB)
mask= cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY).astype('double') / 255.0
# apple = apple[0:484,0:474,:]
# orange = orange[0:484,0:474,:]
# mask = mask[0:484,0:474]
result = laplacian_blend(apple,orange,mask)
plt.imshow(result)


# In[193]:


res=np.zeros(result.shape)
res[:,:,0]=result[:,:,2]*255
res[:,:,1]=result[:,:,1]*255
res[:,:,2]=result[:,:,0]*255
cv2.imwrite('waterkin_blend.jpeg', res) 


# ## More gradient domain processing (up to 20 pts)

# In[15]:


def flatenning(img,imgedge):
    """
    The implementation for gradient domain processing is not complicated, but it is easy to make a mistake, so let's start with a toy example. Reconstruct this image from its gradient values, plus one pixel intensity. Denote the intensity of the source image at (x, y) as s(x,y) and the value to solve for as v(x,y). For each pixel, then, we have two objectives:
    1. minimize (v(x+1,y)-v(x,y) - (s(x+1,y)-s(x,y)))^2
    2. minimize (v(x,y+1)-v(x,y) - (s(x,y+1)-s(x,y)))^2
    Note that these could be solved while adding any constant value to v, so we will add one more objective:
    3. minimize (v(1,1)-s(1,1))^2
    
    :param toy_img: numpy.ndarray
    """
    
    imgGrey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img=img/255.0
    imgGrey=imgGrey/255.0
    bkg=img
    im_h, im_w = imgGrey.shape
    im2var = np.arange(im_h * im_w).reshape(im_h, im_w)
    e=0
    totalE=(4*im_h*im_w)+1
    A=lil_matrix((totalE,im_h*im_w), dtype=np.float32)
    b=lil_matrix((totalE,3), dtype=np.float32)
    print(np.shape(A))
    for y in range(0,im_h):
        for x in range(0,im_w):
#             print(imgedge[y,x])
            if(x!=im_w-1):               
                A[e,im2var[y,x+1]] = -1
                A[e,im2var[y,x]] = 1
                if(imgedge[y,x]==255):
                    b[e]=(bkg[y,x] - bkg[y,x+1])
            else:
                A[e,im2var[y,x]] = 1
                b[e]=bkg[y,x]
            e = e + 1;

            if(y!=im_h-1):               
                A[e,im2var[y+1,x]] = -1
                A[e,im2var[y,x]] = 1
                if(imgedge[y,x]==255):
                    b[e]=bkg[y,x] - bkg[y+1,x]
            else:
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x])
                b[e]=bkg[y,x]
            e = e + 1;
            
            if(x!=0):               
                A[e,im2var[y,x-1]] = -1
                A[e,im2var[y,x]] = 1
                if(imgedge[y,x]==255):
                    b[e]=bkg[y,x] - bkg[y,x-1]
            else:
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x])
                b[e]=bkg[y,x]
            e = e + 1;

            if(y!=0):               
                A[e,im2var[y-1,x]] = -1
                A[e,im2var[y,x]] = 1
                if(imgedge[y,x]==255):
                    b[e]=bkg[y,x] - bkg[y-1,x]
            else:
                A[e,im2var[y,x]] = 1
                vals=abs(bkg[y,x])
                b[e]=bkg[y,x]
            e = e + 1;

#     A[e][im2var[0][0]] = 1
#     vals=abs(bkg[0,0])
#     b[e]=(bkg[0,0])[np.where(vals == np.amax(vals))[0][0]]

    res = np.zeros(img.shape,dtype=np.float32)
    bMatrix=b.tocsr()
    for i in range(0,3):
        v=linalg.lsqr(A.tocsr(), bMatrix.getcol(i).toarray().reshape((-1,)))
        res[:,:,i]=np.reshape(v[0], (im_h,im_w))
    return res


# In[32]:


child = cv2.cvtColor(cv2.imread('samples/child.png'), cv2.COLOR_BGR2RGB)
# color_blind4 = cv2.cvtColor(color_blind4, cv2.COLOR_BGR2GRAY).astype('double') / 255.0
childEdges = cv2.Canny(child,25,50)
# print(childEdges.shape)
plt.imshow(childEdges)
res=flatenning(child,childEdges)
plt.imshow(res)


# In[33]:


res1=np.zeros(res.shape)
res1[:,:,0]=res[:,:,2]*255
res1[:,:,1]=res[:,:,1]*255
res1[:,:,2]=res[:,:,0]*255
cv2.imwrite('child_flat.jpeg', res1) 


# In[27]:


me = cv2.cvtColor(cv2.imread('samples/mount_rushmore.jpg'), cv2.COLOR_BGR2RGB)
# color_blind4 = cv2.cvtColor(color_blind4, cv2.COLOR_BGR2GRAY).astype('double') / 255.0
meEdges = cv2.Canny(me,100,400)
# print(childEdges.shape)
# plt.imshow(meEdges)
res=flatenning(me,meEdges)
plt.imshow(res)


# In[31]:


res1=np.zeros(res.shape)
res1[:,:,0]=res[:,:,2]*255
res1[:,:,1]=res[:,:,1]*255
res1[:,:,2]=res[:,:,0]*255
cv2.imwrite('mount_rushmore_flat.jpeg', res1) 

