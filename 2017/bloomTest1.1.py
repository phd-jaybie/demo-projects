
# coding: utf-8

# In[1]:


# A simple bloom filter implementation for validating its viability
# for object detection (potentially for privacy-preserving AR).

# Date 13-Nov-2017: In this version, we are just using plain features
# and adding them to the Bloom. Specifically, we just convert each
# the 128-element array of SIFT features to their byte representation
# and add them to the Bloom; thus, each

from pybloomfilter import BloomFilter
import numpy as np
import cv2
import sys


# In[2]:


detector = cv2.xfeatures2d.SIFT_create()

bf = BloomFilter(10000000, 0.01, None)

train_img = cv2.imread('train.jpg',0)
query_img = cv2.imread('raw.png',0)


# In[3]:


T_kp, T_des = detector.detectAndCompute(train_img, None)
Q_kp, Q_des = detector.detectAndCompute(query_img, None)

_, dim = Q_des.shape
LSH_dim = 64
np.random.seed(0)
LSH_random_vectors = np.random.randn(dim, LSH_dim)
powers_of_two = 1 << np.arange(LSH_dim-1, -1, -1)


# In[4]:


for m in Q_des:
    # needs to insert here a method for re-hashing or
    # transforming the array list of descriptors to a bit array
    
    m = m.tostring(None) # using this one results to a trivial outcome,
    # as direct implementation of bloom filters requires an exactness,
    # as we have earlier suspected. Thus, some form of generalization
    # to get rid of 'exactness' has to be implemented, before we add it
    # to the bloom filter. In the Duke paper, they implemented LSH as a
    # form of generalization.
    
    bf.add(m)


# In[5]:


count = 0
print(bf)
len(bf)
#des2[0].shape
sys.getsizeof(bf)


# In[6]:


for n in T_des:
    n = n.tostring(None)
    if (n in bf):
        count = count + 1


# In[7]:


print("Matches:", count)
print("Percent Matches:", count*100/len(T_des))


# In[8]:


test_img = cv2.imread('img_fjords.jpg',0)
T_kp, T_des = detector.detectAndCompute(test_img, None)

