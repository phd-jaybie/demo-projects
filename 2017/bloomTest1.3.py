
# coding: utf-8

# In[1]:


# A simple bloom filter implementation for validating its viability
# for object detection (potentially for privacy-preserving AR).

# Date 16-Nov-2017: This version now uses an LSH approach and adds the
# resulting  (post-LSH) reduced binary vector (instead of the integer
# version) representation of a SIFT feature as an element to the Bloom
# filter. The previous version further compresses the resulting binary
# vector into an integer and, then, adds the direct byte representation
# to the Bloom. Here, the byte representation of the binary vector
# (pre-integer) is added to the Bloom.

from pybloomfilter import BloomFilter
import numpy as np
import cv2
import time
import sys


# In[2]:


detector = cv2.xfeatures2d.SIFT_create()

train_img = cv2.imread('train.jpg',0)
query_img = cv2.imread('raw.png',0)


# In[3]:


T_kp, T_des = detector.detectAndCompute(train_img, None)


# In[4]:


_, dim = T_des.shape
LSH_dim = 16
np.random.seed(0)
LSH_random_vectors = np.random.randn(dim, LSH_dim)
powers_of_two = 1 << np.arange(LSH_dim-1, -1, -1)


# In[5]:


bf = BloomFilter(10**(LSH_dim/4),0.01,None)

# We maximize the efficiency by utilizing matrix operations
# for the crude LSH implementation

t0 = time.process_time()

Q_kp, Q_des = detector.detectAndCompute(query_img, None)

t1 = time.process_time()

Q_reflections = Q_des.dot(LSH_random_vectors) >= 0
#Q_bin = Q_reflections.dot(powers_of_two)

# And we remove duplicates to ensure uniqueness of features
for q in np.array(Q_reflections, dtype=int):
    # needs to insert here a method for re-hashing or
    # transforming the array list of descriptors to a bit array
    
    # m = m.tostring(None) # using this one results to a trivial outcome,
    # as direct implementation of bloom filters requires an exactness,
    # as we have earlier suspected. Thus, some form of generalization
    # to get rid of 'exactness' has to be implemented, before we add it
    # to the bloom filter. In the Duke paper, they implemented LSH as a
    # form of generalization.
    
    # Now, we use a crude LSH that results to a LSH_dim-bit output.
    # We hash individual feature vectors of dimension 1 x 128 through a
    # 128 x LSH_dim random vector set.
    # print(m_reflections, m_bin)
    bf.add(q.tostring(None))
    
t2 = time.process_time()


# In[6]:


print(bf)
len(bf)


# In[7]:


t3 = time.process_time()
T_reflections = T_des.dot(LSH_random_vectors) >= 0
#T_bin = T_reflections.dot(powers_of_two)

count = 0
for n in np.array(T_reflections, dtype=int):
    n = n.tostring(None)
    if (n in bf):
        count = count + 1
        
t4 = time.process_time()


# In[10]:


print("Feature Extraction Time:", t1-t0)
print("LSH to Bloom time:", t2-t1)
print("Matches:", count)
print("Number of Training features", len(T_des) )
print("Number of Query features", len(Q_des))
print("Percent Matches:", count*100/len(T_des) )
print("Checking the Bloom time:", t4-t3)


# In[11]:


bf = BloomFilter(10**(LSH_dim/4),0.01,None) # resetting the bloom

# testing it with a different query image

test_img = cv2.imread('img_fjords.jpg',0)

t0 = time.process_time()

Q_kp, Q_des = detector.detectAndCompute(test_img, None)

t1 = time.process_time()

Q_reflections = Q_des.dot(LSH_random_vectors) >= 0
#Q_bin = Q_reflections.dot(powers_of_two)

for q in np.array(Q_reflections, dtype=int):
    q = q.tostring(None)
    bf.add(q)
    
t2 = time.process_time()


# In[12]:


t3 = time.process_time()
T_reflections = T_des.dot(LSH_random_vectors) >= 0
#T_bin = T_reflections.dot(powers_of_two)

count = 0
for n in np.array(T_reflections, dtype=int):
    if (n.tostring(None) in bf):
        count = count + 1
        
t4 = time.process_time()


# In[13]:


print("Feature Extraction Time:", t1-t0)
print("LSH to Bloom time:", t2-t1)
print("Matches:", count)
print("Number of Training features", len(T_des) )
print("Number of Query features", len(Q_des))
print("Percent Matches:", count*100/len(T_des) )
print("Checking the Bloom time:", t4-t3)

