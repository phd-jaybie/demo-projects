
# coding: utf-8

# In[1]:


# A simple bloom filter implementation for validating its viability
# for object detection (potentially for privacy-preserving AR).

# Date 22-Nov-2017: We're adding a verification Bloom filter that stores
# bit positions per insertion as elements.

from pybloomfilter import BloomFilter

import numpy as np
import cv2
import time
import sys
import mmh3


# In[2]:


detector = cv2.xfeatures2d.SIFT_create()

train_img = cv2.imread('train.jpg',0)
query_img = cv2.imread('raw.png',0)

LSH_dim = 8
L_buckets = 1
K = 8


# In[3]:


T_kp, T_des = detector.detectAndCompute(train_img, None)

_, dim = T_des.shape


# In[4]:


# Starting below, we are using the multiple Blooms.


# In[15]:


LSH_random_vectors_set = []
#powers_of_two = 1 << np.arange(LSH_dim-1, -1, -1)

# creating the multiple LSH random vectors
for i in range(L_buckets):
    np.random.seed(i)
    LSH_random_vectors_set.append(np.random.randn(dim, LSH_dim))

# creating the multiple Bloom Filters
BF_set = []
for i in range(L_buckets):
    BF_set.append(np.zeros(2**(LSH_dim+K+1), dtype=np.int))
    
VBF_set = []
for i in range(L_buckets):
    VBF_set.append(BloomFilter(10000000, 0.01, None))


# In[16]:


len(BF_set[0])
#BF_set[0][1]=2
sum(BF_set[0])


# In[17]:


t0 = time.process_time()

Q_kp, Q_des = detector.detectAndCompute(query_img, None)

t1 = time.process_time()

for i in range(L_buckets):
    Q_reflections = Q_des.dot(LSH_random_vectors_set[i]) >= 0
    
    for q in np.array(Q_reflections, dtype=int):
        # Experimenting on bit position distribution for Bloom filters from
        # scratch.
        hq1, hq2 = mmh3.hash64(q,i,signed=False)
        v_element = 0
        
        for k in range(K):
            pos = (hq1 + k*hq2 + k**2)%len(BF_set[i])
            BF_set[i][pos] = BF_set[i][pos] + 1
            
            v_element = v_element + 2**pos
            #print(i, pos, BF_set[i][pos])
        #print(q, q.tostring(None))
        #BF_set[i].add(q.tostring(None))
        VBF_set[i].add(v_element)
    
t2 = time.process_time()
print(t2-t1)


# In[18]:


sum(BF_set[0])


# In[19]:


t3 = time.process_time()

count = 0

for n in T_des:
    inBucket = True
    
    for i in range(len(BF_set)):
        T_reflections = n.dot(LSH_random_vectors_set[i]) >= 0
        r = np.array(T_reflections, dtype=int)
        countk = 0
        
        # This is the querying-the-Bloom-filter stage
        ht1, ht2 = mmh3.hash64(r,i,signed=False)
        v_element = 0
        for k in range(K):
            pos = (ht1 + k*ht2 + k**2)%len(BF_set[i])
            v_element = v_element + 2**pos
            if (BF_set[i][pos] > 0):
                countk = countk + 1
            #elif (BF_set[i][pos-1]> 0 | BF_set[i][pos+1] > 0):
            #    countk = countk + 0.5

        inBucket = inBucket and (countk>K-1) and (v_element in VBF_set[i])
        
    count = count + inBucket

t4 = time.process_time()


# In[20]:


print("Feature Extraction Time:", t1-t0)
print("LSH to Bloom time:", t2-t1)
print("Matches:", count)
print("Number of Reference features:", len(T_des) )
print("Number of Query features:", len(Q_des))
print("Percent Matches:", count*100/len(T_des) )
print("Checking the Bloom time:", t4-t3)


# In[21]:


# resetting the multiple Bloom Filters

test_img = cv2.imread('img_fjords.jpg',0)

BF_set = []
for i in range(L_buckets):
    BF_set.append(np.zeros(2**(LSH_dim+K+1), dtype=np.int))

VBF_set = []
for i in range(L_buckets):
    VBF_set.append(BloomFilter(10000000, 0.01, None))

t0 = time.process_time()

Q_kp, Q_des = detector.detectAndCompute(test_img, None)

t1 = time.process_time()

for i in range(L_buckets):
    Q_reflections = Q_des.dot(LSH_random_vectors_set[i]) >= 0
    
    for q in np.array(Q_reflections, dtype=int):
        # Experimenting on bit position distribution for Bloom filters from
        # scratch.
        v_element = 0
        hq1, hq2 = mmh3.hash64(q,i,signed=False)

        for k in range(K):
            pos = (hq1 + k*hq2 + k**2)%len(BF_set[i])
            BF_set[i][pos] = BF_set[i][pos] + 1
            
            v_element = v_element + 2**pos
            #print(i, pos, BF_set[i][pos])
        #print(q, q.tostring(None))
        #BF_set[i].add(q.tostring(None))
        VBF_set[i].add(v_element)
    
t2 = time.process_time()
print(t2-t1)


# In[22]:


t3 = time.process_time()

count = 0

for n in T_des:
    inBucket = True
    
    for i in range(len(BF_set)):
        T_reflections = n.dot(LSH_random_vectors_set[i]) >= 0
        r = np.array(T_reflections, dtype=int)
        countk = 0
        
        # This is the querying-the-Bloom-filter stage
        v_element = 0
        ht1, ht2 = mmh3.hash64(r,i,signed=False)
        for k in range(K):
            pos = (ht1 + k*ht2 + k**2)%len(BF_set[i])
            v_element = v_element + 2**pos
            if (BF_set[i][pos] > 0):
                countk = countk + 1
            #elif (BF_set[i][pos-1]> 0 | BF_set[i][pos+1] > 0):
            #    countk = countk + 0.5

        inBucket = inBucket and bool(countk>K-1) and (v_element in VBF_set[i])
        
    count = count + inBucket

t4 = time.process_time()


# In[23]:


print("Feature Extraction Time:", t1-t0)
print("LSH to Bloom time:", t2-t1)
print("Matches:", count)
print("Number of Training features:", len(T_des) )
print("Number of Test Query features:", len(Q_des))
print("Percent Matches:", count*100/len(T_des) )
print("Checking the Bloom time:", t4-t3)

