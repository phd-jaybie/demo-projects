
# coding: utf-8

# In[1]:


# A simple bloom filter implementation for validating its viability
# for object detection (potentially for privacy-preserving AR).

# Date 21-Nov-2017: This version now follows the strategy used by Jain's
# which utilizes multiple LSH hash sets and dedicated Bloom filters for
# each. This hopefully reduces the false positives.

from pybloomfilter import BloomFilter
import numpy as np
import cv2
import time
import sys


# In[2]:


detector = cv2.xfeatures2d.SIFT_create()

train_img = cv2.imread('train.jpg',0)
query_img = cv2.imread('raw.png',0)

LSH_dim = 16
L_buckets = 3


# In[3]:


T_kp, T_des = detector.detectAndCompute(train_img, None)

_, dim = T_des.shape


# In[4]:


LSH_random_vectors_set = []
#powers_of_two = 1 << np.arange(LSH_dim-1, -1, -1)

# creating the multiple LSH random vectors
for i in range(L_buckets):
    np.random.seed(i)
    LSH_random_vectors_set.append(np.random.randn(dim, LSH_dim))

# creating the multiple Bloom Filters
BF_set = []
for i in range(L_buckets):
    BF_set.append(BloomFilter(2**(2*LSH_dim),0.01,None))


# In[5]:


t0 = time.process_time()

Q_kp, Q_des = detector.detectAndCompute(query_img, None)

t1 = time.process_time()

# We now add each LSH hash result to their dedicated Bloom Filter
for i in range(L_buckets):
    Q_reflections = Q_des.dot(LSH_random_vectors_set[i]) >= 0
    
    for q in np.array(Q_reflections, dtype=int):
        BF_set[i].add(q.tostring(None))
    
    
t2 = time.process_time()


# In[6]:


#print(bf)
BF_set[0]


# In[7]:


t3 = time.process_time()

count = 0

for n in T_des:
    inBucket = True
    
    for i in range(len(BF_set)):
        T_reflections = n.dot(LSH_random_vectors_set[i]) >= 0
        r = np.array(T_reflections, dtype=int).tostring(None)
        inBucket = inBucket and (r in BF_set[i])
        
    count = count + inBucket

t4 = time.process_time()


# In[8]:


print("Feature Extraction Time:", t1-t0)
print("LSH to Bloom time:", t2-t1)
print("Matches:", count)
print("Number of Training features", len(T_des) )
print("Number of Query features", len(Q_des))
print("Percent Matches:", count*100/len(T_des) )
print("Checking the Bloom time:", t4-t3)
#print("Size of Query Image:",sys.getsizeof(query_img), "and size of Bloom:", sys.getsizeof(bf.to_base64()))


# In[9]:


# resetting the bloom
BF_set = []
for i in range(L_buckets):
    BF_set.append(BloomFilter(2**(2*LSH_dim),0.01,None))

# testing it with a different query image
test_img = cv2.imread('img_fjords.jpg',0)

t0 = time.process_time()

Q_kp, Q_des = detector.detectAndCompute(test_img, None)

t1 = time.process_time()

# We now add each LSH hash result to their dedicated Bloom Filter
for i in range(L_buckets):
    Q_reflections = Q_des.dot(LSH_random_vectors_set[i]) >= 0
    
    for q in np.array(Q_reflections, dtype=int):
        BF_set[i].add(q.tostring(None))
    
    
t2 = time.process_time()


# In[10]:


t3 = time.process_time()

count = 0

for n in T_des:
    inBucket = True
    
    for i in range(len(BF_set)):
        T_reflections = n.dot(LSH_random_vectors_set[i]) >= 0
        r = np.array(T_reflections, dtype=int).tostring(None)
        inBucket = inBucket and (r in BF_set[i])
        
    count = count + inBucket

t4 = time.process_time()


# In[11]:


print("Feature Extraction Time:", t1-t0)
print("LSH to Bloom time:", t2-t1)
print("Matches:", count)
print("Number of Training features", len(T_des) )
print("Number of Query features", len(Q_des))
print("Percent Matches:", count*100/len(T_des) )
print("Checking the Bloom time:", t4-t3)
#print("Size of Query Image:",sys.getsizeof(query_img), "and size of Bloom:", sys.getsizeof(bf.to_base64()))

