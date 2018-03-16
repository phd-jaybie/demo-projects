
# coding: utf-8

# In[1]:


from bitarray import bitarray
import mmh3


# In[4]:


class BloomFilter:
    
    def __init__(self, size, hash_count):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)
        
    def add(self, string):
        for seed in range(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            self.bit_array[result] = 1
        
    def lookup(self, string):
        for seed in range(self.hash_count):
            result = mmh3.hash(string, seed) % self.size
            if self.bit_array[result] == 0:
                return "Nope"

        return "Probably"


# In[5]:


## We import the Unix standard words file.

bf = BloomFilter(2600000, 7)
lines = open("/usr/share/dict/words").read().splitlines()
for line in lines:
    bf.add(line)


# In[9]:


print(bf.lookup("Max"))


# In[ ]:


## Code for time comparison with simple iteration.
import time

huge = []
lines = open("/usr/share/dict/words").read().splitlines()
for line in lines:
    huge.append(line)


# In[19]:


start = time.process_time()
bf.lookup("google")
finish = time.process_time()
print("Bloom filter: "+str(finish-start))

start = time.process_time()
for word in huge:
    if word == "google":
        break
finish = time.process_time()
print("Iteration: "+str(finish-start))


# In[20]:


start = time.process_time()
bf.lookup("apple")
finish = time.process_time()
print("Bloom filter: "+str(finish-start))

start = time.process_time()
for word in huge:
    if word == "apple":
        break
finish = time.process_time()
print("Iteration: "+str(finish-start))

