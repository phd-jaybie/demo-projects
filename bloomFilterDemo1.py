
# coding: utf-8

# In[1]:


from bitarray import bitarray
import mmh3


# In[2]:


bf_size = 10 # Bloom filter size


# In[3]:


bit_array = bitarray(bf_size)
bit_array.setall(0) # Initialize all bits to 0.


# In[11]:


## Inserting "hello" to our bloom filter.

b1 = mmh3.hash("hello", 41) % 10 # Hash our input using mmh3 then get modulo of the bf_size. Equals 7
bit_array[b1] = 1 # Set the bit position correponsing to the result of the hash-module operation above.
b2 = mmh3.hash("hello", 42) % 10 # Do it second time to have 2 bits set per insertion.
bit_array[b2] = 1 # Do it for the second bit set.

bit_array


# In[8]:


## Checking if "hello" is in our set.

b1 = mmh3.hash("hello", 41) % 10 # Do the same hash-modulo operation
b2 = mmh3.hash("hello", 42) % 10 
if bit_array[b1] == 1 and bit_array[b2] == 1: # Check if the resulting bit positions are set.
    print("Probably in set")
else:
    print("Definitely not in set")

