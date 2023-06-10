import numpy as np
from numpy import linalg as LA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.preprocessing import image as I
import matplotlib.pyplot as plt
import os
from numpy.linalg import norm
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
import math 
import pickle
import time
import cv2

def L1_norm_dist(a,b):
    d = 0
    for i in range(len(a)):
        d = d + abs(a[i] - b[i])
#         print(a[i] , b[i])
#     print(d)
    return d 

    
m = 9         # number of eigenvectors
b_len = 7     # length of bits to represent buckets  


#################################loading pre-computed features of images #########################

images = pickle.load(open('features_vggface2.pickle' , 'rb'))
images = np.asarray(images)
filenames = pickle.load(open('filenames_vggface2.pickle' , 'rb'))

## changing address according to files storage #########################
for i in range(len(filenames)):
    filenames[i] = filenames[i].split('datasets/')[1]

for i in range(len(filenames)):
    a = filenames[i].split('/')[-1]
    b = filenames[i].split('/')[-2]
    filenames[i] = filenames[i].rsplit('/',1)[0] + '/' + b + '_' + a
#print(filenames[0])
print('--------------------files loaded---------------------')
##################################################################################



##################################finding number of images in each class ##########################

total_class_count = {}
COUNT=[]

for i in range(len(filenames)):
    x = filenames[i].split("/")
    a = x[-2:-1]

    if(a[0] in total_class_count):
        total_class_count[a[0]] = total_class_count[a[0]] + 1
    else:
        total_class_count[a[0]] = 1
    

for j in total_class_count:
    COUNT.append(total_class_count[j])

#print(total_class_count)
#print(len(COUNT))
total_class_count_list = list(total_class_count)

###############################################################################



########################################computing LDA #########################

Y = []
for i in range(len(total_class_count)):
    for j in range(COUNT[i]):
        Y.append(i)
        


lda = LinearDiscriminantAnalysis(n_components=m)
projection_matrix_lda = lda.fit(images, Y)
reduced_images = lda.transform(images)

prod = reduced_images.T
# print(prod.shape)


from numpy import binary_repr
n ,d = images.shape
buc_in_each_vec = []
buc_in_each_vec.append(2**b_len)


#storing data
b_array = np.ones([m , n] , dtype = int) 
b_array_dec = np.ones([m , n] , dtype = int) 


#w 1 = (max(u T1 X ) − min(u T1 X ))/2 b 1
w1 = (max(prod[0]) - min(prod[0]))/ (2**b_len)
# print(w1)
#b 1 j = decimal2binary (u T1 x j − min(u T1 X ))/w 1
min_prod_0 = min(prod[0])
for j in range(n):
    b1 = binary_repr(math.ceil((prod[0][j] - min_prod_0)/w1)  , width = None)
    b_array[0][j] = b1
    
    # storing decimal number
    b_array_dec[0][j] = math.ceil((prod[0][j] - min_prod_0)/w1)
#     print(b_array_dec[0][j])
    
    
for L in range(1,m):
    #print(L)
    bnL = math.ceil( ( max(prod[L]) - min(prod[L]) ) / w1)
#     print(bnL)
    bL = math.ceil(math.log( ( (max(prod[L]) - min(prod[L]) ) / w1 ) , 2 ) )
    
    w = ((max(prod[L])) - (min(prod[L]))) / (2 ** bL) 
#     print(w)
    
    buc_in_each_vec.append(2**bL)
    
    min_L = min(prod[L])
    for j in range(n):
        bLxj = binary_repr(math.ceil ( (prod[L][j] - min_L )  / w) ) 
        b_array[L][j] = bLxj
                                  
        # storing decimal number
        b_array_dec[L][j] = math.ceil((prod[L][j] - min_L ) / w) 
#         print(b_array_dec[L][j])
        
                            
coordinates = b_array_dec.T
# print(coordinates.shape)

##################################################################################



############################################Creating Bucket lists ########################################
'''m vectors with each vector having number of lists = number of buckets in that vector'''
vectors = []

for i in range(m):
    lists = []
    vectors.append(lists)
print(len(vectors))

for j in range(m):
    for i in range((2**b_len) * 4 + 1 ):
        lists = []
        vectors[j].append(lists)

for j in range(len(filenames)-1):
    for i in range(m):
        vectors[i][coordinates[j][i]].append(j)
    
# print(len(vectors[1][200]))


def testing(query_index , K):
    dist = []
    k = []                              #index values of images
    for i in range(0,K):
        dist.append(50000 * (i+1) )
        k.append(0)

    query_point = coordinates[query_index]                 # query image
    
    for i in range(len(coordinates)):
        d = L1_norm_dist(query_point ,coordinates[i]) 
        if(d < max(dist) ):

            maxpos = dist.index(max(dist)) 
            
            dist.remove(max(dist))
            dist.append(d)
            
            k.pop(maxpos)
            k.append(i)
    

    # plt.figure(figsize=(20,10)) # specifying the overall grid size

    # for i in range(len(k)):
        # plt.subplot(5,5,i+1)    # the number of images in the grid is 5*5 (25)
        # img = mpimg.imread(filenames[k[i]]) 
        # plt.imshow(img)

    # plt.show()
    return k

#################################################################################



"""# Processing Querry Image"""

def processing_query(query_path):
    print('--------------------------------',query_path)
    query_index = 0
    for i in range(len(filenames)):
        a = os.path.split(filenames[i])[-1]
        b = os.path.split(query_path)[1]
        if(a == b):
            query_index = i 
    print('---------------------------------',query_index)
    name = os.path.split(query_path)[-1].split('_')[0]
    return query_index , name



# stores the top 5 images in the folder named results
def printing_results(address_list):
    # print('results:')
    N = 5
    final = address_list[-N:]

    results = os.listdir('static/results/')
    # print(results)
    for f in results:
        os.remove(os.path.join(r'D:\REST_API\static\results', f))
    # results = os.listdir('static/results/')
    # print('results',results)

    d = {"status":"success" ,"result_images": []}
    # my_dict = {"Name":[]}
    # d["status"] = ["success"]

    for i in range(N):
        # print(i)
        img = cv2.imread(final[i])
        # file_ext = os.path.splitext(final[i])[1]
        file_name = os.path.split(final[i])
        # print(file_name[1])
        d['result_images'].append('static/results' + file_name[1])

        cv2.imwrite('static/results/' + file_name[1] , img)

    return d 





#this method calls all the functions one by one
query_index = 0
def call_search(file_path):
    
    query_index, name = processing_query(file_path)
    top_k = 5
    k = testing(query_index , top_k)
    print(k)

    address_list = []
    for i in range(len(k)):
        address_list.append(filenames[k[i]])

    d = printing_results(address_list)
   
    return d , name
