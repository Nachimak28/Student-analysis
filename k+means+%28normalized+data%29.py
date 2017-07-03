
# coding: utf-8

# In[260]:

import csv
import numpy as np
import statistics as st
import matplotlib.pyplot as plt                                #importing matplotlib for plotting graphs
from matplotlib import style
style.use("ggplot")

#from sklearn.cluster import DBSCAN

from sklearn.cluster import KMeans


# In[261]:


data = []
def read_csv_file(filename):
    f = open(filename)
    for row in csv.reader(f):
        data.append(row)
        #copy.append(row)
    print(data)
    f.close()


# In[262]:

read_csv_file("C:/Users/nachiket/Desktop/attendance research correlation/ds.csv")


# In[263]:


print(data)
print(len(data))
#print("\n")
#print(copy)


# In[264]:

#make separate lists for attendance and marks
x = []
y = []
def make_separate_lists(data):
    for row in range(len(data)):
        x.append(float(data[row][0]))
        y.append(float(data[row][1]))
    print(x)
    print(y)

    
make_separate_lists(data)
print(len(x))
print(len(y))


# In[265]:

#normalization new_element = (element-mean)/range(max-min)
p = []
q = []
def normalization(x,y):
    xbar = st.mean(x)
    sx = max(x)-min(x)
    ybar = st.mean(y)
    sy = max(y)-min(y)
    for i in range(len(x)):
        data[i][0] = (x[i] - xbar)/sx
        data[i][1] = (y[i] - ybar)/sy
    print(data)
    #print(p)
    #print("\n")
    #print(q)

normalization(x,y) 
#print("\n")
#print(copy)


# In[266]:

X = np.array(data)
print(X)


# In[267]:

plt.scatter(X[:,0],X[:,1])
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[268]:

km = KMeans(n_clusters=3, random_state = 42)
km.fit(X)


# In[269]:

centroids = km.cluster_centers_
labels = km.labels_
print(centroids)
print(labels)


# In[270]:

colors = 10*['r.','g.','b.','c.','k.','y.','m.']

for i in range(len(X)):
    print("coordinate:",X[i],"label:",labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 150, linewidths = 5, zorder = 10)
plt.xlabel("Attendance")
plt.ylabel("Marks")
plt.show()


# In[271]:

#print("Students with moderate attendance and marks between 8 - 16 label 2: ", np.count_nonzero(labels == 2))



# In[272]:

#print("Students with less attendance and less marks (or absentee) label 1: ", np.count_nonzero(labels == 1))


# In[273]:

#print("Students with high attendance and marks >16 and <= 20 label 0: ", np.count_nonzero(labels == 0))


# In[274]:

print("Attendance ", "Marks ", "Labels")
for i in range(len(x)):
    print(x[i], "\t", y[i], "\t", labels[i])
    print("\n")


# In[275]:

np.count_nonzero(labels == 0)


# In[276]:

np.count_nonzero(labels == 1)


# In[277]:

np.count_nonzero(labels == 2)


# In[278]:

#np.count_nonzero(labels ==3)


# In[281]:

label0 = []
print("Attendance and marks for label 0")
for i in range(len(x)):
    if labels[i] == 0:
        print(x[i],"\t",y[i])
        temp = [x[i],y[i], labels[i]]
        label0.append(temp)
         

np.count_nonzero(labels == 0)


# In[282]:

label1 = []
print("Attendance and marks for label 1")
for i in range(len(x)):
    if labels[i] == 1:
        print(x[i],"\t",y[i])
        temp = [x[i],y[i], labels[i]]
        label1.append(temp)

np.count_nonzero(labels == 1)


# In[283]:

label2= []
print("Attendance and marks for label 2")
for i in range(len(x)):
    if labels[i] == 2:
        print(x[i],"\t",y[i])
        temp = [x[i],y[i],labels[i]]
        label2.append(temp)
        
#print(tp)
np.count_nonzero(labels == 2)


# In[236]:

'''
print("Attendance and marks for label 3")
for i in range(len(x)):
    if labels[i] == 3:
        print(x[i],"\t",y[i])

print("high attendance low marks")
'''


# In[237]:

#correlation 
#formula 
#r = summation(x*y)- n*xbar*ybar
#    ------------------------------
#            n*stddev(x)*stddev(y)


# In[284]:

import statistics as st

def normforcorr(x):
    a = x
    meanval = st.mean(x)
    rx = max(x)-min(x)
    for i in range(len(x)):
        a[i] = (x[i] - meanval)/rx
    return a

def correlation(x,y):
    n = len(x)
    a = normforcorr(x)
    b = normforcorr(y)
    ab = []
    for i in range(len(x)):
        ab.append(float(a[i]*b[i]))
    r = (sum(ab) - n*st.mean(a)*st.mean(b))/(n*st.stdev(a)*st.stdev(b))
    print("Correlation coefficient : ",r)
    


# In[285]:

correlation(x,y)


# In[286]:

plt.hist(labels)
plt.title("Histogram")
plt.show()


# In[329]:

len(label0)


# In[326]:

at0 = []
at1 = []
at2 = []
mk0 = []
mk1 = []
mk2 = []
def disect_lists(label0,label1,label2):
    for i in range(len(label0)):
        at0.append(label0[i][0])
        mk0.append(label0[i][1])
        
    for i in range(len(label1)):   
        at1.append(label1[i][0])
        mk1.append(label1[i][1])
        
    for i in range(len(label2)):
        at2.append(label2[i][0])
        mk2.append(label2[i][1])



# In[330]:

disect_lists(label0, label1, label2)


# In[331]:

at0


# In[350]:

minat0 = min(at0)
minmk0 = min(mk0)
print(minat0)
print(minmk0)


# In[351]:

minat1 = min(at1)
minmk1 = min(mk1)
print(minat1)
print(minmk1)


# In[352]:

minat2 = min(at2)
minmk2 = min(mk2)
print(minat2)
print(minmk2)


# In[362]:

if minat0 < minat1 and minat0 < minat2:
    print("label 0 with low attendance and low marks(<8)")
    per0 = (np.count_nonzero(labels == 0)/len(x))*100
    print(per0,"% students lie in this cluster")
    
    if minat1 < minat2:
        print("label 1 with avg attendance and mks between 8 - 16")
        per1 = (np.count_nonzero(labels == 1)/len(x))*100
        print(per1,"% students lie in this cluster")
    
        print("label 2 with high attendance and high marks (>16)")
        per2 = (np.count_nonzero(labels == 2)/len(x))*100
        print(per2,"% students lie in this cluster")
        
    else:
        print("label 2 with avg attendance and mks between 8 - 16")
        per2 = (np.count_nonzero(labels == 2)/len(x))*100
        print(per2,"% students lie in this cluster")
        
        print("label 1 with high attendance and high marks (>16)")
        per1 = (np.count_nonzero(labels == 1)/len(x))*100
        print(per1,"% students lie in this cluster")
        
elif minat1 < minat0 and minat1 < minat2:
    print("label 1 with low attendance and low marks(<8)")
    per1 = (np.count_nonzero(labels == 1)/len(x))*100
    print(per1,"% students lie in this cluster")
    
    if minat0 < minat2:
        print("label 0 with avg attendance and mks between 8 - 16")
        per0 = (np.count_nonzero(labels == 0)/len(x))*100
        print(per0,"% students lie in this cluster")
        
        print("label 2 with high attendance and high marks (>16)")
        per2 = (np.count_nonzero(labels == 2)/len(x))*100
        print(per2,"% students lie in this cluster")
        
    else:
        print("label 2 with avg attendance and mks between 8 - 16")
        per2 = (np.count_nonzero(labels == 2)/len(x))*100
        print(per2,"% students lie in this cluster")
        
        print("label 0 with high attendance and high marks (>16)")
        per0 = (np.count_nonzero(labels == 0)/len(x))*100
        print(per0,"% students lie in this cluster")
        
else:
    print("label 2 with low attendance and low marks(<8)")
    per2 = (np.count_nonzero(labels == 2)/len(x))*100
    print(per2,"% students lie in this cluster")
    
    if minat0 < minat1:
        print("label 0 with avg attendance and mks between 8 - 16")
        per0 = (np.count_nonzero(labels == 0)/len(x))*100
        print(per0,"% students lie in this cluster")
        
        print("label 1 with high attendance and high marks (>16)")
        per1 = (np.count_nonzero(labels == 1)/len(x))*100
        print(per1,"% students lie in this cluster")
    else:
        print("label 1 with avg attendance and mks between 8 - 16")
        per1 = (np.count_nonzero(labels == 1)/len(x))*100
        print(per1,"% students lie in this cluster")
        
        print("label 0 with high attendance and high marks (>16)")
        per0 = (np.count_nonzero(labels == 0)/len(x))*100
        print(per0,"% students lie in this cluster")


# In[354]:

print(label1)


# In[355]:

print(label0)


# In[356]:

print(label2)


# In[361]:

correlation(x,y)


# In[ ]:



