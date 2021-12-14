#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import ttest_1samp
from statsmodels.stats.power import tt_ind_solve_power


# # t-test
At test is inferntial statistics which is used to determine if there is a significant difference between the menas of two groups which may be related in certain features

T-test has 2 types: 1)One sampled test 2)Two sampled test

    t=(sample mean - population mean) / standard error
# In[2]:


ages=[10,20,35,50,28,40,55,18,16,55,18,16,55,30,25,43,18,30,28,14,24,16,17,32,35,89,25,78,25,60]


# In[3]:


ages_mean=np.mean(ages)
print(ages_mean)


# In[4]:


#Lets take sample
sample_size=10
age_sample=np.random.choice(ages,sample_size)
age_sample


# In[5]:


from scipy.stats import ttest_1samp


# In[6]:


ttest,p_value=ttest_1samp(age_sample,33)


# In[7]:


print(p_value)


# In[8]:


if p_value < 0.05:
    print("We are rejecting null hypothesis")
else:
    print("We are accepting null hypothesis")


# In[9]:


df=pd.read_excel("C:/Users/VidhyaDivya/OneDrive/Desktop/Data/FDS/Practical10/Result.xlsx")
df


# In[10]:


df.describe()


# Hypothesis Testing or Significance Testing is a set of formal procedures used by statisticians to either accept or reject statistical hypothesis.
# 
# Null hypothesis, HO - represents a hypothesis of chance basis.
# 
# Alternative hypothesis, Ha - represents a hypothesis of observations which are influenced by some non-random cause.
# 
# (1)check if total mean value of marks is not more than 113.
# 
# Null Hypothesis will be
# 
# Ho : mu <= 113
# 
# Alternate Hypothesis will be
# 
# Ha : mu > 113
# 
# One way hypothesis

# In[11]:


# Null hypothesis
Ho = "mu <= 113"
# Alternative hypothesis
Ha = "mu > 113"
# alpha
al = 0.05
#mu -> mean
mu = 113
# tail type
tt = 1
# data
marks = df['Total'].values
print("Ho:", Ho)
print("Ha:", Ha)
print("alpha:", al)
print("mu:", mu)
print(marks)
print("")


# In[13]:


ts, pv = ttest_1samp(marks, mu)
print("t-statistics",ts)
print("p-vaues",pv)
t2pv = pv
t1pv = pv*2
print("One tail p-value",t1pv)
print("Two tail p-value",t2pv)


# In[14]:


if tt == 1:
    if t1pv < al:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Not Rejected")
        print("Conclusion:",Ho)
else:
    if t2pv < al/2:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Not Rejected")
        print("Conclusion:",Ho)


# # Two Way Hypothesis

# In[15]:


# null hyp
Ho = "mu = 113"
# alt hyp
Ha = "mu != 113"
# alpha
al = 0.05
# mu - mean
mu = 113


# In[18]:


# tail type
tt = 2
# data
marks = df['Total'].values
print("Ho:", Ho)
print("Ha:", Ha)
print("alpha:", al)
print("mu:", mu)
print(marks)
print("")


# In[19]:


ts, pv = ttest_1samp(marks, mu)
print("t-statistics",ts)
print("p-value",pv)
t2pv = pv
t1pv = pv*2
print("One Tail p-value",t1pv)
print("Two Tail p-value",t2pv)


# In[20]:


if tt == 1:
    if t1pv < al:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Not Rejected")
        print("Conclusion:",Ho)
else:
    if t2pv < al/2:
        print("Null Hypothesis: Rejected")
        print("Conclusion:",Ha)
    else:
        print("Null Hypothesis: Not Rejected")
        print("Conclusion:",Ho)


# # AB Testing
AB Testing is essentially an experiment where two or more variants of a page are shown to users at random, and statistical analysis is used to determine which variation performs better for a given conversion goal.
# In[21]:


sub1 = np.array([45,36,29,40,46,37,43,39,28,33])
sub2 = np.array([40,20,30,35,29,43,40,39,28,31])


# In[22]:


sns.distplot(sub1)


# In[23]:


sns.distplot(sub2)

The two hypothesis for this particulat two sample t-test are as follows:

HO: u1 = u2(The two population means are equal)

HA: u1 not equal to u2(The two population means are not equal)
# In[24]:


t_stat, p_val= stats.ttest_ind(sub1,sub2)
t_stat ,p_val


# In[26]:


#perform two sample t-test with equal variances

stats.ttest_ind(sub1, sub2, equal_var=True)

The t test statistics is 1.3659 and the corresponding two_sided p-value is 0.1887.
Because the p-value of our test(0.1887) is greater than alpha = 0.05, we fail to reject the null hypothesis of the test.

We do not have sufficient evidence to say that the mean marks of sub1 and sub2 between the two is different.
# In[ ]:




