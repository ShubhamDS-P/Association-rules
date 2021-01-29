# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 14:57:34 2021

@author: Shubham
"""

import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules

#Importing the data

my_movies = pd.read_csv("D:\\Data Science study\\assignment\\Sent\\9 Association Rules\\my_movies.csv")

# The given dataframe is combination of the original data columns and their dummies 
# So we will seperate them and create a dataframe that we required for our use

movies = my_movies.iloc[:,5:]   #generally we require the binary data to work so we have extracted the binary data

frequent_movies = apriori(movies,min_support  = 0.2, max_len = 2,use_colnames = True)

frequent_movies.describe()

# Sorting most frequent movies according to support

frequent_movies.sort_values('support',ascending = False, inplace = True)

movie_rules = association_rules(frequent_movies, metric = "confidence",min_threshold = 0.85)

# By changing the various values of support and confidence we have narrowed down our rules to 5
# Now we will see if there are any redundant rules present in them and if there are then remove them to get the final rules

def to_list(i):
    return(sorted(list(i)))
    
ma_movies = movie_rules.antecedents.apply(to_list)+movie_rules.consequents.apply(to_list)

ma_movies = ma_movies.apply(sorted)  #Sorting the data

rule_sets = list(ma_movies)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rule_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rule_sets.index(i))

# Getting rules without any redundancy

rules_no_redundancy = movie_rules.iloc[index_rules,:]

# Sorting them with respect to lift
 
rules_no_redundancy.sort_values('lift',ascending=False)

rules_no_redundancy

# Lets plot the graph for our rules based on the support and confidence

import matplotlib.pyplot as plt

plt.scatter(rules_no_redundancy.support,rules_no_redundancy.confidence);plt.xlabel("Support");plt.ylabel("Confidence")
