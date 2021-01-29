# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:39:39 2021

@author: Shubham
"""

# Now we will create the Association rules for the "book" data set.

import pandas as pd 

from mlxtend.frequent_patterns import apriori,association_rules

# Importing the data 

book = pd.read_csv("D:\\Data Science study\\assignment\\Sent\\9 Association Rules\\book.csv")

# Due to the given data already being in the form of binary we can directly apply the apriori algorithm

frequent_books = apriori(book,min_support = 0.045, max_len = 3,use_colnames = True)

frequent_books.describe()

# Sorting most frequent books according to support

frequent_books.sort_values('support',ascending = False,inplace = True)

frequent_books

book_rules = association_rules(frequent_books,metric = "confidence", min_threshold = 0.8)

print(len(book_rules))

# Now lets try to remove the redundant rules from the list of the rules

def to_list(i):
    return (sorted(list(i)))


ma_book = book_rules.antecedents.apply(to_list)+book_rules.consequents.apply(to_list)


ma_book = ma_book.apply(sorted)

rules_sets = list(ma_book)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redundancy  = book_rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redundancy.sort_values('lift',ascending=False).head(10)

rules_no_redundancy

# In this we have tried various values of the support and confidence with various thresholds.
# And finally got the best 26 rules which can be used for our improvements.

#Plotting the graph for rules

import matplotlib.pyplot as plt

plt.scatter(rules_no_redundancy.support,rules_no_redundancy.confidence);plt.xlabel("Support");plt.ylabel("Confidence")
