# Q1) 
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


# load the data 

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Unsupervised learning Association Rules\dataset\book.csv")
data.info()
data.describe()
data.columns 


# apply the apriori algorithm to find the most frequent books sells

frequent_books =  apriori(data, min_support = 0.025, max_len = 3, use_colnames = True)

# most frequent itemsets based on support in decending order

frequent_books.sort_values('support', ascending = False, inplace = True)

plt.bar(x= list(range(0,11)), height = frequent_books.support[0:11], color = 'rgmyk')
plt.xticks(list(range(0,11)), frequent_books.itemsets[0:11],rotation= 90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_books, metric = "lift", min_threshold= 1)
rules.head(20)
 rules1 = rules.sort_values("lift", ascending = False).head(20)

# profusion rules i.e to remove the redudancy in the rule 

def to_list(i):
    return(sorted(list(i)))
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy

a = rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules  
   
b = rules_no_redudancy.sort_values('lift', ascending = False)head(10)

###############################################################################


# Q2)

 # Importing necessary libraries 
 
import pandas as pd
import matplotlib.pyplot as plt 
from mlxtend.frequent_patterns import apriori, association_rules

#Declaring empty list

groceries = []

# As the file is in transaction data, we will be reading data directly

with open(r"C:\Users\praja\Desktop\Data Science\Unsupervised learning Association Rules\dataset\groceries.csv") as f:
    groceries = f.read()


# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")
groceries_list = []

#itteration of elements in groceries list

for i in groceries:
    groceries_list.append(i.split(","))
    
#For i in groceries_list:
#All_groceries_list = all_groceries_list+i
       
all_groceries_list = [i for item in groceries_list for i in item]

from collections import Counter # OrderedDict

item_frequencies = Counter(all_groceries_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[0:11], x = list(range(0, 11)), color = 'rgbkymc')
plt.xticks(list(range(0, 11), ), items[0:11], rotation = 90)
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data 
# Purpose of converting all list into Series object because to treat each list element as entire element not to separate 

groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835, :] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=90)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

# Declaring rules var for association rule

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift',  ascending = False).head(10)


#To eliminate Redudancy in Rules

def to_list(i):
    return (sorted(list(i)))

# Sorting, listing and appending

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 

a = rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 15 rules
 
b = rules_no_redudancy.sort_values('lift', ascending = False).head(15)

#############################################################################

# Q3)

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
 
# load the data 
 
movies  = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Unsupervised learning Association Rules\dataset\my_movies.csv")
movies1 = movies.iloc[:,5:]
movies1.isnull().sum()
movies1.describe()


# apply the apriori algorithm to find the most frequent movies watched 
frequent_movies = apriori(movies1 , min_support = 0.05, max_len = 3, use_colnames = True )

# most frequent movies based on the support in decending order

frequent_movies.sort_values('support', ascending = False, inplace=True)

# bar plot for top 10 movies based on support
plt.bar(x = list(range(0,11)), height =  frequent_movies.support[0:11], color ="rgmyk")
plt.xticks(list(range(0,11)), frequent_movies.itemsets[0:11], rotation= 90)
plt.xlabel("itemsets")
plt.ylabel("support")
plt.show()

movies_rules = association_rules( frequent_movies, metric ="lift",min_threshold=1)
movie_rules1 = movies_rules.sort_values('lift', ascending = False ).head(20)

# profusion rules i.e to remove the redudancy in the rule 

def to_list(i):
     return(sorted(list(i)))
 
ma_X = movies_rules.antecedents.apply(to_list) + movies_rules.consequents.apply(to_list)

ma_X1 = ma_X.apply(sorted)

rules_sets = list(ma_X1)

unique_rules_sets = [list (m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy

a = rules_no_redudancy = movies_rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules  
   
b = rules_no_redudancy.sort_values('lift', ascending = False)

3###########################################################################

# Q4)

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
 
# load the data 
 
phone  = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Unsupervised learning Association Rules\dataset\myphonedata.csv")
phone1= phone.iloc[:,3:]
phone1.isnull().sum()
phone.describe()


# apply the apriori algorithm to find the most frequent movies watched 
frequent_phone = apriori(phone1, min_support = 0.03, max_len = 3, use_colnames = True )

# most frequent movies based on the support in decending order

frequent_phone.sort_values('support', ascending = False, inplace=True)

# bar plot for  based on support
plt.bar(x = list(range(0,15)), height =  frequent_phone.support[0:15], color ="rgmyk")
plt.xticks(list(range(0,15)), frequent_phone.itemsets[0:15], rotation= 90)
plt.xlabel("itemsets")
plt.ylabel("support")
plt.show()

phone_rules = association_rules( frequent_phone, metric ="lift",min_threshold=1)
phone_rules1 = phone_rules.sort_values('lift', ascending = False ).head(20)

# profusion rules i.e to remove the redudancy in the rule 

def to_list(i):
     return(sorted(list(i)))
 
ma_X = phone_rules.antecedents.apply(to_list) + phone_rules.consequents.apply(to_list)

ma_X1 = ma_X.apply(sorted)

rules_sets = list(ma_X1)

unique_rules_sets = [list (m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy

a = rules_no_redudancy = phone_rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules  
   
b = rules_no_redudancy.sort_values('lift', ascending = False)
