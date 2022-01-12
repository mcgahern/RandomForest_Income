def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Investigating the Data
income_data = pd.read_csv('adult.csv')
# delimter removes a leading space infront of every entry
print(income_data.head(10))
print(income_data.iloc[0])

# Changing Column Types
# random forest doesnt like columns with strings)
income_data['sex-int'] = income_data['sex'].apply(lambda row: 0 if row == 'Male' else 1)

# Looking at countries in the data set
print(income_data['native-country'].value_counts())
income_data['country-int'] = income_data['native-country'].apply(lambda row: 0 if row == 'United-States' else 1)

# Formatting for Scikit-Learn
labels = income_data[['income']]
data = income_data[['age', 'capital-gain', 'capital-loss', 'hours-per-week', 'sex-int', 'country-int']]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

# Creating Random Forest
forest = RandomForestClassifier(random_state=1)
forest.fit(train_data, train_labels)
print(forest.score(test_data, test_labels))



