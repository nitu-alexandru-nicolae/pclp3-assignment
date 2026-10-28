import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# task 6 a) & b):
# calculating the amount of missing values and getting rid of them
train_set = pd.read_csv("train_set.csv")
missing_values = train_set.isnull().sum()
missing_percentile = train_set.isnull().mean() * 100
train_set = train_set.dropna() # getting rid of na values
print(train_set.describe())
print(train_set.describe(include=['object']))
test_set = pd.read_csv("test_set.csv")
missing_values = test_set.isnull().sum()
missing_percentile = test_set.isnull().mean() * 100
test_set = test_set.dropna() # getting rid of na values
print(test_set.describe())
print(test_set.describe(include=['object']))

# task 6 c) & d):
# making every plot in existence
for column in train_set.columns:
    if (is_numeric_dtype(train_set[column])):
        sns.histplot(train_set[column], edgecolor='black')
        plt.title(f'Distribution for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
        sns.boxplot(train_set[column])
        plt.title(f'Outliers for {column}')
        plt.xlabel(column)
        plt.show()
    elif (column != 'Date & Time'):
        sns.countplot(train_set[column])
        plt.title(f'Countplot for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

for column in test_set.columns:
    if (is_numeric_dtype(test_set[column])):
        sns.histplot(test_set[column], edgecolor='black')
        plt.title(f'Distribution for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
        sns.boxplot(test_set[column])
        plt.title(f'Outliers for {column}')
        plt.xlabel(column)
        plt.show()
    elif (column != 'Date & Time'):
        sns.countplot(test_set[column])
        plt.title(f'Countplot for {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

# task 6 e) & f):
# making heatmaps for the numerical variables
numerical_variables = train_set.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numerical_variables.corr(), annot=True)
plt.show()

numerical_variables = test_set.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numerical_variables.corr(), annot=True)
plt.show()

# making every plot in existence with Temperature as the target
target = 'Temperature'
for column in train_set.columns:
    if (column == target):
        continue

    if (is_numeric_dtype(train_set[column])):
        sns.scatterplot(train_set, x=column, y=target)
        plt.title(f"Relation between {column} and {target}")
        plt.xlabel(column)
        plt.ylabel(target)
        plt.show()
    elif (column != 'Date & Time'):
        sns.violinplot(train_set, x=column, y=target)
        plt.title(f"Distribution of {target} in regards to {column}")
        plt.xlabel(column)
        plt.ylabel(target)
        if (column == 'City'):
            plt.xticks(rotation=45, fontsize=9)
        plt.show()

for column in test_set.columns:
    if (column == target):
        continue

    if (is_numeric_dtype(test_set[column])):
        sns.scatterplot(test_set, x=column, y=target)
        plt.title(f"Relation between {column} and {target}")
        plt.xlabel(column)
        plt.ylabel(target)
        plt.show()
    elif (column != 'Date & Time'):
        sns.violinplot(test_set, x=column, y=target)
        plt.title(f"Distribution of {target} in regards to {column}")
        plt.xlabel(column)
        plt.ylabel(target)
        if (column == 'City'):
            plt.xticks(rotation=45, fontsize=9)
        plt.show()

# task 7: we'll train a model to try and predict future temperatures using linear regression

# we don't want to train the model with non-numerical values or the
# temperatures themselves, but we do want the target to be the
# temperatures
useless_columns = ['Temperature', 'City', 'Date & Time', 'Weather', 'Feels Like']
x_train = train_set.drop(columns=useless_columns)
y_train = train_set['Temperature']
x_test = test_set.drop(columns=useless_columns)
y_test = test_set['Temperature']
model = LinearRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# calculating the errors
error1 = mean_absolute_error(y_test, predictions)
error2 = root_mean_squared_error(y_test, predictions)
error3 = r2_score(y_test, predictions)
print(error1)
print(error2)
print(error3)

# making the performance graph
sns.scatterplot(x=y_test, y=predictions)
min_val = min(y_test.min(), predictions.min()) # getting min and max values so
max_val = max(y_test.max(), predictions.max()) # we can draw a reference line
plt.plot([min_val, max_val], [min_val, max_val], color='red') #  on the plot
plt.title('Real vs. predicted temperatures')
plt.xlabel('Real temperatures')
plt.ylabel('Predicted temperatures')
plt.show()