import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("Chicago_Crimes_2012_to_2017.csv")
#print(data.columns)

data.drop(["ID", "Case Number", "Description", "FBI Code", "Updated On", "Year", "Date", "Block",
           'Location Description', 'X Coordinate', 'Y Coordinate',  'Community Area', 'Location',
           'IUCR', "Longitude", "Y Coordinate", "X Coordinate", "Latitude"], inplace=True, axis=1)

#print(data.isnull().sum())

nan_value = float("NaN")
data.replace("", nan_value, inplace=True)
data.dropna(subset=["District", "Ward"], inplace=True)
data.drop_duplicates()

le = LabelEncoder()
le.fit(data["Arrest"])
data["Arrest"] = le.transform(data["Arrest"])

data["Primary Type"] = le.fit_transform(data["Primary Type"])

# print(data["Arrest"])

x = data.drop(["Arrest"], axis=1)
y = data["Arrest"]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3)

rf = RandomForestClassifier(random_state=1)
lr = LogisticRegression(random_state=0)
gb = GradientBoostingClassifier(n_estimators=10)
dt = DecisionTreeClassifier(random_state=0)
sv = svm.SVC()
nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=0)

rf.fit(x_train, y_train)
lr.fit(x_train, y_train)
gb.fit(x_train, y_train)
dt.fit(x_train, y_train)
sv.fit(x_train, y_train)
nn.fit(x_train, y_train)

rf = rf.predict(x_test)
lr = lr.predict(x_test)
gb = gb.predict(x_test)
dt = dt.predict(x_test)
sv = sv.predict(x_test)
nn = nn.predict(x_test)

print('RandomForest', accuracy_score(y_test, rf))
print('LogisticRegression', accuracy_score(y_test, lr))
print('GradientBoostingClassifier', accuracy_score(y_test, gb))
print('DecisionTree', accuracy_score(y_test, dt))
print('SVM', accuracy_score(y_test, sv))
print('NeuralNetwork', accuracy_score(y_test, nn))


#RandomForest 0.8506105481943362
#LogisticRegression 0.780852169394648
#GradientBoostingClassifier 0.8503507404520655
#DecisionTree 0.8061834242660432
#SVM 0.780852169394648
#NeuralNetwork 0.780852169394648
