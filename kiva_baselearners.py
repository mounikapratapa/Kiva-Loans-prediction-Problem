import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
dataset = pandas.read_excel('file:///C:/Users/mouni/Downloads/kiva_fin.xlsx')
pandas.set_option('display.max_columns', None)
print(dataset.shape)
print(dataset.head(3))
#print(dataset.describe())
df = dataset.copy()
df['borrower_genders'] = df['borrower_genders'].map({'female': 1, 'male':2})
df.fillna({'borrower_genders':3}, inplace= True)
df['sector'] = df['sector'].map({'Transportation' : 1, 'Agriculture':2, 'Services':3, 'Construction':4, 'Retail':5, 'Food':6, 'Arts':7, 'Education':8, 'Health':9, 'Manufacturing':10, 'Clothing':11, 'Housing':12,'Personal Use':13, 'Wholesale':14, 'Entertainment':15})
df.fillna({'sector':16}, inplace = True)
count = df['repayment_interval'].value_counts().head(10)
df['repayment_interval'] = df['repayment_interval'].map({'bullet':1, 'monthly':2, 'irregular':3})
df.fillna({'repayment_interval':4}, inplace =True)
df['term_in_months'] = df['term_in_months'].map({43: 1, 44: 2, 6:3, 14:4, 8:5, 15:6 })
df.fillna({'term_in_months':7}, inplace =True)
df['lender_count'] = df['lender_count'].map({9: 1, 10: 2, 11:3, 6:4, 7:5, 12:6 })
df.fillna({'lender_count':7}, inplace =True)
print(df.head(10))
array = df.values
X = array[:, 1:6]
print(X)
Y = array[:, 8]
print(Y)
validation_size = 0.20
scoring = 'accuracy'
seed = 2
X_train, X_test, Y_train, Y_test, = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
models = []
models.append(('RFC', RandomForestClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, Y_train)
Y_pred1 = model.predict(X_test)
print('accuracy_score:', accuracy_score(Y_test, Y_pred1, normalize= True))
print('confusionmatrix:' ,confusion_matrix(Y_test, Y_pred1))
print('Accuracy of RandomForestClassifier on training set: {:.2f}'.format(model.score(X_train, Y_train)))
print('Accuracy of RandomForestClassifier on test set: {:.2f}'.format(model.score(X_test, Y_test)))
labels = ['Low', 'High']
cm = confusion_matrix(Y_test, Y_pred1, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of Random Forest Classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

model1 = DecisionTreeClassifier()
model1.fit(X_train, Y_train)
Y_pred1 = model.predict(X_test)
print('accuracy_score:', accuracy_score(Y_test, Y_pred1, normalize= True))
print('confusionmatrix:' ,confusion_matrix(Y_test, Y_pred1))
print('Accuracy of DTC on training set: {:.2f}'.format(model1.score(X_train, Y_train)))
print('Accuracy of DTC on test set: {:.2f}'.format(model1.score(X_test, Y_test)))
labels = ['Low', 'High']
cm = confusion_matrix(Y_test, Y_pred1, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of DTC')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
