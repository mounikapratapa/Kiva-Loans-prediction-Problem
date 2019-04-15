import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

dataset = pandas.read_excel('file:///C:/Users/mouni/Downloads/kiva_fin.xlsx')
pandas.set_option('display.max_columns', None)
print(dataset.shape)
print(dataset.head(3))
#print(dataset.describe())
df = dataset.copy()
df['borrower_genders'] = df['borrower_genders'].map({'female': 1, 'male':1})
df.fillna({'borrower_genders':0}, inplace= True)
df['sector'] = df['sector'].map({'Agriculture':1, 'Food': 2, 'Housing': 3, 'Services':4, 'Arts':5, 'Retail':6, 'Education':7, 'Clothing':8, 'Personal Use':9, 'Manufacturing':10, 'Transportation':11, 'Construction':12, 'Health':13,'Entertainment':14, 'Wholesale':15})
df.fillna({'sector':16}, inplace = True)
count = df['repayment_interval'].value_counts().head(10)
df['repayment_interval'] = df['repayment_interval'].map({'bullet':1, 'monthly':2, 'irregular':3})
df.fillna({'repayment_interval':4}, inplace =True)
df['term_in_months'] = df['term_in_months'].map({43: 1, 44: 1,4:2, 6:2, 14:2, 8:2, 15:2 })
df.fillna({'term_in_months':2}, inplace =True)
df['lender_count'] = df['lender_count'].map({9: 1, 10: 2, 11:3, 6:4, 7:5, 12:6 })
df.fillna({'lender_count':7}, inplace =True)
print(df.head(10))
array = df.values
X = array[:, 1:6]
print(X)
Y = array[:, 8]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
print(Y)

validation_size = 0.20
scoring = 'accuracy'
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=1)
X_training1, X_val1, Y_training1, Y_val1 = model_selection.train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
model1 = RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=10, criterion='entropy')
model2 = DecisionTreeClassifier()
model1.fit(X_training1, Y_training1)
model2.fit(X_training1, Y_training1)
preds1 = model1.predict(X_val1)
preds2 = model2.predict(X_val1)
test_preds1 = model1.predict(X_test)
test_preds2 = model2.predict(X_test)
stacked_predictions = np.column_stack((preds1, preds2))
X = pandas.DataFrame(stacked_predictions)
print(X.count())
test_stacked_predictions = np.column_stack((test_preds1, test_preds2))
print(test_stacked_predictions)
# baseline model
def model():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=2, kernel_initializer='glorot_uniform', activation='relu'))
	model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
meta_model = KerasClassifier(build_fn=model, epochs=10, batch_size=5, verbose=0)
meta_model.fit(stacked_predictions, Y_val1)
Y_pred1 = meta_model.predict(test_stacked_predictions)
kfold = model_selection.KFold(n_splits=10,random_state=2)
cv_results = model_selection.cross_val_score(meta_model,stacked_predictions, Y_val1, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % ('Output', cv_results.mean(), cv_results.std())
print(msg)
print('accuracy_score:', accuracy_score(Y_test, Y_pred1, normalize= True))
print('f1_score:', accuracy_score(Y_test, Y_pred1, normalize= True))
print('Accuracy of classifier on training set: {:.2f}'.format(model1.score(X_train, Y_train)))
print('Accuracy of classifier on test set: {:.2f}'.format(model1.score(X_test, Y_test)))
labels = [1,0]
cm = confusion_matrix(Y_test, Y_pred1, labels)
print('confusionmatrix:' ,cm)
