import numpy as np
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
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
print(Y)
validation_size = 0.20
scoring = 'accuracy'
seed = 2
X_train, X_test, Y_train, Y_test, = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
models = []
models.append(('RFC', RandomForestClassifier()))
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, Y_train)
best = CV_rfc.best_params_
print(best)
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

# Calling Method 
plot_grid_search(CV_rfc.cv_results_, 'n_estimators', 'max_features', 'N- Estimators', 'Max_Features')
#model5=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 200, max_depth=10, criterion='entropy')
#model5.fit(X_train, Y_train)
#Y_pred1 = model5.predict(X_test)
#print('accuracy_score:', accuracy_score(Y_test, Y_pred1, normalize= True))
#print('confusionmatrix:' ,confusion_matrix(Y_test, Y_pred1))
#print('Accuracy of RandomForestClassifier on training set: {:.2f}'.format(model5.score(X_train, Y_train)))
#print('Accuracy of RandomForestClassifier on test set: {:.2f}'.format(model5.score(X_test, Y_test)))
#labels = ['Low', 'High']
#cm = confusion_matrix(Y_test, Y_pred1, labels)
#print(cm)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(cm)
#plt.title('Confusion matrix of Random Forest Classifier')
#fig.colorbar(cax)
#ax.set_xticklabels([''] + labels)
#ax.set_yticklabels([''] + labels)
#plt.xlabel('Predicted')
#plt.ylabel('True')
#plt.show()
#X_train = pandas.DataFrame(X_train)
#for i, j in sorted(zip(X_train.columns, model5.feature_importances_)):
    #print(i, j)
