# MLP for Kiva Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
import pandas
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# Function to create model, required for KerasClassifier
def create_model(optimizer='rmsprop', init='glorot_uniform'):
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
#random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = pandas.read_excel('file:///C:/Users/mouni/Downloads/kiva_fin.xlsx')
pandas.set_option('display.max_columns', None)
print(dataset.shape)
print(dataset.head(3))
#print(dataset.describe())
df = dataset.copy()
df['borrower_genders'] = df['borrower_genders'].map({'female': 1, 'male':2})
df.fillna({'borrower_genders':3}, inplace= True)
df['sector'] = df['sector'].map({'Agriculture':1, 'Food': 2, 'Housing': 3, 'Services':4, 'Arts':5, 'Retail':6, 'Education':7, 'Clothing':8, 'Personal Use':9, 'Manufacturing':10, 'Transportation':11, 'Construction':12, 'Health':13,'Entertainment':14, 'Wholesale':15})
df.fillna({'sector':16}, inplace = True)
df['repayment_interval'] = df['repayment_interval'].map({'bullet':1, 'monthly':2, 'irregular':3})
df.fillna({'repayment_interval':4}, inplace =True)
df['term_in_months'] = df['term_in_months'].map({43: 1, 44: 2, 6:3, 14:4, 8:5, 15:6 })
df.fillna({'term_in_months':7}, inplace =True)
df['lender_count'] = df['lender_count'].map({9: 1, 10: 2, 11:3, 6:4, 7:5, 12:6 })
df.fillna({'lender_count':7}, inplace =True)
print(df.head(10))
array = df.values
X = array[:, 1:6].astype(float)
print(X)
Y = array[:, 8]
print(Y)
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(encoded_Y)
model = KerasClassifier(build_fn=create_model, verbose=0)
# grid search epochs, batch size and optimizer
optimizers = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init = init)
grid = GridSearchCV(estimator=model, param_grid=param_grid, refit = False)
grid_result = grid.fit(X, encoded_Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
