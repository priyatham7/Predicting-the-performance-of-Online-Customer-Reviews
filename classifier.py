import numpy
import matplotlib.pyplot as plt
import pandas
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

X = pandas.read_csv('afterextraction.csv')

X = X[0:6550]

X = X.reindex(numpy.random.permutation(X.index))
#X.iloc[numpy.random.permutation(len(X))]
#shuffle(X)

X.drop(['REVIEW_TEXT'],axis=1,inplace=True)

#print(X)
#store class label in Y

y = X.pop('CLASS')  

#print(X)
X_train, y_train = X[2000:], y[2000:]
X_test, y_test = X[:2000], y[:2000]
#print(y_test)
# fit regression modez
#print(y)

parameters = {'n_estimators': 2500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01,'loss':'deviance'}
gbc = ensemble.GradientBoostingClassifier(**parameters)

rfc = RandomForestClassifier(n_estimators= 100 ,criterion="gini",  n_jobs= 2)
rfc.fit(X_train, y_train)
print("\n")
print("random forest classifier accuracy : %.4f" % metrics.accuracy_score(y_test, rfc.predict(X_test)))
#print(metrics.accuracy_score(y_test, rfc.predict(X_test)))

print("\n")
gbc.fit(X_train, y_train)
#print(gbc.estimators_)
mse_ytest = []
for x in y_test:
	if(x==-1):
		mse_ytest.append(0)
	else:
		mse_ytest.append(1)

mse_xtest = []
for x in gbc.predict(X_test):
	if(x==-1):
		mse_xtest.append(0)
	else:
		mse_xtest.append(1)

mse = mean_squared_error(mse_ytest,mse_xtest)
#mse = mean_squared_error(y_test, gbc.predict(X_test))
#print(gbc.predict(X_test))

print("gradient boost classifier MSE: %.4f" % mse)
print("gradient boost classifier accuracy : %.4f" % metrics.accuracy_score(y_test, gbc.predict(X_test)))

#print(metrics.accuracy_score(y_test, gbc.predict(X_test)))
print("\n")
#plot training deviance

# compute test set deviance
test_score = numpy.zeros((parameters['n_estimators'],), dtype=numpy.float64)

for i, y_pred in enumerate(gbc.staged_predict(X_test)):
    test_score[i] = gbc.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title('Deviance')
plt.plot(numpy.arange(parameters['n_estimators'])+1, gbc.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(numpy.arange(parameters['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

feature_importance = gbc.feature_importances_
print(feature_importance)
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
#sorted_idx=[]
sorted_idx = numpy.argsort(feature_importance)
print(sorted_idx)
#print(sorted_idx)
pos = numpy.arange(sorted_idx.shape[0]) + .5
plt.subplot(2, 1, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
features=['rating','length','avgwordlength','avgsentlength','captialratio','question','dalechal','flesch','adj_list','noun_list','verb_list','polarity','title_polarity','keywords']
order=[]
val = []
for i in sorted_idx:
	order.append(features[i])
for i in sorted_idx:
	val.append(feature_importance[i])
print("features in ascending order")
print(order)
print("\n")
print("values of feature importance in ascending order\n")
print(val)
plt.yticks(pos, order)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()




