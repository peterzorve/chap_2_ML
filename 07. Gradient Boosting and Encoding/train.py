
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import data_handler as dh

X_train, X_test, y_train, y_test = dh.file_handle("./insurance.csv")


def train_model(X_train, X_test, y_train, y_test): 
     X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test

     """ Define and Fit  the Models   """
     dt_model = DecisionTreeRegressor(max_depth=5)
     dt_model.fit(X_train, y_train)
     dt_train_accuracy = dt_model.score(X_train, y_train) * 100
     dt_test_accuracy  = dt_model.score(X_test, y_test) * 100


     rf_model = RandomForestRegressor(n_estimators=7)
     rf_model.fit(X_train, y_train)
     rf_train_accuracy = rf_model.score(X_train, y_train) * 100
     rf_test_accuracy  = rf_model.score(X_test, y_test) * 100


     gb_model = GradientBoostingRegressor(n_estimators=200)
     gb_model.fit(X_train, y_train)
     gb_train_accuracy = gb_model.score(X_train, y_train) * 100
     gb_test_accuracy  = gb_model.score(X_test, y_test) * 100


     return dt_train_accuracy, dt_test_accuracy, rf_train_accuracy, rf_test_accuracy, gb_train_accuracy, gb_test_accuracy

dt_train_accuracy, dt_test_accuracy, rf_train_accuracy, rf_test_accuracy, gb_train_accuracy, gb_test_accuracy = train_model(X_train, X_test, y_train, y_test)

print('Decision Tree Train Accuracy : ', round(dt_train_accuracy, 1), '%')
print('Decision Tree Test Accuracy  : ',  round(dt_test_accuracy, 1), '%')

print('Random Forest Train Accuracy : ', round(rf_train_accuracy, 1), '%')
print('Random Forest Test Accuracy  : ',  round(rf_test_accuracy, 1), '%')

print('Gradient Boosting Train Accuracy : ', round(gb_train_accuracy, 1), '%')
print('Gradient Boosting Test Accuracy  : ',  round(gb_test_accuracy, 1), '%')

