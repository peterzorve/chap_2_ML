
# model = ""
# while True:

#     age = int(input("How old are you? \n"))
#     child = int(input("How many children do you have? \n"))
#     smoke = bool(input("Do you smoke? \n"))
#     '''
#     Preprocess
#     predict
    
#     '''
#     print("You are too fucked up 1 milly")



from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import train as trn 
import data_handler as dh

X_train, X_test, y_train, y_test = dh.file_handle("./insurance.csv")
trn.train_model(X_train, X_test, y_train, y_test)

dt_model = DecisionTreeRegressor(max_depth=5)
# dt_model.fit(X_train, y_train)

predict = dt_model.score(X_test, y_train)

print(predict)




