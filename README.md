# DecisionTree
ID3_implement_prototype


This is a ID3 algorithm python implementation, its a prototype for the next step in creating RandomForests of these trees, with pruning of them enabled, 
This current version does not work with continious attribute values, it only works with categorical values. 
The inputs are fetched from a excel spreadsheet as a pandas dataframe using read_csv(), and then converted into suitable inputs as tuples using ittertuples(name='', index=False)
and a class is given called classify which takes as input the constructed tree which it refers to while trying to classify the given input tuple or tuples 
