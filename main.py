from sklearn import tree

# # [Height , Weight , Shoe size]
# X = [ [190,90,33] , [160,60,23] , [180,80,43] ,[165,65,39] , [182,85,41] ]
# # label associated with above data set in above list 
# Y = ["male","female","male","female","female"]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# store decision tree classifier
clf = tree.DecisionTreeClassifier()

# fit method trains decision tree on our data set
clf = clf.fit(X,Y)

prediction = clf.predict([[189,34,45]])
print(prediction)