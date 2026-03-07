from Pipeline import Pipeline
from sklearn.svm import SVC 

p = Pipeline(SVC=SVC(kernel='linear', max_iter=10))
p.fit()