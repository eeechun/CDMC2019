import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from gensim.models.word2vec import Word2Vec

def read_labels():
  print("【LABEL COLLECTING】")

  df = pd.read_csv(r"C:\Users\tammy\lab\CDMC2019\CDMC2019\CDMC2019Task2Train.csv")
  df = df.drop(["no"], axis=1)  # delete the first column
  label = df.to_numpy()      # transform into numpy array
  label.ravel()

  print("type of label")
  print(type(label))
  print("shape of label")
  print(label.shape)
  #print(label[0,])
  print("==========================================")

  return label

def read_data_str():
  print("【TRAINING DATA COLLECTING】【STR】")

  data_list = []
  for i in range(1, 4168):  #4168
    content = ""
    i = "%04d" % i
    file_dir = r'C:\Users\tammy\lab\CDMC2019\CDMC2019\TRAIN\\' 
    file_dir += str(i)
    file_dir += '.seq'

    with open(file_dir) as f:     # open each training data file
      for line in f:              # read each line in file
        temp = line.replace("\n", " ")
        content += temp

    data_list.append(content)

  data = np.array(data_list)
  data.reshape(-1, 1)

  print("type of data")
  print(type(data))
  print("shape of data")
  print(data.shape)
  #print(data[0])
  print("==========================================")

  return data

def read_data_2D():
  print("【TRAINING DATA COLLECTING】【2D】")
  data_list = []
  for i in range(1, 4168):  #4168
    content = []
    i = "%04d" % i
    file_dir = r'C:\Users\tammy\lab\CDMC2019\CDMC2019\TRAIN\\' 
    file_dir += str(i)
    file_dir += '.seq'

    with open(file_dir) as f:     # open each training data file
      for line in f:              # read each line in file
        word=""
        for alphabet in line:
          if(alphabet != " " and alphabet != "\n"):
            word+=alphabet
          else:
            content.append(word)
            word=""

    data_list.append(content)

  data = np.array(data_list)
  #data.reshape(-1, 1)

  print("type of data")
  print(type(data))
  print("shape of data")
  print(data.shape)
  print(data[0])
  print("==========================================")

  return data

def MeanEmbeddingVectorizer(model, data):
  print("【MEAN EMBEDDING VECTORIZER】")
  docVec_list=[]

  for i in data:
    count = 0.0
    docVec = 0.0

    for word in i:
      count += 1
      docVec += model.wv[word]

    docVec /= count
    docVec_list.append(docVec)

  print(docVec_list[:])
  return docVec_list

def TFIDF(data_str, labels):
  
  # load all data and labels
  tv = TfidfVectorizer()

  X = data_str[:]
  X = tv.fit_transform(X)
  X = X.toarray()
  X = np.array(X)

  #print("X after TF-IDF")
  #print(X[0])
  #print("==========================================")

  Y = labels[:]
  Y.ravel()

  # process data
  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)

  # random forest
  rf = ensemble.RandomForestClassifier(n_estimators = 100)
  rf.fit(X_train, Y_train)
  Y_predict = rf.predict(X_test)

  print("【RANDOM FOREST】【TF-IDF】")
  print(accuracy_score(Y_test, Y_predict))
  print("==========================================")

  #KNN
  knn = KNeighborsClassifier()
  knn.fit(X_train, Y_train)
  Y_predict = knn.predict(X_test)
  
  print("【KNN】【TF-IDF】")
  print(accuracy_score(Y_test, Y_predict))
  print("==========================================")

  #SVM Linear
  svm_linear = SVC(kernel='linear')
  svm_linear.fit(X_train, Y_train)
  Y_predict = svm_linear.predict(X_test)

  print("【SVM Linear】【TF-IDF】")
  print(accuracy_score(Y_test, Y_predict))
  print("==========================================")

  #SVM Poly
  svm_poly = SVC(kernel='poly')
  svm_poly.fit(X_train, Y_train)
  Y_predict = svm_poly.predict(X_test)

  print("【SVM Poly】【TF-IDF】")
  print(accuracy_score(Y_test, Y_predict))
  print("==========================================")

  #SVM RBF
  svm_rbf = SVC(kernel='rbf')
  svm_rbf.fit(X_train, Y_train)
  Y_predict = svm_rbf.predict(X_test)

  print("【SVM RBF】【TF-IDF】")
  print(accuracy_score(Y_test, Y_predict))
  print("==========================================")

def W2V(data_2D,labels):
  # load all data and labels
  X = data_2D[:]

  Y = labels[:]
  Y.ravel()

  # get word vector and document vector
  model_cbow = Word2Vec(X, sg=0)
  model_cbow.save('cbow.model')
  #model_cbow = Word2Vec.load('cbow.model')
  print(model_cbow.wv.most_similar('exit', topn=10))
  print(model_cbow.wv['exit'])
  print(model_cbow.wv['exit_group'])
  #docVecCbow = MeanEmbeddingVectorizer(model_cbow, X)
    
  #model_skip = Word2Vec(X, sg=1)
  #model_skip.save('skip.model')
  #docVecSkip = MeanEmbeddingVectorizer(model_skip, X)

  # split data into training set and testing set
  #cbowX_train, cbowX_test, cbowY_train, cbowY_test = train_test_split(docVecCbow, Y, test_size=0.2)
  #skipX_train, skipX_test, skipY_train, skipY_test = train_test_split(docVecSkip, Y, test_size=0.2)

  # linear regression
  '''
  lr = LinearRegression()
  lr.fit(cbowX_train, cbowY_train)
  cbowY_predict = lr.predict(cbowX_test)

  print("【LINEAR REGRESSION】【W2V CBOW】")
  print(accuracy_score(cbowY_test, cbowY_predict))
  print("==========================================")
  '''




data_str = read_data_str()
#data_2D = read_data_2D()
labels = read_labels()

TFIDF(data_str, labels)
#W2V(data_2D, labels)