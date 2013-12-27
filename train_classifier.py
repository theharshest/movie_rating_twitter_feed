import nltk
import os
import pickle

stop = nltk.corpus.stopwords.words('english')

def get_features(tweet):
    global stop
    words = [w for w in tweet if w not in stop]
    f = {}
    for word in words:
        f[word] = word
    return f

def build_train_set():
    p = []
    n = []

    global stop

    for root,dirs,files in os.walk('/home/theharshest/data_science/sentiment_analysis/pos'):
        for file in files:
            if file.endswith(".txt"):
                try:
                    f=open('/home/theharshest/data_science/sentiment_analysis/pos/' + file, 'r')
                    for line in f:
                        tmp = [w.lower() for w in line.split() if w not in stop]
                        p.append((tmp, 'positive'))
                    f.close()
                except:
                    continue


    for root,dirs,files in os.walk('/home/theharshest/data_science/sentiment_analysis/neg'):
        for file in files:
            if file.endswith(".txt"):
                try:
                    f=open('/home/theharshest/data_science/sentiment_analysis/neg/' + file, 'r')
                    for line in f:
                        tmp = [w.lower() for w in line.split() if w not in stop]
                        p.append((tmp, 'negative'))
                    f.close()
                except:
                    continue
    
    return p+n

if __name__=="__main__":
    lst = build_train_set()

    training_set = nltk.classify.apply_features(get_features, lst)
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    f = open('bayes.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()