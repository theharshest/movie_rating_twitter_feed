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
    
    cvp=int(len(p)*0.3)
    cvn=int(len(n)*0.3)

    return ((p[:cvp]+n[:cvn]), (p[cvp:]+n[cvn:]))

if __name__=="__main__":
    cv_data, train_data = build_train_set()

    training_set = nltk.classify.apply_features(get_features, train_data)
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    p=0.0
    n=0.0

    for t in cv_data:
        if classifier.classify(get_features(t[0])) == 'positive':
	    if t[1]=='positive':
            	p+=1.0
	    else:
		n+=1.0
        elif classifier.classify(get_features(t[0])) == 'negative':
	    if t[1]=='negative':
            	p+=1.0
	    else:
		n+=1.0

    print "Efficiency using cross-validation: " + str(p/(p+n))

    f = open('bayes.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()
