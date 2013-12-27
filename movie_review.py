import tweepy
import nltk
import pickle

ckey = 'Consumer key'
csecret = 'Consumer secret'
atoken = 'Access token'
asecret = 'Access token secret'


def get_tweets(movie):
    auth = tweepy.OAuthHandler(ckey, csecret)
    auth.set_access_token(atoken, asecret)

    api = tweepy.API(auth)

    tweets = []
    i=0

    for tweet in tweepy.Cursor(api.search, q=movie.lower(), count=100, result_type="recent", include_entities=True, lang="en").items():
        tweets.append(tweet.text)
        i+=1
        if i==100:
            break

    return tweets

def get_features(tweet):
    stop = nltk.corpus.stopwords.words('english')
    words = [w for w in tweet if w not in stop]
    f = {}
    for word in words:
        f[word] = word
    return f

if __name__=="__main__":
    movie = raw_input("Enter movie name: ")
    tweets = get_tweets(movie)

    f= open('bayes.pickle')
    classifier = pickle.load(f)

    p=0.0
    n=0.0

    for t in tweets:
        if classifier.classify(get_features(t.split())) == 'positive':
            #print t + "POSITIVE"
            #print classifier.prob_classify(get_features(t.split())).prob('positive')
            p+=1.0
        elif classifier.classify(get_features(t.split())) == 'negative':
            #print t + "NEGATIVE"
            #print classifier.prob_classify(get_features(t.split())).prob('negative')
            n+=1.0

    print "Rating as per last 100 tweets: " + str((p/(p+n))*5.0)
