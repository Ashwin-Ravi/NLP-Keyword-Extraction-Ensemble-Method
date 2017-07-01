#Python Version: 2.7
#Instructions to Execute:
#make sure BeautifulSoup4, praw, nltk has been installed before running the program
#if not, install using the pip install commands (Example : 'pip install BeautifulSoup4')

#Importing necessary libraries
from urllib2 import Request, urlopen
import urllib2
from time import sleep
from bs4 import BeautifulSoup
import praw
from praw.models import MoreComments
import thread
from multiprocessing.pool import ThreadPool
import nltk
from nltk.corpus import stopwords
import collections
from nltk.tokenize import word_tokenize
import re
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
import numpy as np

#Importing necessary files
import Mytfidf
import rake
import AKE
nltk.download('stopwords')


#Function to read data from an url and return data
def urlRead(requested):
    print 'reading data from \'https://www.reddit.com/r/all/\'  ...\n'
    try:
        response = urlopen(requested)
        data = response.read()
    except:
        sleep(7)
        response = urlopen(requested)
        data = response.read()
    return data


#Defining url and headers
headers = {'User-agent':'WeekendProject'}
URL = 'https://www.reddit.com/r/all/'

#Requesting data from the url
request = Request(URL, None, headers)
data = urlRead(request)

#Filtering out only posts from the data using BeautifulSoup
#Filters out data of class = "first" - class of posts
soup = BeautifulSoup(data, 'html.parser')
commentLinks = soup.find_all("li", class_="first")

#Filtering details of Top 10 posts
topTitles = soup.find_all("a", class_=["title may-blank ", "title may-blank outbound"])[:10]

#saving Comment url of each post for future use
seed = 'https://www.reddit.com'
commentLinkList = []
for eachObject in commentLinks:
    links = eachObject.find("a", class_="bylink comments may-blank")
    if (links.get('href'))[0:4]=='http':
        currentLink = (links.get('href'))
    else:
        currentLink = seed + (links.get('href')) #internal reddit links will not have beginning part of the link(seed)
    commentLinkList.append(currentLink)
    

#**************************************************************************************************************

#Function to return - Top comments and Top Keywords of the subreddit link.
def getCommentsAndKeywords(commentLinkComment):
    
    #Using python Reddit API Wrapper (praw) to retrieve comments from
    #//using praw due to its efficiency and ease of use//
    #Defining user details to access reddit using praw
    myuser_agent = 'Enter myuser_agent'
    myclient_secret = 'Enter your myclient_secret'
    myusername = 'Enter your myusername'
    mypassword = 'Enter your mypassword'
    myclient_id = 'Enter your myclient_id'
    reddit = praw.Reddit(user_agent = myuser_agent,
                         client_id = myclient_id, client_secret = myclient_secret,
                         username = myusername, password = mypassword)

    #Retrieve comments from url
    submission = reddit.submission(url = commentLinkComment)

    #Replaces/Removes "More comments" from the data obtained
    submission.comments.replace_more(limit=0)

    #Storing Comment body and points in dictionary
    commentDict = {}
    for comment in submission.comments.list():
        commentDict[comment.body] = comment.score

    #Sorting and obtaining the top 10 comments by points
    TopComments = (sorted(commentDict, key=commentDict.__getitem__, reverse=True))[:10]

    #Obtaining whole comment text
    commentFull=''
    for comment in submission.comments.list():
        commentFull = commentFull + '\n' + comment.body
    commentFull = commentFull.lower()
    


    return (TopComments, commentFull)


def preprocessing(commentFull):
    #Removing the most frequent english words ex: 'the', 'a', 'an'
    #s=set(stopwords.words('english'))
    myfile = open('stopwords.txt')
    stopwords = myfile.read().split()
    myfile.close()
    
    #removing URLs from text
    commentFull = re.sub(r'http\S+', '', commentFull)

    #Lemmatizing the text
    lem = WordNetLemmatizer()
    commentFull = lem.lemmatize(commentFull, "v")
    tokens = filter(lambda word: not word in stopwords,commentFull.split())

    return tokens

#Using MultiThreading to obtaing Top 10 comment and keywords
pool = ThreadPool(processes=1)
Thread = []
for j in range(0,10):
    Thread.append(pool.apply_async(getCommentsAndKeywords, (commentLinkList[j], )))

print 'Reading comments from top 10 reddit posts....\n'

completeComment = []
#Getting results from all threads
Thread_result = []
for j in range(0,10):
    Thread_result.append(Thread[j].get())
    completeComment.append((Thread_result[j])[1])

print 'Obtaining top 50 words from TF IDF....\n'
#getting top 50 comments from TF IDF algorithm (code written in Mytfidf.py)
topNoOfComments = 50
tfidfresults = Mytfidf.runmytfidf(completeComment,topNoOfComments)

#getting top 50 comments from RAKE (code in rake.py)
topwordsRAKE = []
print 'Obtaining top 50 words from RAKE....\n'
minlenth = 4
mintimes = 5
maxnoofphrases = 2
for i in range(0,10):
    rake_object = rake.Rake("SmartStoplist.txt",minlenth,mintimes,maxnoofphrases)
    joinedComment=' '
    keywords = rake_object.run(joinedComment.join(preprocessing(((Thread_result[i])[1]))))
    topwordsRAKE.append(keywords[:50])


#getting top 50 comments from AKE (code in AKE.py)
print 'obtaining top 50 words from running Automatic keyword Extraction - textrank '
topwordsAKE=[]
for j in range(0,10):
    result = AKE.score_keyphrases_by_textrank((Thread_result[j])[1])
    temptop=[]
    for i in range(0,50):
        try:
            temptop.append((result[i])[0])
        except:
            continue
    topwordsAKE.append(temptop)


#Ensemble technique - Weighted majority voting
#Higher weitage is given to top ranked words
print 'Running Ensemble technique - Weighted majority voting'
topwordsEnsemble = []
for i in range(0,10):
    candidate_score = {}
    cadidateIDF = tfidfresults[i]
    cadidateRAKE = topwordsRAKE[i]
    cadidateAKE = topwordsAKE[i]
    for candidate in cadidateIDF:
        candidate_score.setdefault(candidate, 0)
        candidate_score[candidate] = candidate_score[candidate] + (50 - cadidateIDF.index(candidate))

    for candidate in cadidateRAKE:
        candidate_score.setdefault(candidate, 0)
        candidate_score[candidate] = candidate_score[candidate] + (50 - cadidateRAKE.index(candidate))

    for candidate in cadidateAKE:
        candidate_score.setdefault(candidate, 0)
        candidate_score[candidate] = candidate_score[candidate] + (50 - cadidateAKE.index(candidate))

    sortedCandidates = sorted(candidate_score.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]
    candidateList = []
    for i in range(0,10):
        candidateList.append((sortedCandidates[i])[0])
    topwordsEnsemble.append(candidateList)



    
#Printing the results
rank = 0
for item in topTitles:
    rank=rank+1
    print "Item No.: " + str(rank)
    print item.text
    if (item.get('href'))[0:4]=='http':
        print "url: " + (item.get('href'))
    else:
        print "url: " + seed + (item.get('href'))
    print "Comment url: " + commentLinkList[rank-1]

    print "Top Comments: "
    for i in range(0,10):
        print str(i+1) + ': ',
        print ((Thread_result[rank-1])[0])[i]
    print '\n'
    print "Top Key Words (TF IDF): "
    for i in range(0,10):
        print str((tfidfresults[rank-1])[i].encode('utf-8')),
        print ', ',
    print '\n\n'
    print "Top Key Words (RAKE): "
    for i in range(0,10):
        try:
            print str((topwordsRAKE[rank-1])[i].encode('utf-8')),
            print ', ',
        except:
            print ', ',
    print '\n\n'
    print "Top Key Words (AKE): "
    for i in range(0,10):
        try:
            print str((topwordsAKE[rank-1])[i].encode('utf-8')),
            print ', ',
        except:
            print ', ',
    print '\n\n'
    print "Top Key Words (Ensemble Technique): "
    for i in range(0,10):
        print str((topwordsEnsemble[rank-1])[i].encode('utf-8')),
        print ', ',
    print '\n\n'

