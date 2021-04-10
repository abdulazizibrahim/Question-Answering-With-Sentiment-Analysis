import nltk
import sys
import os
import math
FILE_MATCHES = 1
SENTENCE_MATCHES = 1
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stop_words = set(nltk.corpus.stopwords.words('english'))
def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    #print(files)

    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    #print("filewords " , file_words['artificial_intelligence.txt'])
    file_idfs = compute_idfs(file_words)
    #print('file: ', file_idfs)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    topic = filenames[0]
    topic = topic.replace('.txt', '')
    print("Topic of Query:", topic)
    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)

    for match in matches:
        print("Answer:",match,"\n")
    print("Performing Sentiment Analysis on query.......")
    precent = SentimentAnalysis(query)
    print("Positive:", format(precent[0]*100, '.2f'),"%")
    print("Negative:", format(precent[1]*100, '.2f'), "%")

def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename),encoding="utf8") as f:
            files[filename] = f.read()
    return files
def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    newList = []
    for word in tokenizer.tokenize(document):
        if word.isalpha():
            word = word.lower()
        if word not in stop_words:
            newList.append(word)
    return newList


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    words = set()
    for filename in documents:
        words.update(documents[filename])
    idfs = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidf = dict()
    for filenames in files:
        tfidf[filenames] = 0
    for word in query:
        for filename in files:
            tf = files[filename].count(word)
            tfidf[filename] += tf * idfs[word]
    flist = list(tfidf.items())
    flist.sort(key = lambda x: x[1])
    fnlist = []
    for i in range(1, n+1):
        fnlist.append(flist[len(flist)-i][0])
    return fnlist


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    score = dict()
    scorex = dict()
    scorey = dict()
    for sentname in sentences:
        score[sentname] = 0
    for words in query:
        for sentname in sentences:
            if words in sentences[sentname]:
                score[sentname] += idfs[words]
    flist = list(score.items())
    flist.sort(key = lambda x: x[1])
    query = list(query)
    for i in range(1, 4):
        scorey[flist[-i][0]] = flist[-i][1]
    for keys in scorey:
        wc = 0
        for word in sentences[keys]:
            wc += query.count(word)
        scorex[keys] = wc / len(sentences[keys])
    fnlist = []
    falist = list(scorex.items())
    falist.sort(key = lambda x: x[1])
    for i in range(1,n+1):
        fnlist.append(falist[-i][0])
    return fnlist
def SentimentAnalysis(query):
    #test_tokens = test.split(' ')
    good = nltk.corpus.wordnet.synsets('good')
    bad = nltk.corpus.wordnet.synsets('evil')
    score_pos = score_neg = 0

    for token in query:
        t = nltk.corpus.wordnet.synsets(token)
        if len(t) > 0:
            sim_good = nltk.corpus.wordnet.wup_similarity(good[0], t[0])
            sim_bad = nltk.corpus.wordnet.wup_similarity(bad[0], t[0])
            if(sim_good is not None) :
                score_pos = score_pos + sim_good
            if(sim_bad is not None):
                score_neg = score_neg + sim_bad
    score_total = score_neg + score_pos
    #print("totaL", score_total, "pos", score_pos,"neg", score_neg)
    return [score_pos/score_total, score_neg/score_total]
if __name__ == "__main__":
    main()
