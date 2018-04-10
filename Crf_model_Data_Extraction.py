import sklearn_crfsuite
# CREATING pos DATA
data = [(["linux", "is", "my", "OS"],["OS","IR","IR","IR"]),
(["Ubuntu", "is", "my","faviourite", "OS"],["OS","IR","IR","IR", "IR"])
 ]

corpus = []
for (docs, tags) in data:
    doc_tag = []
    for word,tag in zip(docs,tags):
        doc_tag.append((word, tag))
    corpus.append(doc_tag)

print(len(corpus))

for doc in corpus:
    print(len(doc))

# DEFINITING FEATURE EXTRACTION
def doc2features(doc, i):

    # for current word
    word = doc[i][0]
    features ={
        "word.word" : word
    }

    # for previous word
    if i>0:
        prevword = doc[i-1][0]
        features["word.prevword"] = prevword
    else:
        features["BOS"]= True

#     for Next word
    if i< len(doc)-1:
        nextword= doc[i+1][0]
        features["word.nextword"]= nextword
    else:
        features["EOS"]= True
    return features

def featureExtraction(doc):
    return[doc2features(doc, i) for i in range(len(doc))]

X = [featureExtraction(doc) for doc in corpus]

print(X)

# DEFINE LABELS
def get_label(doc):
    return [tag for (token,tag) in doc]

Y = [get_label(doc) for doc in corpus]
print(Y)

# creating crf model
crf = sklearn_crfsuite.CRF(
algorithm='lbfgs',
c1 = 0.1,
c2=0.1,
max_iterations= 200,
all_possible_transitions=True
)

# TRAINING MODEL
crf.fit(X, Y)

test = [["my","OS","is","Centos"]]
X_test = featureExtraction(test)
print(crf.predict_single(X_test))