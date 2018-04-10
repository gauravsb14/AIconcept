import gensim
from nltk.tokenize import word_tokenize

#
raw_documents = ["python developer with 3years experience.has to deep learning experience will add advantage, shoud stay in bangalore or willing to reloacate have experience in pandas, numpy, skitlearn.good in ml and deep learning. deep learning experience will add advantage, shoud stay in bangalore or willing to reloacate",
                 "java professional with 5years experience.good skill on struts hibernate, shoud stay in bangalore or willing to reloacate. deep learning experience will add advantage"
                ]

# print("Number of documents:",len(raw_documents))

gen_docs = [[w.lower() for w in word_tokenize(text)]
            for text in raw_documents]
print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary)
# print(dictionary.token2id['road'])
print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    print(i, dictionary[i])
#
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)
print(tf_idf[corpus[0]])
# s = 0
# for i in corpus:
#     s += len(i)
# print(s)
sims = gensim.similarities.Similarity('F:\FugenX_Project1',tf_idf[corpus],num_features=len(dictionary))
print(sims)
print(type(sims))
#
query_doc = [w.lower() for w in word_tokenize("python developer with 3years experience.")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)

similarity_score = sims[query_doc_tf_idf]
print(list(enumerate(similarity_score)))
print(similarity_score)
#
#