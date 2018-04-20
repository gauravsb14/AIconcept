import gensim
import docx2txt
from nltk import word_tokenize
import glob


file_text_list =[]
# filedata = docx2txt.process("C:\\Users\\Gauvarav\\Desktop\\192685865_Surendra_Sah_QA.docx")
filedata = glob.glob("C:\\Users\\Gauvarav\\Desktop\\profile_resumes\\*.docx")
# print(len(filedata))
for file in filedata:
    # print(file)
    file_text = docx2txt.process(file)
    # print(file_text)
    filedata_token = word_tokenize(file_text)
    file_text_list = file_text_list + filedata_token
# print(file_text_list)
model=  gensim.models.Word2Vec([file_text_list], min_count =1, size= 32)

# to save the model
model.save("test_model")

# to load the model
model = gensim.models.Word2Vec.load("test_model")

# to get the most similar words
print(model.most_similar("java"))

# to get the vector of word
print(model.wv["testing"])