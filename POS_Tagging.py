# coding: utf-8


from nltk.corpus import treebank,brown
from nltk import bigrams, ngrams, trigrams
import math
import copy
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


corpus = brown.tagged_sents(tagset='universal')[:-100] 
 
tag_dict={}
word_dict={}

for sent in corpus:
    for elem in sent:
        w = elem[0]
        tag= elem[1]

        if w not in word_dict:
            word_dict[w]=0

        if tag not in tag_dict:
            tag_dict[tag]=0

        word_dict[w]+=1
        tag_dict[tag]+=1
        
        
'''-----------------Build Start, Transition and Emission Matrix-----------------'''
start={}
for i in corpus:
    if i[0] not in start:
        start[i[0][1]]= 0
for i in corpus:
    start[i[0][1]]+=1

for i in start:
    start[i]=math.log2(start[i]/len(corpus))
    
transition={}
transition1 = {}
for i in corpus:
    for j in range(len(i)-1):
        if i[j][1] not in transition1:
            transition1[i[j][1]]={}
        if i[j+1][1] not in transition1[i[j][1]]:
            transition1[i[j][1]][i[j+1][1]]=0
        transition1[i[j][1]][i[j+1][1]]+=1
transition= copy.deepcopy(transition1)        
for w1 in transition:
    tot = float(sum(transition[w1].values()))
    for w2 in transition[w1]:
        transition[w1][w2]=math.log2((0.001+ transition[w1][w2])/(0.001*len(word_dict) + tot))
        
transition_data={}
for key,value in transition.items():
    for key1,value1 in value.items():
        transition_data[(key,key1)]=value1
        
        
emission={}
for i in corpus:
    for j in range(len(i)):
        if i[j][1] not in emission:
            emission[i[j][1]]={}
        if i[j][0] not in emission[i[j][1]]:
            emission[i[j][1]][i[j][0]]=0
        emission[i[j][1]][i[j][0]]+=1
        

for w1 in emission:
    tot = float(sum(emission[w1].values()))
    for w2 in emission[w1]:
        emission[w1][w2]=math.log2((emission[w1][w2]+0.001)/(0.001*len(word_dict) + tag_dict[w1]))
    
'''---------------------------------------------------------------------'''

'''--------------------Viterbi Algorithm--------------------------------'''


test_data= brown.tagged_sents(tagset='universal')[-10:]
sentences =[]
for i in test_data:
    temp = [j[0] for j in i]
    sentences.append(temp)

average = 0    
seq = []
def find_max(i,dic):
    maxi = -999999
    
    tag=""
    for key,value in dic.items():
        if dic[key][i] > maxi:
            maxi = dic[key][i]
            tag = key
    return maxi,tag

for sent in range(len(sentences)):
    test = []
    dic={}  
    flag = 0
    for i in list(tag_dict.keys()):
        dic[i]=[]

    
    for i in sentences[sent]:
        if flag==0:
            for j in list(tag_dict.keys()):
                try:
                    prob = emission[j][i]+start[j]
                except Exception as e:
                    temp1 = math.log2(0.001/(0.001*len(word_dict) + tag_dict[j]))
                    prob = start[j] + temp1
                dic[j].append(prob)
        break
    
    max_prob, tag = find_max(0,dic)
    test.append(tag)
    
   
    
    
    flag = 0
    for i,w1 in enumerate(sentences[sent]):
        if i==0:
            continue
        for j in list(tag_dict.keys()):
            try:
                temp = transition[tag][j]
            except Exception as e:
                temp = math.log2((0.001/(0.001*len(word_dict) +float(sum(transition1[tag].values())))))
            try:
                prob_w2 = max_prob + emission[j][w1] + temp
            except Exception as e:
                temp1 = math.log2(0.001/(0.001*len(word_dict) + tag_dict[j]))
                prob_w2 = max_prob + temp1 + temp 
            dic[j].append(prob_w2)
            
        max_prob, tag = find_max(i,dic)
        test.append(tag)
    
    seq.append(test)
            
        
actual_tag=[]
for sent in test_data:
    temp=[]
    for word in sent:
        temp.append(word[1])
    actual_tag.append(temp)
  
'''-------------------------------CRF----------------------------'''

train_sents= corpus

def word2features(sent,i):
    word = sent[i][0]
    
    features ={
    'bias': 1.0,
    'word.lower()': word.lower(),
    'word[-3:]': word[-3:],
    'word[-2:]': word[-2:],
    'word.isupper()': word.isupper(),
    'word.istitle()': word.istitle(),
    'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word_prev = sent[i-1][0]
        features.update({
            '-1:word.lower()': word_prev.lower(),
            '-1:word.istitle()': word_prev.istitle(),
            '-1:word.isupper()': word_prev.isupper(),
        })
    else:
        features['start_sentence'] = True

    if i < len(sent)-1:
        word_after = sent[i+1][0]
        features.update({
            '+1:word.lower()': word_after.lower(),
            '+1:word.istitle()': word_after.istitle(),
            '+1:word.isupper()': word_after.isupper(),
        })
    else:
        features['end_sentence'] = True
                
    return features

def sent2features(sent):
    return [word2features(sent,i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for i,label in sent]


X_train=[sent2features(s) for s in train_sents]
y_train=[sent2labels(s) for s in train_sents]

X_test=[sent2features(s) for s in test_data]
y_test=[sent2labels(s) for s in test_data]


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs', 
    c1=0.1, 
    c2=0.1, 
    max_iterations=100, 
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)
labels=list(crf.classes_)

sorted_labels = sorted(
    labels, 
    key=lambda name: (name[1:], name[0])
)

print('Number of test sentences used = 10')
print('----------------------Viterbi Results---------------------------')
print('Viterbi Accuracy Score :',metrics.flat_f1_score(actual_tag, seq,average='weighted', labels=labels))
print(metrics.flat_classification_report(
    actual_tag, seq, labels=sorted_labels, digits=3
))
print('------------------------CRF Results-----------------------------')
print('CRF Accuracy Score :',metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

