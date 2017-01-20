from __future__ import division
import re
import math
import numpy
import nltk
import pandas
from nltk.corpus import sentiwordnet 
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import *
numbered = 6550



def  DaleChall( text, locale='en_GB', simplewordlist=[]):
        
        
        min_age = 0
	value = 0
        finalvalue = 0
        scores = valuesofscores(text, locale, simplewordlist)

        dw = scores['totalwords'] -scores['simpletotalwords']

        dw_perc = dw*100 / scores['totalwords'] 

        value =  (0.0496 * scores['avgsentlength'])+ (0.1579 * dw_perc) 
        if dw_perc > 5:
            value =3.6365+value 

        if value <= 5.9:
            finalvalue = value
	elif value > 8.9 and value <= 9.9:
            finalvalue = 4+value + (value - int(value))
	elif value > 7.9 and value <= 8.9:
            finalvalue = 3+value 
        elif value > 6.9 and value <= 7.9:
            finalvalue = 2+value 
	elif value > 5.9 and value <= 6.9:
            finalvalue = 1+value 
        else:
            finalvalue = value + 6

        return int(round(finalvalue + 5))


			

def Flesch(text, locale='en_GB'):
        
        scores = valuesofscores(text, locale)
        return  206.835 - ( 1.015 * scores['avgsentlength'] ) - ( 84.6 * scores['avgwordlength'] )


# creating target class
def Create_target(reviews):
    
    rd=reviews['HELPFUL']
    c=0
    H = []
    for votes in rd:
		if c<numbered :   		
			#print (helpful)
			splitnos = votes.split(',')
			#print (helpful_votes)
			a=float(splitnos[0][1:])
			b=float(splitnos[1][:-1])
			hr=a/b;
			#for helpful ones
			if(hr > 0.7):
				H.append(1)
			# for unhelpful ones
			else:
				H.append(-1)
			c=c+1
		else:
			break
    return H


# for counting nouns , verbs , adjectives using pos_tag

def valuesofscores(text, locale='en_GB', simplelist=[]):

    sentences = sent_tokenize(text)
    textfeat= {'totalsent': 0,'totalwords': 0,'letter_count':0,'totalsyllables': 0,'simpletotalwords': 0,'avgsentlength': 0,'avgwordlength': 0}
    textfeat['totalsent'] = len(sentences)

    for s in sentences:
        words = re.findall(r'\w+', unicode(s.decode('utf-8')), flags = re.UNICODE)
        textfeat['totalwords'] = textfeat['totalwords'] + len(words)
        for w in words:
            sc =0
            textfeat['letter_count'] = len(w)+textfeat['letter_count'] 
            textfeat['totalsyllables'] =sc+textfeat['totalsyllables'] 
                 
            if simplelist and w.lower() in simplelist:
                    textfeat['simpletotalwords'] = textfeat['simpletotalwords'] + 1
    
    if textfeat['totalsent'] > 0:
        textfeat['avgsentlength'] = textfeat['totalwords'] / textfeat['totalsent']

    if textfeat['totalwords'] > 0:
        textfeat['avgwordlength'] = textfeat['totalsyllables'] / textfeat['totalwords']
    
    return textfeat



def count_pos(reviews):
	c=0	
	nounl=[]
	verbl=[]
	adjl=[]
	text_reviews = reviews['REVIEW_TEXT']
	for eachreview in text_reviews:
		if(c<numbered):
			adjc = 0
			verbc = 0
			nounc = 0
			taggedsent = pos_tag(word_tokenize(eachreview))
			#checking for adjective , noun ,verb
			for word in taggedsent:
			    if word[1].startswith('JJ') :
				adjc += 1
			    elif word[1].startswith('NN') :
				nounc+= 1
			    elif word[1].startswith('VB') :
				verbc += 1
			    
			c+=1
			#print(c)
			adjl.append(adjc)
			nounl.append(nounc)
			verbl.append(verbc)
		else:
			break
	
	return adjl, nounl, verbl

def polarity_calculate(doc):
	reviewsent = nltk.sent_tokenize(doc)
	#print(sentences)
	sentwords = [nltk.word_tokenize(sent) for sent in reviewsent]
	#print(stokens)
	taggedsentenceslist=[]
	for sentword in sentwords:        
	     taggedsentenceslist.append(nltk.pos_tag(sentword))
	wnl = nltk.WordNetLemmatizer()
	#print(taggedsentenceslist)
	values_list=[]
	for i,tsent in enumerate(taggedsentenceslist):
	    values_list.append([])
	    #print(i)
	    #print(tsent)
	    for j,t in enumerate(tsent):
		sym=''
		lemmatized=wnl.lemmatize(t[0])
		if t[1].startswith('NN'):
		    sym='n'
		elif t[1].startswith('V'):
		    sym='v'
		elif t[1].startswith('JJ'):
		    sym='a'
		elif t[1].startswith('R'):
		    sym='r'
		else:
		    sym=''       
		if(sym!=''):    
		    multiwords = list(sentiwordnet.senti_synsets(lemmatized, sym))
		    #print("statements")
		    #print(synsets)      
		    value=0
		    if(len(multiwords)>0):
		        for word in multiwords:
		            value+=word.pos_score()-word.neg_score()
		        values_list[i].append(value/len(multiwords))
		    
	#print(score_list)
	sentpol=[]
	
	
	for valuesent in values_list:
	    if(len(valuesent)>0):
	    	sentpol.append(sum(valuesent)/len(valuesent))

	return sum(sentpol)


def polarity_values(reviews):
	polarity=[]
	c=0
	text_reviews = reviews['REVIEW_TEXT']
	for each_review in text_reviews:
		if(c<numbered):
			pol_value = polarity_calculate(each_review)
			polarity.append(pol_value)
			c+=1
		else:
			break
	return polarity


def title_polarity(reviews_data):
	Polarity=[]
	c=0
	text_reviews = reviews_data['REVIEW_SUMMARY']
	for each_review in text_reviews:
		if(c<numbered):
			pol_value =polarity_calculate(each_review)
			Polarity.append(pol_value)
			c+=1
		else:
			break
	return Polarity


#brandtext = tp.get_txt_data("product_brand.txt", "lines")

def brand(reviews):
	keycount=[]
	c=0
	keywords = []
	keywordsfile =  open("specs.txt","r") 
	
    	for each in keywordsfile:
    		keywords.append(str(each.strip("\n")))
	# for removing extra characters
	keywords = [x.rstrip() for x in keywords]
	    
	text_reviews = reviews['REVIEW_TEXT']
	keywords = [x.lower() for x in keywords]
	
	#print(keywords)	
	
	for eachreview in text_reviews:
		if(c<numbered):
			count=0
			words = word_tokenize(eachreview)
			words = [item.lower() for item in words]
			for word in words:
				if word in keywords:
					count+=1
			keycount.append(count)
			c+=1
		else:
			break
	return keycount

def dalechall_values(reviews_data):
	awordlist=open('simplewords.txt').read()
	Dalechal=[]
	c=0
	text_reviews = reviews_data['REVIEW_TEXT']
	for each_review in text_reviews:
		if(c<numbered):
			#print(c)
			#print(each_review)
			if(c==863):
				#print("hello world")
				Dalechal.append(0)
			else:
				dc = DaleChall(each_review, simplewordlist=awordlist, locale='de_DE')		
				Dalechal.append(dc)
			c+=1
		else:
			break
	return Dalechal
def flesch_values(reviews_data):
	flesch=[]
	c=0
	text_reviews = reviews_data['REVIEW_TEXT']
	for each_review in text_reviews:
		if(c<numbered):
			dc = Flesch(each_review,locale='de_DE')		
			flesch.append(dc)
			c+=1
		else:
			break
	return flesch
	

def eliminate(review):
    
    pre =  open("inp.txt","r")  
    agil = []                                       
                                   
    for xy in pre:
        agil.append(str(xy.strip("\n")))    
    finallist = [] 
    solu = str(review).decode('utf-8',errors='ignore')
    result = re.findall(r"\b[a-zA-Z']+\b", solu)
    temper = []
    for every in result:
        temper.append(str(every))
    big =  map(str.upper,agil)

    for ab in temper:
        if (ab.lower() not in agil and len(ab)>1 and ab not in big): 
            finallist.append(ab)
    return finallist

def properties(reviews):
    opinions = reviews['REVIEW_TEXT']
    reviewlengths = []
    avgwordlengths = []
    c=0
    avgsentlengths = []
    capsrat = []
    qer = []
    
    for eachopinion in opinions:
	if(c<numbered):       
		wl = []
		sl = []
		warray = eliminate(eachopinion)
		capscount = 0
		qec = 0

		for every in warray:
		    wl.append(len(every))
		    if(str(every).isupper() == True):
		        capscount+=1

		reviewlengths.append(len(str(eachopinion).lower()))
		for ch in str(eachopinion):
		    if(ch == "!" or ch == "?"):
		        qec+=1
		
		sentences = sent_tokenize(str(eachopinion),language='english')
		for everyc in sentences:
		    sl.append(len(everyc))

		
	    	if(len(sl)>0):
	    		avgsentlengths.append(sum(sl)/float(len(sl)))	
	    	else:
			avgsentlengths.append(0)
	        if(len(wl)>0):
		    	avgwordlengths.append(sum(wl)/float(len(wl)))
	        else:
			avgwordlengths.append(0)
	        if(len(warray)>0):
	    		capsrat.append(capscount/float(len(warray)))
	        else:
			capsrat.append(0)
	        if(len(str(eachopinion))>0):
	    		qer.append(qec/float(len(str(eachopinion))))
	        else:
			qer.append(0)
		c+=1
	else: 
		break

    return reviewlengths , avgwordlengths , avgsentlengths , capsrat, qer


# reading data

data_frame = pandas.read_csv('reviews.csv') 

print 'generating features wait for some time about 5 min'

#  getting review ratings 

Review_Score_S = pandas.Series(data_frame['RATINGS'].values)

# getting text features like  Review_Length , Avg_Word_Length, Avg_Sentence_Length, Capital_Words_Ratio,Question_Exlamation_Ratio 

Review_Length, Avg_Word_Length, Avg_Sentence_Length, Capital_Words_Ratio, Question_Exlamation =properties(data_frame)

Review_Length_S = pandas.Series(Review_Length)
Avg_Word_Length_S = pandas.Series(Avg_Word_Length)
Avg_Sentence_Length_S = pandas.Series(Avg_Sentence_Length)
Capital_Words_Ratio_S = pandas.Series(Capital_Words_Ratio)
Question_Exlamation_S = pandas.Series(Question_Exlamation) 

# calculating dalechal values 

dalechall_values_S = pandas.Series(dalechall_values(data_frame))

# calculating flesch values

flesch_values_S = pandas.Series(flesch_values(data_frame))

# calulating  count of adjectives, nouns ,verbs

adj_list, noun_list, verb_list = count_pos(data_frame)
adj_list_S = pandas.Series(adj_list)
noun_list_S = pandas.Series(noun_list)
verb_list_S = pandas.Series(verb_list)

# getting polarity of review text
polarity = polarity_values(data_frame)
polarity_S = pandas.Series(polarity)

# getting polarity of title 
title_pol = title_polarity(data_frame)
title_pol_S = pandas.Series(title_pol)

# getting count of keywords
keywords_S= pandas.Series(brand(data_frame))

# getting target values
target_S     = pandas.Series(Create_target(data_frame))

# creating data frame of features

dataframe = pandas.DataFrame([Review_Score_S, Review_Length_S, Avg_Word_Length_S, Avg_Sentence_Length_S, Capital_Words_Ratio_S, Question_Exlamation_S,dalechall_values_S,flesch_values_S,adj_list_S,noun_list_S,verb_list_S,polarity_S,title_pol_S,keywords_S,target_S])

dataframe2 = dataframe.unstack().unstack()

dataframe2.rename(columns={0:'RATINGS',1:'RW_LEN',2:'WORD_LEN',3:'SENT_LEN',4:'CAPS_RATIO',5:'QERATIO',6:'Dalechall', 7:'flesch',8:'adj_list',9:'noun_list',10:'verb_list',11:'polarity',12:'title_polarity',13:'keywords',14:'CLASS'}, inplace=True)
dataframe2[['RATINGS','RW_LEN','WORD_LEN','SENT_LEN','CAPS_RATIO','QERATIO','Dalechall','flesch','adj_list','noun_list','verb_list','polarity','title_polarity','keywords','CLASS']]=                                                                                                                                       dataframe2[['RATINGS','RW_LEN','WORD_LEN','SENT_LEN','CAPS_RATIO','QERATIO','Dalechall','flesch','adj_list','noun_list','verb_list','polarity','title_polarity','keywords','CLASS']].convert_objects(convert_numeric=True)


Reviews = data_frame['REVIEW_TEXT']

pandas.concat([Reviews,dataframe2],axis=1).to_csv('afterextraction.csv')
print 'a file named afterextraction.csv is generated plz remove 1st column from afterextraction.csv (that column is indexing column)'
