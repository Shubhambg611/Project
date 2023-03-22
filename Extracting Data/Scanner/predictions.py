#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy 
import re
import string 
import warnings
warnings.filterwarnings('ignore')

#load model NER
model_ner=spacy.load('C:/Users/Shubham/clg project/1_bisscardNER/Version_2/output/model-best/')

def cleanText(txt):
    whitespace = string.whitespace
    punctuation = "!#$%&\'()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('','',whitespace)
    tablePunctuation = str.maketrans('','',punctuation)
    text = str(txt)
    text = text.lower()
    removewhitespace = text.translate(tableWhitespace)
    removepunctuation = removewhitespace.translate(tablePunctuation)
    
    return str(removepunctuation)

#group the labels 
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self,text):
        if self.text == text:
            return self.id
        else:
            self.id +=1
            self.text = text
            return self.id
        


def parser(text,label):
    if label == 'PHONE':
        text = text.lower()
        text = re.sub(r'\D','',text)
        
    elif label == 'EMAIL':
        text = text.lower()
        allow_special_char = '@)_.\-'
        text = re.sub(r'[^A-za-z0-9{}]' .format(allow_special_char),'',text)
        
    elif label == 'WEB':
        text = text.lower()
        allow_special_char = ':/.%\-'
        text = re.sub(r'[^A-za-z0-9{}]' .format(allow_special_char),'',text)
        
    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]','',text)
        text = text.title()
                   
    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9]','',text)
        text = text.title()
        
    elif label == 'ADD':
        text = text.lower()
        text = re.sub(r'[^a-z0-9]','',text)
        text = text.title()
        
    return text

grp_gen = groupgen()

def getPredictions(image):
    tessData = pytesseract.image_to_data(image)
    tesslist = list(map(lambda x:x.split('\t'), tessData.split('\n')))
    df = pd.DataFrame(tesslist[1:],columns=tesslist[0])
    df.dropna(inplace=True) #drop missing values
    df['text'] = df['text'].apply(cleanText)

    df_clean = df.query('text != "" ')
    content = " ".join([w for w in df_clean['text']])
    print(content)

    #get predictions
    doc = model_ner(content)


    #converting doc into json
    docjson = doc.to_json()
    doc_text = docjson['text']


    datafram_tokens = pd.DataFrame(docjson['tokens'])
    datafram_tokens['tokens'] = datafram_tokens[['start','end']].apply(lambda x:doc_text[x[0]:x[1]],axis = 1)

    right_table = pd.DataFrame(docjson['ents'])[['start','label']]
    datafram_tokens = pd.merge(datafram_tokens,right_table,how='left',on='start')
    datafram_tokens.fillna('O',inplace=True)


    #joining lable to df_clean dataframe
    df_clean['end'] = df_clean['text'].apply (lambda x: len(x)+1).cumsum()- 1
    df_clean['start'] = df_clean[['text','end']].apply(lambda x: x[1] - len(x[0]),axis=1)

    #inner join with start 
    dataframe_info = pd.merge(df_clean,datafram_tokens[['start','tokens','label']],how='inner',on='start')



    #bounding box

    bb_df = dataframe_info.query(" label !='O' ")

    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)


    #create right and bottom of bonding box
    bb_df[['left','top','width','height']] = bb_df[['left','top','width','height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']

    #tagging : groupby group
    col_group = ['left','top','right','bottom','label','tokens','group']
    group_tag_img = bb_df[col_group].groupby(by='group')

    img_tagging = group_tag_img.agg({

        'left':min,
        'right':max,
        'top':min,
        'bottom':max,
        'label':np.unique,
        'tokens':lambda x: " ".join(x)
    })


    img_bb = image.copy()
    for l,r,t,b,label,tokens in img_tagging.values:
        cv2.rectangle(img_bb,(l,t),(r,b),(0,255,0),2)

        cv2.putText(img_bb,str(label),(l,t),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)


    # entities
    info_array = dataframe_info[['tokens','label']].values
    entities = dict(NAME=[],ORG=[],DES=[],ADD=[],PHONE=[],EMAIL=[],WEB=[])
    previous = 'O'

    for tokens, label in info_array:
        #print(tokens,label)
        bio_tag = label[:1]
        label_tag = label[2:]

        #step 1 parse the tokens
        text = parser(tokens,label_tag)

        if bio_tag in ('B','I'):

            if previous != label_tag:
                entities[label_tag].append(text)

            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)

                else: 
                    if label_tag in ("NAME",'ORG','DES','ADD'):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text

                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text

        previous = label_tag
        
        
    return img_bb, entities





