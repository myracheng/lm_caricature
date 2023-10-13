# helper functions to compute exaggeration scores    
import sys
import pandas as pd
import numpy as np
from collections import Counter
import argparse
from collections import defaultdict
import math
from nltk.tokenize import sent_tokenize
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt

def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))

def get_list_of_sentences(target_df):
    full_list = []
    for c in target_df['response']:
        full_list.extend(sent_tokenize(c))
    return full_list

def rescale(x,sm,bi):
    return (x-sm)/(bi-sm)

def pprint(r,dic):
    
    full_list = []
    for word in sorted(dic,key=lambda x: x[1],reverse=True):
        full_list.append(word[0])
    return full_list

def mw_custom(df1,df2,df0):
    thr=1.96
    grams={}
    delt = get_log_odds(df1['response'], df2['response'],df0['response'],False) #first one is the positive-valued one

    c1 = []
    c2 = []
    for k,v in delt.items():
        if v > thr:
            c1.append([k,v])
        elif v < -thr:
            c2.append([k,v])

    if 'target' in grams:
        grams['target'].extend(c1)
    else:
        grams['target'] = c1
    return grams['target']

def get_log_odds(df1, df2, df0,verbose=False,lower=True):
    """Monroe et al. Fightin' Words method to identify top words in df1 and df2
    against df0 as the background corpus"""
    if lower:
        counts1 = defaultdict(int,[[i,j] for i,j in df1.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        counts2 = defaultdict(int,[[i,j] for i,j in df2.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        prior = defaultdict(int,[[i,j] for i,j in df0.str.lower().str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
    else:
        counts1 = defaultdict(int,[[i,j] for i,j in df1.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        counts2 = defaultdict(int,[[i,j] for i,j in df2.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])
        prior = defaultdict(int,[[i,j] for i,j in df0.str.split(expand=True).stack().replace('[^a-zA-Z\s]','',regex=True).value_counts().items()])

    sigmasquared = defaultdict(float)
    sigma = defaultdict(float)
    delta = defaultdict(float)

    for word in prior.keys():
        prior[word] = int(prior[word] + 0.5)

    for word in counts2.keys():
        counts1[word] = int(counts1[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    for word in counts1.keys():
        counts2[word] = int(counts2[word] + 0.5)
        if prior[word] == 0:
            prior[word] = 1

    n1 = sum(counts1.values())
    n2 = sum(counts2.values())
    nprior = sum(prior.values())

    for word in prior.keys():
        if prior[word] > 0:
            l1 = float(counts1[word] + prior[word]) / (( n1 + nprior ) - (counts1[word] + prior[word]))
            l2 = float(counts2[word] + prior[word]) / (( n2 + nprior ) - (counts2[word] + prior[word]))
            sigmasquared[word] =  1/(float(counts1[word]) + float(prior[word])) + 1/(float(counts2[word]) + float(prior[word]))
            sigma[word] =  math.sqrt(sigmasquared[word])
            delta[word] = ( math.log(l1) - math.log(l2) ) / sigma[word]

    if verbose:
        for word in sorted(delta, key=delta.get)[:10]:
            print("%s, %.3f" % (word, delta[word]))

        for word in sorted(delta, key=delta.get,reverse=True)[:10]:
            print("%s, %.3f" % (word, delta[word]))
    return delta


def compute_exaggeration_scores(df,emb_dict,default_persona_term,default_topic_term):
    topwords = {}
    for t in df['topic'].unique():
        print(t)
        for p in df['persona'].unique():
            tempdf1 = df.loc[(df['persona']==default_persona_term)&(df['topic']==t)]
            tempdf2 = df.loc[(df['persona']==p)&(df['topic']==default_topic_term)]
            subdf = df.loc[(df['persona'].isin([default_persona_term,p]))&(df['topic'].isin([default_topic_term,t]))]
            ptopwords = pprint(p+' vs. ' +t,mw_custom(tempdf2,tempdf1,subdf))
            ttopwords = pprint(t+' vs. ' +p,mw_custom(tempdf1,tempdf2,subdf))
            if len(ptopwords)>0 and len(ttopwords)>0:
                topwords['%s-VS-%s'%(p,t)] = {p:ptopwords,
                                          t:ttopwords
                }
    sentence_lists = {}
    for personatopic, dic in topwords.items():

        p = personatopic.split('-VS-')[0]
        t = personatopic.split('-VS-')[1]

        subdf_unmarked_topic = df.loc[(df.topic==t) & (df.persona==default_persona_term)]
        subdf_persona_only = df.loc[(df.topic==default_topic_term) & (df.persona==p)]

        topic_sentence_list = []
        persona_sentence_list = []

        for word in dic[t]:
            all_sentences = get_list_of_sentences(subdf_unmarked_topic)
            temp_good = [se for se in all_sentences if word in se]
            topic_sentence_list.extend(temp_good)

        for word in dic[p]:
            all_sentences = get_list_of_sentences(subdf_persona_only)
            temp_good = [se for se in all_sentences if word in se]
            persona_sentence_list.extend(temp_good)


        if len(topic_sentence_list) > 0 and len(persona_sentence_list) > 0:

            sentence_lists[personatopic] = {}
            sentence_lists[personatopic][p] = persona_sentence_list
            sentence_lists[personatopic][t] = topic_sentence_list
    pole_embs = {}
    for personatopic in sentence_lists:
        pole_embs[personatopic] = {}
        p = personatopic.split('-VS-')[0]
        t = personatopic.split('-VS-')[1]

        plist = sentence_lists[personatopic][p]
        tlist = sentence_lists[personatopic][t]


        pole_embs[personatopic][p] = np.mean([emb_dict[s] for s in plist],axis=0)
        pole_embs[personatopic][t] = np.mean([emb_dict[s] for s in tlist],axis=0)
    axes = {}
    for personatopic in pole_embs:
        p = personatopic.split('-VS-')[0]
        t = personatopic.split('-VS-')[1]
        p_pole = pole_embs[personatopic][p]
        t_pole = pole_embs[personatopic][t]
        axes[personatopic] = [p_pole - t_pole, p_pole,t_pole]
    # Compute Normalized Similarities to Axes
    topic_dict_res = {}
    persona_dict_res = {}
    for personatopic in sentence_lists.keys():
                pt = personatopic.split('-VS-')
                p=pt[0]
                t= pt[1]
                if p!= 'person':
                    df_imp = df.loc[(df.topic==t)&(df.persona==p)]
            #         print(df_imp)
                    df_control_t = df.loc[(df.topic==t)&(df.persona==default_persona_term)]
                    df_control_p = df.loc[(df.topic=='comment')&(df.persona==p)]
                    sims = [cos_sim(x,axes[personatopic][0]) for x in df_imp['embeddings']]
                    sims_t = [cos_sim(x,axes[personatopic][0]) for x in df_control_t['embeddings']]
                    sims_p = [cos_sim(x,axes[personatopic][0]) for x in df_control_p['embeddings']]
                    avg_sim=np.mean(sims)
                    if t in topic_dict_res:
                        temp = topic_dict_res[t]
                        temp.append((p,rescale(avg_sim,np.mean(sims_t),np.mean(sims_p))))
                    else:
                        topic_dict_res[t] = [(p,rescale(avg_sim,np.mean(sims_t),np.mean(sims_p)))]

                    if p in persona_dict_res:
                        temp = persona_dict_res[p]
                        temp.append((t,rescale(avg_sim,np.mean(sims_t),np.mean(sims_p))))
                    else:
                        persona_dict_res[p] = [(t,rescale(avg_sim,np.mean(sims_t),np.mean(sims_p)))]

    return persona_dict_res, topic_dict_res
