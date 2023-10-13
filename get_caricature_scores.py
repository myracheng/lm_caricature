'''
Usage:
    python get_caricature_scores.py filename default_persona default_topic
where 
- filename is the filename of the CSV (pandas dataframe) with columns named persona, topic, and response (generated outputs)
- default_persona is the persona used to denote the default-persona (e.g. 'user' or 'person')
- default_topic is the persona used to denote the default-persona (e.g. 'comment')

Example usage:
    python get_caricature_scores.py example/twitter_mini user comment

The topics and personas will be printed from biggest to smallest exaggeration score, and plots of individuation and exaggeration scores will be saved as PDFs. 
'''
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sys
from sklearn.metrics import f1_score, accuracy_score
import scipy
import os
from helper_functions import compute_exaggeration_scores


def main():
    args = sys.argv[1:]
    filename = args[0] 
    default_persona_term = args[1]
    default_topic_term = args[2]

    df = pd.read_csv('%s.csv'%filename) # simulations with column headers: persona, topic, response
    directory = os.path.dirname(filename)
    if not os.path.exists('data/%s'%directory):
        os.makedirs('data/%s'%directory)

    # encode sentences
    print("Encoding Sentences")
    emb_dict = encode_sentences(df)
    with open('data/%s_sentence_embeddings.pickle'%filename, 'wb') as handle:
        pickle.dump(emb_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # encode responses
    print("Encoding All Responses")
    df_with_embeddings = encode_df(df)
    df_with_embeddings.to_pickle("data/%s.pickle"%filename)
    
    # get individuation scores
    print("Computing Individuation Scores")
    indiv_scores = individuation_scores(df_with_embeddings, default_persona_term)
    
    # plot individuation scores
    plot_indiv_scores(indiv_scores, filename, df, default_persona_term)
    
    # get exaggeration scores
    print("Computing Exaggeration Scores")
    exag_scores_p, exag_scores_t = compute_exaggeration_scores(df_with_embeddings, emb_dict, default_persona_term, default_topic_term)

    # plot exaggeration socres
    plot_exag_scores(exag_scores_t, df, filename,default_persona_term, default_topic_term)

    # list exaggeration score by topic and persona
    temps = []
    for p in df.persona.unique():
        if p in exag_scores_p:
            temps.append([p,np.mean([y for x,y in exag_scores_p[p]])])
    print("MOST TO LEAST CARICATURED PERSONAS")
    for x,y in sorted(temps,key=lambda a:a[1],reverse=True):
        print('%s (%.2f),'%(x,y))
        
    temps = []
    for t in df.topic.unique():
        temps.append([t,np.nanmean([y for x,y in exag_scores_t[t]])])
    print("MOST TO LEAST CARICATURED TOPICS")
    for x,y in sorted(temps,key=lambda a:a[1],reverse=True):
        print('%s (%.2f),'%(x,y))
        
def encode_sentences(target_df):
    model = SentenceTransformer('all-mpnet-base-v2')
    full_list = []
    for c in target_df['response']:
        try:
            full_list.extend(sent_tokenize(c))
        except TypeError:
            print(c)
    sentence_set = list(set(full_list))
    embeddings = model.encode(sentence_set)
    emb_dict = {}
    for i in range(len(sentence_set)):
        emb_dict[sentence_set[i]] = embeddings[i]
    return emb_dict

def encode_df(target_df):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(target_df.response)
    target_df['embeddings'] = list(embeddings)
    return target_df
    
def individuation_scores(responses,default_persona_term):
    
    alls = {}
    for p in responses.persona.unique():
        if p!= default_persona_term:
            f1s=[]
            accs=[]
            for t in responses.topic.unique():
                        inds = responses.loc[(responses['topic']==t)&(responses['persona'].isin([p,default_persona_term]))].index
                        X_train, X_test, y_train, y_test = train_test_split(
                            list(responses['embeddings'][inds]), responses.persona.astype("category").cat.codes[inds], 
                            test_size=0.2, random_state=42)
                        clf = RandomForestClassifier(random_state=0)
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        f1 = f1_score(y_test, y_pred, average='macro')
                        f1s.append(f1_score(y_test, y_pred, average='macro'))
                        accs.append(accuracy_score(y_test, y_pred))
                        if accuracy_score(y_test, y_pred) < 0.5:
                            print("Less than 50%")
                            print(t)
                            print(p)
            alls[p]=(f1s,accs)
    return alls

def plot_indiv_scores(alls,filename,df,default_persona_term):
    CB = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']
    plt.rcParams["figure.figsize"] = (20,5)
    plt.rcParams.update({'font.size': 22})

    meanlist=[]
    pers = list(df.persona.unique())
    pers.remove(default_persona_term)
    for p in pers:
        if p != default_persona_term:
            if len(alls[p][1])>0:
                meanlist.append(alls[p][1])

    plt.errorbar(range(len(meanlist)), [np.mean(x) for x in meanlist],[scipy.stats.sem(x) for x in meanlist],elinewidth=5,color=CB[0],marker='o',
              ms=20, mew=5,label='%s'%filename,linewidth=0,alpha=0.7)
    print([scipy.stats.sem(x) for x in meanlist])
    plt.title('Individuation (Differentiability from Default-Persona)')
    plt.ylabel('Accuracy')
    plt.axhline(y=0.5, linestyle='--', color='black', linewidth=1, alpha=0.3)
    plt.axhline(y=1, linestyle='--', color='black', linewidth=1, alpha=0.3)
    plt.legend(loc='lower center')
    labs=[x.replace('person','') for x in pers]
    plt.xticks(range(len(meanlist)),labs,rotation=0,fontsize=22)
    plt.savefig('%s_individuation_scores.pdf'%filename)

def plot_exag_scores(topic_dict_res, df, filename,default_persona_term,default_topic_term):
    matplotlib.rcParams.update({'font.size': 30})
    plt.figure(figsize=(20,14))
    plt.title('Similarity to Persona-Topic Axes')
    topics = list(df.topic.unique())
    topics.remove(default_topic_term)
    p_dict = {}
    for t in topics:
        for x,y in topic_dict_res[t]:
            if x in p_dict:
                p_dict[x].append(y)
            else:
                p_dict[x] = [y]
    plt.errorbar(range(len(p_dict.keys())), [np.mean(v) for p, v in p_dict.items()], [np.std(v)for p, v in p_dict.items()],elinewidth=5,marker='o',label=filename,ms=20, mew=5,linewidth=0,alpha=0.5)
    parse_labels = []
    for x,y in topic_dict_res[t]:
        if x ==default_persona_term:
            parse_labels.append('%s\n(default-persona)'%default_persona_term)
        else:
            parse_labels.append(x)
    plt.xticks(ticks=range(len(parse_labels)),labels=parse_labels,rotation=60)
    plt.tight_layout()
    plt.savefig('%s_exaggeration_scores.pdf'%filename)

    
if __name__ == '__main__':
    main()
    
    
    