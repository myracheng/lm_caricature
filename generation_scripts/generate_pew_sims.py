import pandas as pd
import numpy as np
import openai
import itertools
import sys
import backoff

def main():

    # This is set to `azure`
    openai.api_type = "azure"
    # The API key for your Azure OpenAI resource.
    with open('keys.txt','r') as f:
        openai.api_key = [line.rstrip('\n') for line in f][0]
    # The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
    openai.api_base = TODO
    # Currently Chat Completion API have the following versions available: 2023-03-15-preview
    openai.api_version = '2023-03-15-preview'
    args = sys.argv[1:]
    filename = args[0] 

    full_file = open('outputs_%s_gpt4_100_nojson.json'%filename, 'w')
    
    full_file.write("[")
    text_file = open('outputs_%s_gpt4_100_nojson.txt'%filename, 'w')
    with open("../topics/%s.txt"%filename, 'r') as f:
        topic_list = [line.rstrip('\n') for line in f]
    topic_list.append('comment')
    num_per_gen = 10
    persona_dict = {
    'age':['20-year-old','person','80-year-old','40-year-old'],
    'ideo':['conservative','moderate','liberal'],
    'race':['white','Black','Asian','Middle-Eastern','Hispanic'], 'gender':['woman','man','nonbinary']}
    for j in range(10):
        for topic in topic_list:
            for char_type, char_list in persona_dict.items():
                for char in char_list:
                    persona = char
                    if 'man' not in persona and 'person' not in persona:
                        persona+= ' person'
                    smalldict = {char_type:char}
                    outputs = get_op_json_question(topic,smalldict,num_completions=num_per_gen)
                    full_file.write("%s,\n" % outputs)
    
                    for i in range(num_per_gen):
                        try:
                            text_file.write("%s\t%s\t%s\n" % (persona,topic, outputs.choices[i]['message']['content']))
                        except (AttributeError,KeyError):
                            text_file.write("%s\t%s\t%s\n" % (persona,topic, "NO OUTPUT"))
    
 
@backoff.on_exception(backoff.expo, openai.error.APIError)
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_op_json_question(question,chardict,num_completions=1):
    prompt='Below you will be asked to provide a short description of your identity and then answer some questions.\nDescription: '
    
    if 'ideo' in chardict:
        prompt+='In politics today, I would describe my political views as %s. '%chardict['ideo']
    if 'race' in chardict:
        prompt+='I am %s. '%chardict['race']
    if 'age' in chardict:
        prompt+='I am %s. '%chardict['age']
    if 'gender' in chardict:
        prompt+= 'I identify as a %s.'%chardict['gender']
    prompt+='\nQuestion: %s'%question
    prompt+='\nAnswer:'
    response = openai.ChatCompletion.create(
                  engine='gpt4',
                    n=num_completions,
                  messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
    return response
if __name__ == '__main__':
    main()


