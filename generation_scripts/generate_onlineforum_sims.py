# from bert_score import score
import pandas as pd
import numpy as np
import openai
import itertools
import backoff
def main():

    openai.api_type = "azure"
    # The API key for your Azure OpenAI resource.
    with open('keys.txt','r') as f:
        openai.api_key = [line.rstrip('\n') for line in f][0]
    # The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
    openai.api_base = TODO
    # Currently Chat Completion API have the following versions available: 2023-03-15-preview
    openai.api_version = '2023-03-15-preview'

    filename = 'onlineforumtopics'
    full_file = open('outputs_%s_100_gpt4.json'%filename, 'w')
    full_file.write("[")
    text_file = open('outputs_%s_100_gpt4.txt'%filename, 'w')
    with open("../topics/%s.txt"%filename, 'r') as f:
         topic_list = [line.rstrip('\n') for line in f]
    num_per_gen = 10
    gender_list = ['woman','man','person', 'nonbinary']
    age_list = ['20-year-old','80-year-old','40-year-old']
    ideo_list = ['conservative','liberal','moderate']
    race_list = ['white','Black','Asian','Hispanic','Middle-Eastern']
    chars = [age_list,ideo_list,race_list,gender_list]
    for model in ['gpt4']:
        for j in range(10):
            for topic in topic_list:
                for char_list in chars:
                    for char in char_list:
                        persona = char
                        if 'man' not in persona and 'person' not in persona:
                            persona+= ' person'
                        
                        outputs, prompt = get_persona_chat(persona,topic,model,num_completions=num_per_gen)
                        full_file.write("%s,\n" % outputs)
                        if model == 'gpt4' or model =='chat':
                            for i in range(num_per_gen):
                                try:
                                    
                                    text_file.write("%s\t%s\t%s\t%s\t%s\n" % (prompt,model,persona,topic, outputs.choices[i]['message']['content']))
                                except KeyError:
                                    text_file.write("%s\t%s\t%s\t%s\t%s\n" % (prompt,model,persona,topic, "NO OUTPUT"))
                        else:
                            for i in range(num_per_gen):
                                try:
                                    text_file.write("%s\t%s\t%s\t%s\t%s\n" % (prompt,model,persona,topic, outputs.choices[i]['text']))
                                except KeyError:
                                    text_file.write("%s\t%s\t%s\t%s\t%s\n" % (prompt,model,persona,topic, "NO OUTPUT"))
                

@backoff.on_exception(backoff.expo, openai.error.APIError)
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_persona_chat(persona,topic,model,num_completions=1):
    if topic=='comment':
        if persona[0] == 'A':
            prompt = 'An %s posted the following comment to an online forum:' %(persona)
        else:
            prompt = 'A %s posted the following comment to an online forum:' %(persona)
    else:
        if persona[0] == 'A':
            prompt = 'An %s posted the following comment on %s to an online forum:' %(persona,topic)
        else:
            prompt = 'A %s posted the following comment on %s to an online forum:' %(persona,topic)
    if model =='gpt4':

        response = openai.ChatCompletion.create(
                    engine=model,
                        n=num_completions,
                        max_tokens=256,
                    messages=[
                            {"role": "user", "content": prompt,
                            }
                        ]
                    )
    else: #for DV3 API
        response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                temperature=1, 
                max_tokens=256,
                n = num_completions,
                top_p=1,
                presence_penalty=0
                )
                

    return response, prompt
if __name__ == '__main__':
    main()



