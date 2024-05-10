import json

def get_data(data_path):
    speakers=[]
    dialogue_acts=[]
    with open(data_path,mode='r',encoding='utf-8-sig') as f:
        datas=eval(f.read())
        for values in datas.values():
            for data in values:
                speaker=data['speaker']
                speakers.append(speaker)
                dialogue_act=data['dialogue_act']
                dialogue_acts.append(dialogue_act)
    speakers=list(set(speakers))
    dialogue_acts=list(set(dialogue_acts))
    with open('speakers.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(speakers))
    with open('dialogue_acts.txt','w',encoding='utf-8') as f:
        f.write('\n'.join(dialogue_acts))



if __name__=='__main__':
    get_data('./IMCS-DAC_train.json')
