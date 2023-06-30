import json
import pandas as pd

# path = 'Data_Models/20230525_AllTransportPagingAlerts_RAW.json'
# with open(path,'r',encoding='UTF-16') as f:
#     data = json.loads(str(f.read()).strip("'<>() "))  # .replace('\'', '\"')  # .replace('\'', '\"')
# df = pd.DataFrame(data)
# df = df.loc[:,['AlertId','CreatedTime']]
# train_data = pd.read_json('Data_Models/all_info.jsonl', lines=True)
# train_data['AlertId']=train_data['AlertId'].apply(str)
# df_merged = train_data.merge(df, on='AlertId', how='left', suffixes=('_a', '_b'))
# df_merged = df_merged.loc[:, ['AlertId', 'CreatedTime_b','AlertType','ScopeType','Keyword','AdviceDetail','KeyValuePairs','Advice_Detail_Response']]
# df_merged=df_merged.rename(columns={'CreatedTime_b':'CreatedTime'})
# df_merged=df_merged.dropna(subset=['CreatedTime'])
# df_merged.to_json('Data_Models/all_info_adjust.jsonl',orient='records', lines=True)

train_data = pd.read_json('Data_Models/all_info_adjust.jsonl', lines=True)
train_data.to_json('Data_Models/all_info_adjust.jsonl',orient='records', lines=True)
train_data.to_json('Data_Models/all_info.jsonl',orient='records', lines=True)
print('done')