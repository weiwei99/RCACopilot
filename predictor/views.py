import nltk
nltk.download('stopwords')
nltk.download('punkt')
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import HttpResponse
import fasttext
import pandas as pd
import os
import faiss
import re
import numpy as np
import openai
import time
from utils import text_clean,truncate_text,count_word_num,clean,process


'''
This file primarily processes web application requests and manages the flow of data processing.

The <Predict> method handles client requests.
The <GPT_predict> method generates output results using the GPT model.
The <fasttext_predict> method generates output results using the FastText model.
'''

os.environ['CHATGPT_API_KEY'] = ''
oai_key = os.environ.get("CHATGPT_API_KEY")
openai.api_type = "azure"
openai.api_base = "https://cloudgpt.openai.azure.com/"
openai.api_key = oai_key
openai.api_version = "2023-03-15-preview"
chatgpt_model_name= "gpt-4-20230321"


def fasttext_predict(input):
    model = fasttext.load_model('Data_Models/models')
    ft_label = model.predict(input)[0][0]
    ft_label = ft_label[9:]
    return ft_label



def GPT_predict(data,alpha=0.3,num=5):
    train_data = pd.read_json('Data_Models/all_info.jsonl', lines=True)
    model = fasttext.load_model('Data_Models/models')
    data = pd.DataFrame.from_dict(data, orient='index').T
    data['AdviceDetail'][0] = clean(data['AdviceDetail'][0])
    prompt = "Input:\n" + data['AdviceDetail'][
        0] + """\nContext: Please summarize the above input. Please note that the above input is a log information. The summary results should be about 200 words no more than 250 words and should cover important information of the log as much as possible, just return the summary without any additional output"""
    retry_count = 0
    max_retries = 6
    flag = False
    added=False
    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=chatgpt_model_name,
                messages=[
                    {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                    {"role": "user", "content": prompt}
                ]
            )
            break
        except Exception as e:
            print(e)
            retry_count += 1
            print("{}th retry".format(retry_count))
            time.sleep(1)
            if retry_count > max_retries:
                flag = True
                break
    if flag:
        AdviceDetailSummary=data['AdviceDetail'][0]
    else:
        AdviceDetailSummary = response['choices'][0]['message']['content']
    AdviceDetailSummary = process(AdviceDetailSummary)
    new_prompt = """Context:The following description shows the error log information of an incident. Please help me select the incident information that is most likely to have the same root cause and give your explanation(just give one answer).If not, please select the first item 'None' .If not, please select the first item 'None'.The output format should be JSON format,for example:{"Option": "B","Explanation":"..."}, Please use transfer characters'\\"'in explanation To prevent JSON from mistaking it for JSON syntax. \n"""
    new_prompt += "Input:\n" + AdviceDetailSummary + "\n"
    data['CreatedTime'] = pd.to_datetime(data['CreatedTime'])

    train_data['CreatedTime'] = pd.to_datetime(train_data['CreatedTime'])
    created_time = data['CreatedTime'][0]
    IncidentId=data['AlertId'][0]
    filtered_data = train_data
    # get indices
    # if not os.path.exists('Data_Models/index_file.index'):
    filtered_data["metadata"] = filtered_data.to_dict(orient="records")
    filtered_data_text = filtered_data['metadata'].apply(lambda x: text_clean(
        str(x['KeyValuePairs']) + ' ' + x['AlertType'] + ' ' + str(x['ScopeType']) + ' ' + x['AdviceDetail']))
    vectors = np.array([model.get_sentence_vector(text) for text in filtered_data_text])
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    # faiss.write_index(index,'Data_Models/index_file.index' )
    # else:
    #     index = faiss.read_index('Data_Models/index_file.index')

    def decay_factor(alpha, time_diff):
        return np.exp(-alpha * time_diff)

    def similarity_with_time_decay(x, length, created_time, alpha, num):
        # distance
        distances, indices = index.search(x, k=length)
        distances = distances.flatten()
        sorted_indices = np.argsort(indices)
        sorted_distances = distances[sorted_indices]
        # time_diff
        time_diffs = (created_time - filtered_data['CreatedTime']).values / np.timedelta64(1, 'D')
        # decay
        decay_factors = decay_factor(alpha, time_diffs)
        # similarity
        similarities = 1 / (1 + sorted_distances) * decay_factors
        # find k similar
        sorted_indices = np.argsort(similarities[0])[::-1]
        return filtered_data.iloc[sorted_indices[:num]]

    # mask = (train_data['AlertType'] == alert_type) & (train_data['CreatedTime'] > created_time - pd.Timedelta(days=60))
    query = text_clean(
        str(data['KeyValuePairs'][0]) + ' ' + data['AlertType'][0] + ' ' + str(data['ScopeType'][0]) + ' ' +
        data['AdviceDetail'][0])
    query_vector = np.array([model.get_sentence_vector(query)])
    # find num nearest
    train_samples = similarity_with_time_decay(query_vector, len(filtered_data), created_time, alpha, num)
    order = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M','N','O']
    map_dict = {}
    new_prompt += 'Option:\n' + 'A:None' + '\n'
    i = 0
    log=[]
    for _, train_sample in train_samples.iterrows():
        if str(train_sample['AlertId'])==str(IncidentId):
            added=True
            continue
        new_prompt += order[i]+':'+train_sample['Advice_Detail_Response']+'\n'
        map_dict[order[i]]={'AlertId':train_sample['AlertId'],'Keyword':train_sample['Keyword']}
        log.append(str(train_sample['AlertId']))
        i+=1
    new_prompt+='Answer:'

    retry_count = 0

    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=chatgpt_model_name,
                messages=[
                    {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                    {"role": "user", "content": new_prompt}
                ]
            )
            break
        except Exception as e:
            print(e)
            retry_count += 1
            print("{}th retry".format(retry_count))
            time.sleep(1)
            if retry_count > max_retries:
                break
    res = response['choices'][0]['message']['content']
    match = re.search(r'\{.*?\}', res,flags=re.DOTALL)
    if match:
        try:
            json_dict = json.loads(match.group())
            option = json_dict['Option']
            option = option[0]  # to avoid muiltple answers
        except:
            match = re.search(r'"Option":\s*"(\w)\b', res)
            if match:
                option = match.group(1)
    else:
        words = res.split()
        option = words[0][0]
    if option in map_dict.keys():
        AlertId=map_dict[option]['AlertId']
        Keyword=map_dict[option]['Keyword']
    else:
        AlertId='-1'
        Keyword='None'
    res_dict={'prompt':new_prompt,'SimilarId':log,'AlertId':AlertId,'Keyword':Keyword,'Explanation':res,'IsExist':added}
    data['Advice_Detail_Response']=AdviceDetailSummary
    data['Keyword']='None'
    data=data[['AlertId', 'CreatedTime','AlertType','ScopeType','Keyword','AdviceDetail','KeyValuePairs','Advice_Detail_Response']]#for alignment
    row_to_add = data.iloc[0]
    train_data=train_data.drop(columns=['metadata'])
    train_data = pd.concat([train_data, row_to_add.to_frame().T], axis=0, ignore_index=True)
    train_data['AlertId']=train_data['AlertId'].apply(str)
    train_data=train_data.drop_duplicates(subset='AlertId')
    train_data['CreatedTime'] = train_data['CreatedTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    train_data.to_json('Data_Models/all_info.jsonl', orient='records', lines=True)
    return res_dict


@csrf_exempt
def predict(request):
    if request.method == 'POST' :
        data = json.loads(request.body)
        data['AlertId']=data.pop('IncidentId')
        data['CreatedTime'] = data.pop('IncidentTime')
        data['AlertType'] = data.pop('IncidentType')
        data['ScopeType'] =data.pop('IncidentScopeType')
        data['KeyValuePairs'] =data.pop('Features')
        input=text_clean(str(data['KeyValuePairs']) + ' ' + data['AlertType'] + ' ' + str(data['ScopeType']) + ' ' + data['AdviceSummary'] + ' ' + data['AdviceDetail'])
        ft_label=fasttext_predict(input)
        # GPT_predict_label=GPT_predict_old(data,input)
        GPT_predict_label = GPT_predict(data)
        response_data = {'fasttext_predict': ft_label, 'GPT_predict_label':GPT_predict_label}
        return HttpResponse(json.dumps(response_data), content_type="application/json")
    else:
        return HttpResponse(status=404)


# For Test
# data={
# 	"IncidentId": "384590937",
# 	"IncidentTime": "2023-04-25T09:03:28",
# 	"IncidentType": "Crash/microsoft.exchange.store.worker/8290-dumptidset",
# 	"IncidentSeverity": 3,
#     "IncidentScopeType": "None",
# 	"IncidentSubject": "[PROD] Sev 3: ID 384590937: Forest-wide process crash over threshold for: microsoft.exchange.store.worker in eurp250 for BE",
# 	"IncidentDescription": "IcM Auto-Correlation has linked incident(s) <a href=\"https://portal.microsofticm.com/imp/v3/incidents/details/384584377/home\">384584377</a> as RELATED. Learn more <a href=\"https://aka.ms/icmguide/aces\">here</a>",
# 	"AdviceSummary": "",
# 	"AdviceDetail": "<b>Exception details in last 12 hours in the forest</b> (Execute in [Web(<a href=\"https://dataexplorer.azure.com/clusters/o365monwus.westus/databases/o365monitoring?query=H4sIAAAAAAAEAI1UW0/bMBR+51dYe0k6BdoG2ICtk0opUGlARWE8TFPlJi41dezMl5ZOaL99J3Gaxr1M+CXxOd+5f8eMaER4/EATglooxppo+PW9o3p4XA8b4SE6PWscnoUnqH3j1b7sMcArjaVesxjiOPa9a2GkF6D9ZhgsvRY2kcRq0uNjATZeJ7vUExpJocRYH5DXaIL5MzlQWkhyMBdySmT9JDxt7McmSTWNFdGedUQ5hOcRucV5fJUyqv3Se4A+1D8EqFn72fhVBMaMgUU0vQbIToPQGnRfn7BWgn8Xz90Z4Xp4xcQIs703lErxQiK9h+AUpRM+G2alB7nwnvLnQhgxYeIh/FEpeAJuLOISalPawcQkZWKRQR451QHKce0ZpgyPKOS5uJLCpI4Jx2VEwYijkiCwqhscTSjf1PaK5lnUDyIVFRxQI0NZXFytrue2+dIw1k6hdRHWgLktk7gRscnT6B8VVRoeaeu1f2xF3deIpEvZJyvrrI2l/9nKB1PjlNCXYkaztKC9Aw1kc7RtCDWDNlU0PZWzy17AOiJK9WJ7vVOrku8GTr1dKYW8J6mQGiJ1OR4xEgOMjsc+VQRIuPC3YmoB8h75lIs5B+Zn8K0wQD1IQwDiXWKmiFerAavmEyKJ5dO31mqv8oyyg3lstV9by31ydEtKwU4RI9PwuOE5emeKqPXX2R4H6c4jgzqLs9oANKLczxIJ0DnlF1jj80V2faI8FnO/rGH1AECHqnkEKz4ElQG5GQQlj4KCYpBBTBV0FFL4CBdlkgRL+oegjjA8W6so+/o1NFqs+9oZfksQBWPLXeRe35DGU4KajQaoXgTlaAp1tijnRBpOf5usi/6uZ8O2dznm5Xuxe9QlYse4C25n826Wzpdz2bVR9rx3n+3ZtrFrmgui4ZkqN4QLbZekI5IU3l1SAmH8G7LKEAZZyrVKhOpoN6NhDnvoSrdNvBKpWrmNA2W9D15hvUuhnBwblP0PsYL1Ut4YTagGYv0DRJaxVoEHAAA=\">Link</a>)])\r\n<html><body><table border=\"1\" cellspacing=\"0\"><tr><th>InstanceName</th><th>Count</th><th>CallstackHash</th><th>Exception</th><th>Function</th><th>Module</th><th>ExceptionDetail</th></tr><tr><td>Microsoft.Exchange.Store.Worker</td><td>1897</td><td>8290-dumptidset</td><td>System.InvalidOperationException</td><td>M.E.S.S.L.BigFunnelFilter.GetMatchingDocumentIdsInternalNoRanking</td><td>M.E.S.Storage.LogicalDataModel</td><td>System.InvalidOperationException: <message redacted>\r\n   at BigFunnel.IndexStream.QueryCompiler.ReportRankInputForDocumentBottomUpInplace(XRankRecallForwardIterator xrankForwardIterator, Int32 documentId)\r\n   at BigFunnel.IndexStream.QueryHandler.Query(IIndexReader indexReader, StoreIndexProvider storeIndexProvider, DocumentFrequencyEstimator dfEstimator, Action`1 reportDocumentFrequencies, ReportDocumentRankInputDelegate reportDocumentRankInput, IRankInputCollector rankInputCollector, Int32 chunkSize)+MoveNext()\r\n   at BigFunnel.PostingList.QueryHandler.Query(IItemIdAccessor itemIdAccessor, IRankInputCollector rankInputCollector, DocumentFrequencyEstimator dfEstimator)+MoveNext()\r\n   at Microsoft.Exchange.Server.Storage.LogicalDataModel.BigFunnelFilter.GetMatchingDocumentIdsInternalNoRanking(Context context, IMailboxAccessor mailboxAccessor, BigFunnelDiagnostics bigFunnelDiagnostics, SearchCriteria folderIdCriteria, IBigFunnelQuery bigFunnelQuery, IBigFunnelPostingListQuery externalPostingListQuery, Int32 maximumItems, Int32 pagedMinimumResultSet, Boolean mustProcessAttributeVector, Boolean mustLoadAllItemIds, QueryParameters& queryParameters, IsFolderVisibleDelegate isFolderVisibleDelegate, BigFunnelMailboxWideIndexSortProvider mailboxWideIndexSortProvider, Nullable`1& attributeBlobVersion)\r\n   at Microsoft.Exchange.Server.Storage.LogicalDataModel.BigFunnelFilter.GetMatchingDocumentIdsInternal(Context context, IMailboxAccessor mailboxAccessor, BigFunnelDiagnostics bigFunnelDiagnostics, SearchCriteria folderIdCriteria, IBigFunnelQuery bigFunnelQuery, IBigFunnelPostingListQuery externalPostingListQuery, Int32 maximumItems, Int32 pagedMinimumResultSet, Boolean mustProcessAttributeVector, Boolean mustLoadAllItemIds, QueryParameters queryParameters, IsFolderVisibleDelegate isFolderVisibleDelegate, BigFunnelMailboxWideIndexSortProvider mailboxWideIndexSortProvider, SegmentedList`1& resultDocIds, Int32& postCollapseCount)\r\n   at Microsoft.Exchange.Server.Storage.LogicalDataModel.BigFunnelFilter.GetAttributeSortedIndexInternal(Context context, IMailboxAccessor mailboxAccessor, BigFunnelDiagnostics bigFunnelDiagnostics, SearchCriteria folderIdCriteria, IBigFunnelQuery bigFunnelQuery, IBigFunnelPostingListQuery externalPostingListQuery, QueryParameters queryParameters, IList`1 additionalRenames, Boolean mustLoadAllItemIds, IsFolderVisibleDelegate isFolderVisibleDelegate, BigFunnelMailboxWideIndexSortProvider mailboxWideIndexSortProvider, Boolean& indexMayContainFalsePositive, SegmentedList`1& resultDocIds, Int32& postCollapseCount)\r\n   at Microsoft.Exchange.Server.Storage.LogicalDataModel.BigFunnelFilter.GetAttributeSortedIndex(Context context, IMailboxAccessor mailboxAccessor, BigFunnelDiagnostics bigFunnelDiagnostics, SearchCriteria folderIdCriteria, IBigFunnelQuery bigFunnelQuery, IBigFunnelPostingListQuery externalPostingListQuery, QueryParameters queryParameters, IList`1 additionalRenames, IsFolderVisibleDelegate isFolderVisibleDelegate, Boolean mustLoadAllItemIds, BigFunnelMailboxWideIndexSortProvider mailboxWideIndexSortProvider, Boolean& indexMayContainFalsePositive, SegmentedList`1& resultDocIds, Int32& postCollapseCount)\r\n   at Microsoft.Exchange.Server.Storage.LogicalDataModel.InstantSearchViewTable.PrepareResultsIndexChunked(Context context)+MoveNext()\r\n   at Microsoft.Exchange.Server.Storage.StoreCommonServices.ChunkedEnumerable.DoChunk(Context context)\r\n   at Microsoft.Exchange.Server.Storage.MapiDisp.RopHandler.ChunkedPrepare(MapiContext context, IChunked prepare)+MoveNext()\r\n   at Microsoft.Exchange.Server.Storage.MapiDisp.RopHandler.QueryRows(MapiContext context, MapiViewTableBase view, QueryRowsFlags flags, Boolean useForwardDirection, UInt16 rowCount, QueryRowsResultFactory resultFactory)+MoveNext()\r\n   at Microsoft.Exchange.Server.Storage.MapiDisp.RopHandlerBase.QueryRows(IServerObject serverObject, QueryRowsFlags flags, Boolean useForwardDirection, UInt16 rowCount, QueryRowsResultFactory resultFactory)\r\n   at Microsoft.Exchange.RpcClientAccess.Parser.RopQueryRows.InternalExecute(IServerObject serverObject, IRopHandler ropHandler, IOutputBuffers outputBuffers)\r\n   at Microsoft.Exchange.RpcClientAccess.Parser.InputRop.Execute(IConnectionInformation connection, IRopDriver ropDriver, ServerObjectHandleTable handleTable, IOutputBuffers outputBuffers)\r\n   at Microsoft.Exchange.RpcClientAccess.Parser.RopDriver.ExecuteRops(List`1 inputArraySegmentList, AuxiliaryData auxiliaryData, Byte[]& fakeOut)\r\n   at Microsoft.Exchange.RpcClientAccess.Parser.RopDriver.ExecuteOrBackoff(IList`1 inputBufferArray, AuxiliaryData auxiliaryData, IRpcOutputBuffers rpcOutputBuffers, Byte[]& fakeOut)\r\n   at Microsoft.Exchange.Server.Storage.MapiDisp.MapiRpc.<>c__DisplayClass40_0.<DoRpcInternal>b__0(MapiContext operationContext, MapiSession& session, Boolean& deregisterSession, AuxiliaryData auxiliaryData)\r\n   at Microsoft.Exchange.Server.Storage.MapiDisp.MapiRpc.Execute(IExecutionDiagnostics executionDiagnostics, MapiContext outerContext, String functionName, Boolean isRpc, IntPtr& contextHandle, Boolean tryLockSession, String userDn, IList`1 dataIn, Int32 sizeInMegabytes, ArraySegment`1 auxIn, ArraySegment`1 auxOut, Int32& sizeAuxOut, ExecuteDelegate executeDelegate)\r\n   at Microsoft.Exchange.Server.Storage.MapiDisp.MapiRpc.DoRpcInternal(IExecutionDiagnostics executionDiagnostics, IntPtr& contextHandle, IList`1 ropInArraySegments, Boolean internalAccessPrivileges, Boolean nonLocalReplicaAccessPrivileges, Boolean fullMailboxRightsPrivileges, ArraySegment`1 sessionDataIn, ArraySegment`1 auxIn, IDoRpcOutputBuffers outputBuffers)\r\n   at Microsoft.Exchange.Server.Storage.MapiDisp.PoolRpcServer.EcDoRpc(MapiExecutionDiagnostics executionDiagnostics, IntPtr& sessionHandle, UInt32 flags, UInt32 maximumResponseSize, ArraySegment`1 request, ArraySegment`1 sessionDataIn, ArraySegment`1 auxiliaryIn, IPoolSessionDoRpcCompletion completion)\r\n   at Microsoft.Exchange.Server.Storage.MapiDisp.PoolRpcServer.EcPoolSessionDoRpc_Unwrapped(MapiExecutionDiagnostics executionDiagnostics, IntPtr contextHandle, UInt32 sessionHandle, UInt32 flags, UInt32 maximumResponseSize, ArraySegment`1 request, ArraySegment`1 sessionDataIn, ArraySegment`1 auxiliaryIn, IPoolSessionDoRpcCompletion completion)\r\n   at Microsoft.Exchange.Server.Storage.MapiDisp.PoolRpcServer.<>c__DisplayClass55_0.<EcPoolSessionDoRpc2>b__0()\r\n   at Microsoft.Exchange.Server.Storage.Common.WatsonOnUnhandledException.Guard(IExecutionDiagnostics executionDiagnostics, Action body)</td></tr></table></body></html>\r\n<b>Top 5 crash machines</b> (Execute in [Web(<a href=\"https://dataexplorer.azure.com/clusters/o365monwus.westus/databases/o365monitoring?query=H4sIAAAAAAAEAH1UbU/bMBD+zq+w+JJ0MrQEugFbJxVWoNKAipfxYZoqN75Sr44d2U5LJ7Tfvkucvrgby5fk7p67e+4tEhwBxR9EBqRDOHPg8DOOjppJu5m0kkNycto6PE2OSfc6anzckYi3jhm35TFknMfRlS5MRMneQUKXUWuf1DA76auxRp/ovBSamUiNtnrs9uElnTD1DPvWaQP7c22mYJrHyUlrjxdZ7gS34CIfSChMr1K4YVV+m0vh4lV0Snabu5QcNL63ftSJmZTokU6vEPKmQ+Idei9PzFmtvurn3gyUG15KPWJy55XkRv+E1O0QfOrSQc2GZem0Ut4J9VwrU6kLPsQvYbTKMIxHXGBt1gUYDrnUixLyqISjpMJ1Z0xINhLIc3FpdJEHLoqtMmoJgcmgwpuuWToR6m9rv26eR30DY4VWiBoVQvJa9LZ+2OaLQspujq1LmUPMzYrEteZFRWNwVFdZqNT5qIO2V/VeUsiXuvded741lsEHr7+fFkEJA6NnoqSF7b13uGyBtYupZtimDUvfVtvlBfROwdo+9+KtXZd8ex/U2zNGmzvItXGYqafYSAJHmBiPY2EBl3AR/xPToCR6VFOl5wo3v4T/E4aoB1MAQqILJi1EjQZu1XwCBvw+fe6s76piVD5McW/91FneU2BbrhTeFBQmT9qtKLAHUySd38H1BMhwHiU0PBwSxvVNLhMfrI+DjISKS46UnAn1hTl2tijFJ6G4nser8tb/BmzeJkW6XhW6MbuQHF2tGK23jy7XAalwYbHryOUdCrbIMmbELyAV3eG5LlR5gGn5jhtktNgO/Sab/+W0OOkq1jrJRlNqHN0yS5EJR9qv9fsP9yTEJYgFAAA=\">Link</a>)])\r\n<html><body><table border=\"1\" cellspacing=\"0\"><tr><th>Machine</th><th>Crash_Count</th></tr><tr><td>GV2P250MB0756</td><td>20</td></tr><tr><td>DU2P250MB0046</td><td>19</td></tr><tr><td>AM8P250MB0043</td><td>19</td></tr><tr><td>DU2P250MB0272</td><td>19</td></tr><tr><td>AS8P250MB0219</td><td>17</td></tr></table></body></html>\r\n",
# 	"Features": [
# 		{
# 			"ScopeRelativeRole": None,
# 			"Name": "031d441f-4ac1-4f58-9151-08d8126f7a1e(Path:Properties.AlertType|Pattern:*25e4*)",
# 			"Value": "False",
# 			"Statement": "The value of Is the work item match the pattern with parameters (Path:Properties.AlertType|Pattern:*25e4*) is False",
# 			"OcaFeatureId": "Feature.IsWorkItemMatchPattern"
# 		},
# 		{
# 			"ScopeRelativeRole": None,
# 			"Name": "031d441f-4ac1-4f58-9151-08d8126f7a1e(Path:Properties.AlertType|Pattern:*4c6a*)",
# 			"Value": "False",
# 			"Statement": "The value of Is the work item match the pattern with parameters (Path:Properties.AlertType|Pattern:*4c6a*) is False",
# 			"OcaFeatureId": "Feature.IsWorkItemMatchPattern"
# 		}
# 	],
# 	"ExternalActions": []
# }
# data['AlertId']=data.pop('IncidentId')
# data['CreatedTime'] = data.pop('IncidentTime')
# data['AlertType'] = data.pop('IncidentType')
# data['ScopeType'] =data.pop('IncidentScopeType')
# data['KeyValuePairs'] =data.pop('Features')
# data['AlertSeverity'] =data.pop('IncidentSeverity')
# res=GPT_predict(data)
# print(res)


