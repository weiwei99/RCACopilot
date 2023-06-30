# CloudGPT for Microsoft Services



## Directory Structure

- `rca/`

  - `Data_Models/`: Contains the raw data and preprocessed data used in the analysis.
    - `20230525_AllTransportPagingAlerts_RAW.json`: Raw data file.
    - `all_info.jsonl`:Preprocessed data(after summarization)
    - `file_exts.txt` to process string.
  - `data_base_build.py`: This script enhances the clarity of the original data and summarizes the AdviceDetails of all incidents using GPT4.The final generated JSON file is stored in the [Data_Models] directory.
  - `FastEmbeddingTrain.py`: To trian a fasttext model for retrieving similar incidents.
  - `manage.py`:To run a Django Server
  - `utils.py`: Commonly used methods stored here.
  - `requirements.txt` dependencies for packages
  - `predictor/`:
     * `views.py`:**Accept HTTP requests for GPT inference and return results(Important)**

   

## Preprocessing

### data_base_build.py

- This script enhances the clarity of the original data and summarizes the AdviceDetails of all incidents using GPT4
- Input: `20230525_AllTransportPagingAlerts_RAW.json`
- Output: `Data_Models/all_info.jsonl`

### FastEmbeddingTrain.py

- This script is used for training the FastText model.
- Input: `Data_Models/all_info.jsonl`
- Output: `Data_Models/models`

## Run a Django Server

`python manage.py runserver`

### views.py

- This script is used for inferring using the trained model and GPT prompt.
- Input: `Http requests` (specific input files)
- Output: `return a response`

## Details

### views.py

* `predict(request)`:To handle the incoming request ,parse the request body and return the output.
* `fasttext_predict(input)`: Deprecated.
* `GPT predict`(data,alpha,num): 
  * [load the summarized data](https://dev.azure.com/minghuama/GPT4IcM/_git/cloudgpt-oca?path=/predictor/views.py&version=GBmaster&line=44&lineEnd=44&lineStartColumn=5&lineEndColumn=72&lineStyle=plain&_a=contents)
  * [load the fasttext model](https://dev.azure.com/minghuama/GPT4IcM/_git/cloudgpt-oca?path=/predictor/views.py&version=GBmaster&line=45&lineEnd=45&lineStartColumn=5&lineEndColumn=54&lineStyle=plain&_a=contents)
  * [preprocess the incoming data and get Sum. of AdviceDetail](https://dev.azure.com/minghuama/GPT4IcM/_git/cloudgpt-oca?path=/predictor/views.py&version=GBmaster&line=47&lineEnd=76&lineStartColumn=5&lineEndColumn=55&lineStyle=plain&_a=contents)
  * [build a historical data vector indices](https://dev.azure.com/minghuama/GPT4IcM/_git/cloudgpt-oca?path=/predictor/views.py&version=GBmaster&line=87&lineEnd=92&lineStartColumn=5&lineEndColumn=23&lineStyle=plain&_a=contents)
  * [get the k=num nearest incidents by Fasttext model](https://dev.azure.com/minghuama/GPT4IcM/_git/cloudgpt-oca?path=/predictor/views.py&version=GBmaster&line=117&lineEnd=122&lineStartColumn=5&lineEndColumn=107&lineStyle=plain&_a=contents)
  * [construct the final prompt and call GPT](https://dev.azure.com/minghuama/GPT4IcM/_git/cloudgpt-oca?path=/predictor/views.py&version=GBmaster&line=123&lineEnd=157&lineStartColumn=5&lineEndColumn=55&lineStyle=plain&_a=contents)
  * [parse the response from GPT and add incoming incidents into historical data](https://dev.azure.com/minghuama/GPT4IcM/_git/cloudgpt-oca?path=/predictor/views.py&version=GBmaster&line=158&lineEnd=187&lineStartColumn=5&lineEndColumn=83&lineStyle=plain&_a=contents)



