# Introduction 
This repo implements OCA with LLM, and remote requests receive FastText prediction results and similar incidents returned by GPT(the framework is Django)

The built model is based on the 700+data and if you have another dataset, Please run data_base_build.py to retrain the model.

The function implementation is mainly in the views. py file, Data_ Models store models and data

# Getting Started
1. python manage.py runserver

2. Software dependencies: python 3.10.9 win11

3. API Example:

   ```python
   data=DatrFrame()
   row_data=data.iloc[0].to_json()[0]
   url='http://127.0.0.1:8000/predict/'
   response=requests.post(url, data=row_data, headers={'Content-Type': 'application/json'})
   ```

4.Note that you should set your environment variables on the server like:
```
export OPENAI_API_KEY='your API_KEY'
```
