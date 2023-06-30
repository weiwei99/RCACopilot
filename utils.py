import fasttext
import numpy as np
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
from nltk.corpus import stopwords
import cufflinks as cf
from transformers import AutoTokenizer
import warnings
from bs4 import BeautifulSoup
cf.set_config_file(offline=True)
warnings.filterwarnings('ignore')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text
def text_clean(text):# used for fasttext
    cache_english_stopwords = stopwords.words('english')
    cache_english_stopwords+=['!', ',', '.', '?', '-s', '-ly', '</s> ', 's','[',']',':','(',')','{','}','\'','<','>','+','-','_','__','--','|','\'\'']
    # Remove HTML tags (e.g. &amp;)
    text_no_special_entities = re.sub(r'\&\w*;|#\w*|@\w*', '', text)
    # Remove certain value symbols
    text_no_tickers = re.sub(r'\$\w*', '', text_no_special_entities)
    # Remove hyperlinks
    text_no_hyperlinks = re.sub(r'https?:\/\/.*\/\w*', '', text_no_tickers)
    # # Remove some specialized abbreviation words, in other words, words with fewer letters
    string4 = " ".join(re.findall(r'\b\w+\b', text_no_hyperlinks))
    # Tokenization
    tokens = word_tokenize(string4)
    # Remove stopwords
    list_no_stopwords = [i for i in tokens if i not in cache_english_stopwords]
    # Final filtered result
    text_filtered = ' '.join(list_no_stopwords)  # ''.join() would join without spaces between words.
    return text_filtered
def count_word_num(text):
    if text is np.nan:
        return 0
    else:
        return len(text.split())
def remove_numbers(text):
    cleaned_text = re.sub(r'\d+', '', text)
    return cleaned_text
def truncate_text(text, length):
    if text is np.nan:
        return text
    else:
        return ' '.join(text.split()[:length])
def remove_letters_and_redundant_chars(text):
    cleaned_text = re.sub(r'[a-zA-Z,:]+', '', text)
    return cleaned_text
def trainingFile_preprocess(df,test=False):

    label_list=df['Keyword'].apply(lambda x:'-'.join(x.split()))
    document_list=df['metadata']
    text_filtered_list=document_list.apply(lambda x:text_clean(str(x['KeyValuePairs'])+' '+x['AlertType']+' '+str(x['ScopeType'])+' '+x['AdviceDetail']))  #str(x['KeyValuePairs'])+' '+
    if not test:
        file=np.array('__label__'+label_list+' '+text_filtered_list)
    else:
        file=np.array(text_filtered_list)
    return file

def clean(text,max_len=300):
    content = text
    content=remove_html_tags(content)
    content=remove_numbers(content)
    # Replace all whitespaces
    content = re.sub(r'\&\w*;|#\w*|@\w*', '', content)
    content = content.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ').replace('\\\\', ' ')
    content = content.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\\', ' ')
    # Remove all quotes and brackets
    content = content.replace('\"', ' ').replace("\'", ' ').replace('(', ' ').replace(')', ' ')
    # Remove all vertical bars
    content = content.replace('|', ' ')
    # Replace all consecutive '-'s with only one '-'
    content = re.sub('-+', '-', content)
    # Remove filenames if its extension is in 'file_exts.txt'
    common_file_extensions = open('Data_Models/file_exts.txt', 'r').read().splitlines()
    content = ' '.join([word for word in content.split() if '.' + word.split('.')[-1] not in common_file_extensions])
    # If there are multiple whitespaces, replace with one whitespace
    content =  re.sub(' +', ' ', content)
    content = re.sub(r"http\S+", '', content)
    content = re.sub(r'\w{8}-\w{4}-\w{4}-\w{4}-\w{12}', ' ', content)
    content = truncate_text(content, max_len) if count_word_num(content) > max_len else content
    content = content.replace(' ','')
    # content = content[:6000]
    return content

def process(text):
    # match = re.search(r'(?:Summary|Output):(.*?\n(?:\n.*)*)', text, re.DOTALL)
    match = re.search(r'Summary:(.*?)\n|Output:(.*?)\n\n', text, re.DOTALL)
    output_text=''
    if match:
        summary_text = match.group(1)
        if summary_text:
            output_text=summary_text
        else:
            output_text=match.group(2)
    else:
        return text
    if not output_text:
        match=re.search(r'Summary:\n(.*?)\n',text,re.DOTALL)
        if match:
            output_text=match.group(1)
        else:output_text=' '
    output_text=output_text.replace('\n',' ')
    if output_text==' ' or output_text=='':
        return text
    return output_text
def fasttext_predict(input):
    model = fasttext.load_model('Data_Models/models')
    ft_label = model.predict(input)[0][0]
    ft_label = ft_label[9:]
    return ft_label