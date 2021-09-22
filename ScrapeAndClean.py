import pickle
import urllib
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import pandas as pd


data={}

def url_to_transcript(url):

    text=[]
    site = urllib.request.urlopen(url)
    content = site.read()
    soup = BeautifulSoup(content, 'lxml')
    table = soup.find_all('p')
    
    for i in table:
        text.append(str(i.text))
    print(url)
    return text


# URLs of transcripts in scope
urls = ['https://avalon.law.yale.edu/18th_century/wash1.asp',
        "https://avalon.law.yale.edu/20th_century/froos1.asp",
        "https://avalon.law.yale.edu/20th_century/kennedy.asp",
        "https://avalon.law.yale.edu/20th_century/carter.asp",
        "https://avalon.law.yale.edu/21st_century/gbush1.asp",
        "https://avalon.law.yale.edu/21st_century/obama.asp",
        "https://avalon.law.yale.edu/20th_century/taft.asp",
        "https://avalon.law.yale.edu/19th_century/jefinau1.asp",
        "https://avalon.law.yale.edu/20th_century/bush.asp",
        "https://avalon.law.yale.edu/20th_century/eisen1.asp",
        "https://www.politico.com/story/2017/01/full-text-donald-trump-inauguration-speech-transcript-233907",
        "https://www.whitehouse.gov/briefing-room/speeches-remarks/2021/01/20/inaugural-address-by-president-joseph-r-biden-jr/"
        ]

presidents = ["Washington", "Roosvelt", "Kennedy", "Carter","BushJr", "Obama", "Taft", "Jefferson", "BushSr", "Eisenhower","Trump","Biden"]

transcripts = [url_to_transcript(u) for u in urls]

for i, p in enumerate(presidents):
     with open("transcripts/" + p + ".txt", "wb") as file:
         pickle.dump(transcripts[i], file)


for i, p in enumerate(presidents):
    with open("transcripts/" + p + ".txt", "rb") as file:
        data[p] = pickle.load(file)


#combine the transcript into one chunk of text
def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text


combined_data = {key: [combine_text(value)] for (key, value) in data.items()}


pd.set_option('max_colwidth', 600)

data_df = pd.DataFrame.from_dict(combined_data).transpose()
data_df.columns = ['speeches']
data_df = data_df.sort_index()




# Apply a first round of text cleaning techniques
def TextCleaning(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)  # remove square brackets
    text = re.sub('[%s]' % re.escape(string.punctuation),'', text)  # remove puncitutations marks
    text = re.sub('\w*\d\w*', '', text)  # remove words that contain numbers
    text = re.sub('[‘’“”…]', '', text)  # remove quotes
    text = re.sub('\r', '', text)  # remove \r
    text = re.sub('\n', '', text)  # remove \n
    text = re.sub('\t', '', text)  # remove \t
    return text


def Clean(x): return TextCleaning(x)


data_clean = pd.DataFrame(data_df.speeches.apply(Clean))


data_df['full_name'] = presidents
data_df.to_pickle("corpus.pkl")


cv = CountVectorizer(stop_words='english')
data_cv = cv.fit_transform(data_clean.speeches)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index #document-term matrix
data_dtm.to_pickle("dtm.pkl")
data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))








