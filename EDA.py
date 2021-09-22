import numpy as np
from sklearn.feature_extraction import text
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd

data = pd.read_pickle('dtm.pkl')
data = data.transpose() #transpose to term document matrix

# Get top 30 words used by each president
top_dict = {}
for p in data.columns:
    top = data[p].sort_values(ascending=False).head(30)
    top_dict[p] = list(zip(top.index, top.values))
    
# Append top 30 words used by each president to a list of words
words = []
for presidents in data.columns:
    top = [word for (word, count) in top_dict[presidents]]
    for t in top:
        words.append(t)

Counter(words).most_common()

add_stop_words = [word for word, count in Counter(words).most_common() if count > 7]

# Read in cleaned data
data_clean = pd.read_pickle('data_clean.pkl')

# Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.speeches)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

# Pickle it for later use
pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")


# Create a wordcloud object
wc = WordCloud(stopwords=stop_words, background_color="white",colormap="Dark2", max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [16, 6]



full_names = ["George Washington", "Franklin Roosevelt", "John F. Kennedy", "Jimmy Carter", "George W. Bush",
              "Barack Obama", "William Taft", "Thomas Jefferson", "George Bush Sr.", "Dwight D. Eisenhower","Donald Trump","Joe Biden"]


# Create subplots for each presidents
for index, president in enumerate(data.columns):
    wc.generate(data_clean.speeches[president])
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(full_names[index])

plt.show()



# Find the number of unique words used by each president
# by determinig the non zero columns for each president
unique_list = []
for presidents in data.columns:
    uniques = data[presidents].to_numpy().nonzero()[0].size
    unique_list.append(uniques)



# Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['presidents', 'unique_words'])
data_unique_sort = data_words.sort_values(by='unique_words')


# Let's plot our findings
y_pos = np.arange(len(data_words))
plt.subplot(1, 2, 1)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos, data_unique_sort.presidents)
plt.title('Number of Unique Words', fontsize=20)


plt.tight_layout()
plt.show()
