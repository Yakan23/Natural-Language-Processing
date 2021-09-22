import math
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
import pandas as pd

data = pd.read_pickle('corpus.pkl')


def pol(x): return TextBlob(x).sentiment.polarity
def sub(x): return TextBlob(x).sentiment.subjectivity


data['polarity'] = data['speeches'].apply(pol)
data['subjectivity'] = data['speeches'].apply(sub)
print(data)


# Let's plot the results

plt.rcParams['figure.figsize'] = [15, 10]

for index, president in enumerate(data.index):
    x = data.polarity.loc[president]
    y = data.subjectivity.loc[president]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data['full_name'][index], fontsize=10)
    plt.xlim(-1, 1)

plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()


# Split each routine into 10 parts


def split_text(text, n=10):
    '''Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.'''

    # Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)

    # Pull out equally sized pieces of text and put it into a list
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list


list_pieces = []
for t in data.speeches:
    split = split_text(t)
    list_pieces.append(split)

list_pieces


polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)

polarity_transcript


# Show the plot for all presidents
plt.rcParams['figure.figsize'] = [15, 12]

for index, comedian in enumerate(data.index):
    plt.subplot(3, 5, index+1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0, 10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.075, ymax=.5)

plt.show()
