import re
from urlextract import URLExtract
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont

extract = URLExtract()

FATHATAN = u'\u064b'
DAMMATAN = u'\u064c'
KASRATAN = u'\u064d'
FATHA = u'\u064e'
DAMMA = u'\u064f'
KASRA = u'\u0650'
SHADDA = u'\u0651'
SUKUN = u'\u0652'
LAM = u'\u0644'
ALEF = u'\u0627'
HAMZA = u'\u0621'
TATWEEL = u'\u0640'
TEH_MARBUTA = u'\u0629'
HEH = u'\u0647'
LAM_ALEF = u'\ufefb'
LAM_ALEF_HAMZA_ABOVE = u'\ufef7'
LAM_ALEF_HAMZA_BELOW = u'\ufef9'
LAM_ALEF_MADDA_ABOVE = u'\ufef5'
SIMPLE_LAM_ALEF = u'\u0644\u0627'
ALEF_MADDA = u'\u0622'
ALEF_HAMZA_ABOVE = u'\u0623'
WAW_HAMZA = u'\u0624'
ALEF_HAMZA_BELOW = u'\u0625'
HAMZA_ABOVE = u'\u0654'
ALEF_MAKSURA = u'\u0649'
YEH = u'\u064a'
YEH_HAMZA = u'\u0626'
HAMZA_BELOW = u'\u0655'
SIMPLE_LAM_ALEF_HAMZA_ABOVE = u'\u0644\u0623'
SIMPLE_LAM_ALEF_HAMZA_BELOW = u'\u0644\u0625'
SIMPLE_LAM_ALEF_MADDA_ABOVE = u'\u0644\u0622'

HARAKAT_PAT = re.compile(u"["+u"".join([FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SUKUN,SHADDA])+u"]")
LAMALEFAT_PAT = re.compile(u"["+u"".join([LAM_ALEF,LAM_ALEF_HAMZA_ABOVE,LAM_ALEF_HAMZA_BELOW, LAM_ALEF_MADDA_ABOVE])+u"]")
ALEFAT_PAT = re.compile(u"["+u"".join([ALEF_MADDA, ALEF_HAMZA_ABOVE,ALEF_HAMZA_BELOW, HAMZA_ABOVE,HAMZA_BELOW])+u"]")
HAMZAT_PAT = re.compile(u"["+u"".join([WAW_HAMZA, YEH_HAMZA])+u"]")

def remove_non_arabic(text):
    return ' '.join(re.sub(u"[^\u0621-\u063A\u0640-\u0652 ]", " ", text,  flags=re.UNICODE).split())

def strip_tashkeel(text):
    text = HARAKAT_PAT.sub('', text)
    text = re.sub(u"[\u064E]", "", text,  flags=re.UNICODE) # fattha
    text = re.sub(u"[\u0671]", "", text,  flags=re.UNICODE) # waSla
    return text

def strip_tatweel(text):
    return re.sub(u'[%s]' % TATWEEL, '', text)

def normalize_lamalef(text):
    return LAMALEFAT_PAT.sub(u'%s%s'%(LAM, ALEF), text)

def normalize_hamza(text):
    text = ALEFAT_PAT.sub(ALEF, text)
    return HAMZAT_PAT.sub(HAMZA, text)

def normalize_spellerrors(text):
    text = re.sub(u'[%s]' % TEH_MARBUTA, HEH, text)
    return re.sub(u'[%s]' % ALEF_MAKSURA, YEH, text)

def normalize_arabic_text(text):
    text = remove_non_arabic(text)
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    text = normalize_lamalef(text)
    text = normalize_hamza(text)
    text = normalize_spellerrors(text)
    return text

def nd(x):
  x['Normalized_Text'] = normalize_arabic_text(x['Message'])
  return x

def sentiment(d):
    if d['Sentiment'] == 'positive':
        return 1
    if d['Sentiment'] == 'negative':
        return -1
    if d['Sentiment'] == 'neutral':
        return 0


def predict_multi_level(X, neu_vectorizer, neu_clf, vectorizer, clf):
    # return clf.predict(vectorizer.transform(X))
    neu_y_pred = neu_clf.predict(neu_vectorizer.transform(X))
    if len(X[neu_y_pred == 'NonNeutral']) > 0:
        y_pred = clf.predict(
            vectorizer.transform(X[neu_y_pred == 'NonNeutral']))  # classify non neutral into positive or negative
        neu_y_pred[neu_y_pred == 'NonNeutral'] = y_pred

    final_y_pred = neu_y_pred
    return final_y_pred

def fun_stats(selected_user, df):
    if selected_user != "Overall":
        df = df[df['User'] == selected_user]
    total_messages = df.shape[0]
    total_words = []
    for message in df['Normalized_Text']:
        total_words.extend(message.split())
    return total_messages, len(total_words)


def most_busy_users(data):
    x = data['User'].value_counts().head()
    data2 = round((data['User'].value_counts() / data.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'User': 'percetage'})
    return x,data2

def month_activity_map(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value'] == k]
    return df['month'].value_counts()

def week_activity_map(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value'] == k]
    return df['day_name'].value_counts()


def activity_heatmap(selected_user, df, k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value'] == k]

    # Creating heat map
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='Normalized_Text', aggfunc='count').fillna(0)
    return user_heatmap

def daily_timeline(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value']==k]
    # count of message on a specific date
    daily_timeline = df.groupby('only_date').count()['Normalized_Text'].reset_index()
    return daily_timeline

def monthly_timeline(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    df = df[df['value']==-k]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['Normalized_Text'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

def percentage(df,k):
    df = round((df['User'][df['value']==k].value_counts() / df[df['value']==k].shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    return df

def most_common_words(selected_user,df,k):
    f = open('list.txt','r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['User'] == selected_user]
    temp = df[df['User'] != 'group_notification']
    temp = temp[temp['Normalized_Text'] != '<Media omitted>\n']
    words = []
    for message in temp['Normalized_Text'][temp['value'] == k]:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    # Creating data frame of most common 20 entries
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def most_busy_users(df):
    x = df['User'].value_counts().head()
    df = round((df['User'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'User': 'percent'})
    return x, df

def pie_chart(data):
  red = [(0.8901960784313725, 0.10196078431372549, 0.10980392156862745)]
  orange =[(1.0, 0.4980392156862745, 0.0)]
  green =[(0.2, 0.6274509803921569, 0.17254901960784313)]
  labels = ["positive", "neutral", "negative"]
  plt.pie(data , labels = labels, colors=green+orange+red, autopct='%.0f%%')
  plt. show()

def get_word_cloud(self):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import arabic_reshaper
        from bidi.algorithm import get_display

        data = arabic_reshaper.reshape(' '.join(self.Normalized_Text))
        artext = get_display(data)
        # Create and generate a word cloud image:
        wordcloud = WordCloud(font_path='arial.ots',max_font_size=80, max_words=50, background_color="white").generate(
            artext)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


def get_word_cloud_negative(self):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import arabic_reshaper
        from bidi.algorithm import get_display

        data = arabic_reshaper.reshape(' '.join(self[self['Sentiment'] == 'negative'].Normalized_Text))
        artext = get_display(data)
        # Create and generate a word cloud image:
        wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=30, background_color="white").generate(artext)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

def get_word_cloud_positive(self):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import arabic_reshaper
        from bidi.algorithm import get_display

        data = arabic_reshaper.reshape(' '.join(self[self['Sentiment'] == 'positive'].Normalized_Text))
        artext = get_display(data)
        # Create and generate a word cloud image:
        wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=30, background_color="white").generate(
            artext)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


def get_word_cloud_neutral(self):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import arabic_reshaper
        from bidi.algorithm import get_display

        data = arabic_reshaper.reshape(' '.join(self[self['Sentiment'] == 'neutral'].Normalized_Text))
        artext = get_display(data)
        # Create and generate a word cloud image:
        wordcloud = WordCloud(font_path='arial', max_font_size=80, max_words=30, background_color="white").generate(
            artext)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


