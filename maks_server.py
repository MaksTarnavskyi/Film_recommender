
# coding: utf-8

# # Server

# ### Import libraries

# In[1]:


import numpy as np
import pandas as pd
import time
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from tabulate import tabulate

import sys

import warnings; warnings.simplefilter('ignore')


# ### My functions

# Weighted Rating (WR) = $(\frac{v}{v + m} . R) + (\frac{m}{v + m} . C)$
# 
# where,
# * *v* is the number of votes for the movie
# * *m* is the minimum votes required to be listed in the chart
# * *R* is the average rating of the movie
# * *C* is the mean vote across the whole report

# In[2]:


def short_title(x):
    if len(x) > 42:
        x = x[:39]
        x = x+'...'
    return x   

def check_input(ans,start,end):
    try:
        value = int(ans)
        if value in range(start,end+1):
            return False
        else:
            print('Number not in range (',start,',',end,')')
            return True
    except ValueError:
        print('Not a number')
        return True

def create_tab(films,num_films):
    show_tab = films.loc[:,header[1:]].values[:int(num_films)]
    index_s = np.arange(1,len(show_tab) + 1).reshape((-1,1))
    new_tab_g = np.hstack((index_s,show_tab))
    tab_g = tabulate(new_tab_g,header, tablefmt="pipe")
    return tab_g

def choose_genre(genre_id,year):
    str_welcome = '\nThe best films from '
    if genre_id == '0':
        return build_chart_all(year), str_welcome + 'All genres:\n'
    else:
        genre = genres[int(genre_id)-1]
        return build_chart(genre,year), str_welcome + '"'+genre+'":\n'
    
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

def build_chart_all(year):
    vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = md[md['vote_average'].notnull()]['vote_average']
    C = vote_averages.mean()
    m = vote_counts.quantile(0.95)
    qualified = md[(md['vote_count'] >= m) & (md['year'] >= int(year)) &(md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'genre']]
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified = qualified.sort_values('wr', ascending=False)
    
    return qualified

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

def build_chart(genre, year,percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    qualified = df[(df['vote_count'] >= m) & (df['year'] >= int(year)) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'genre']]
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified = qualified.sort_values('wr', ascending=False)
    
    return qualified

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year','genre']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False)
    return qualified

def to_int(x):
    if x == 'NaT':
        return 0
    else: 
        return int(x)
    
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
    
def choose_one_genre(x):
    if len(x) == 0:
        return 'Foreign'
    else:
        y = x[0]
        if y in bad_genres:
            return 'Foreign'
        else:
            return y


# ### Import data

# In[3]:


md = pd. read_csv('data/movies_metadata.csv')
md.head(1)


# In[4]:


md.drop_duplicates()

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
md['year'] = md['year'].apply(to_int)
md['title'] = md['title'].apply(str)
md['title'] = md['title'].apply(short_title)

s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)

genres = s.unique()
bad_genres = genres[20:]
md['genre'] = md['genres'].apply(choose_one_genre)


# ## Metadata Based Recommender

# In[5]:


links_small = pd.read_csv('data/links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

md['id'] = md['id'].apply(convert_int)
del_nul = md[md['id'].isnull()].index
md = md.drop(del_nul)
smd = md[md['id'].isin(links_small)]
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')

credits = pd.read_csv('data/credits.csv')
keywords = pd.read_csv('data/keywords.csv')

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd = md[md['id'].isin(links_small)]
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x])

s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
s = s[s > 1]

stemmer = SnowballStemmer('english')
smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))


# In[6]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
smd = smd.reset_index()
titles = smd['title']
upper_titles = smd['title'].values
UP_titles = [upper_titles[i].upper() for i in range(len(upper_titles))]
indices = pd.Series(smd.index, index=UP_titles)


# # Program

# In[7]:


welcome = '---------------------------------------------------------\n\nIn this program you can find\n1 - the best films of a certain genre or all genres\n2 - simmilar films to the given one\n3 - random films'
header = ['place','title','year','vote_count','vote_average','genre']


# In[8]:


def prog():
    print('    Welcome to the film recommender!')

    while 1:
        print(welcome)
        time.sleep(0.5)
        ans = input("\nInput number of the mode: ")
        if ans == '1':
            print('\nChoose one of the genres:')
            print(0,'\tAll genres')
            for i,genre in enumerate(genres[:20]):
                print(i+1,'\t'+genre)

            print()
            genre_id = input("Input genre id: ")
            if check_input(genre_id,0,20):
                break

            print('\nHow many films do you want to see in the list (from 1 to 1000)')
            num_films = input("Input number of films: ")
            if check_input(num_films,1,1000):
                break

            print('\nInput the year of the oldest film: (from 0 to 2017)')
            year = input("Input year of films: ")
            if check_input(year,0,2018):
                break

            films, str_welcome = choose_genre(genre_id,year)
            tab = create_tab(films,num_films)
            print(str_welcome)
            print(tab)

        elif ans == '2':
            print()
            title = input("Write the name of the film: ")
            if title.upper() in UP_titles:
                films = improved_recommendations(title.upper())
                tab = create_tab(films,'10')
                idx = indices[title.upper()]
                print('\nThe similar films for film "' +titles[idx]+'":\n')
                print(tab)
            else:
                print("Sorry, I don't know these film")
        elif ans == '3':
            num_films = input("\nInput number of films: ")
            if check_input(num_films,0,1000):
                break
            print('\n\nInput the year of the oldest film: (if all input 0)')
            year = input("Year of the film: ")
            if check_input(year,0,2017):
                break
            films = md[md.year >= int(year)]
            films.index = range(len(films))
            rand_index = np.random.randint(0,len(films),int(num_films))
            new_films= films.loc[rand_index,:]
            tab = create_tab(new_films,num_films)
            print(tab)
        else:
            print("Wrong input :(\n")
        rep = input("\nSomething else? (Y)")
        if rep.upper() == 'Y':
            continue
        break
    print('\nGood bye :)')   


# In[9]:


#prog()


# # Server

# In[10]:


import socket

def send_table(tab,mode ='tab',ans = 'OK'):
    send(mode)
    if receive() == ans:
        step = 256
        i = 0
        l = len(tab)
        while i < l:
            send(tab[i:i+step])
            i += step
            if receive() != ans:
                print('eror tab part')
        send('end')
        if receive() == 'wait':
            while 1:
                res = receive()
                if len(res) > 0:
                    return res
                    break
        else:
            print('eror tab end')
        
    else:
        print('error tab start')
            

def send_msg(text,mode,ans):
    send(mode)
    if receive() == ans:
        send(text)
        rec = receive()
        if rec == 'wait':
            while 1:
                res = receive()
                if len(res) > 0:
                    return res
                    break
        if rec == ans:
            return rec
        else:
            print('error 2')
            return 'error'
    else:
        print('error 1')
        return 'error'
    
def receive():
    data = conn.recv(1024)
    udata = data.decode("utf-8")
    return udata

def send(text):
    conn.sendall(text.encode("utf-8"))
    
def check_input_server(ans,start,end):
    try:
        value = int(ans)
        if value in range(start,end+1):
            return False
        else:
            text = 'Number not in range ('+str(start)+','+str(end)+')'
            send_msg(text,'pr','OK')
            return True
    except ValueError:
        send_msg('Not a good number','pr','OK')
        return True


# # Reccomender

# In[ ]:


sock = socket.socket()
sock.bind(('', 9091))
sock.listen(1)
conn, addr = sock.accept()
print('connected:', addr)

send('Welcome to the film recommender!')
ans = receive()
if ans == 'Start':
    while 1:
        send_msg(welcome,'pr','OK')
        text = '\nInput number of the mode: '
        mode = send_msg(text,'in','OK')
        if mode == '1':
            text = '\nChoose one of the genres:'
            text += '\n0\tAll genres\n'
            for i,genre in enumerate(genres[:20]):
                text += str(i+1) + '\t'+genre + '\n'
            text += '\n'
            send_msg(text,'pr','OK')

            genre_id = send_msg('Input genre id: ','in','OK')
            if check_input_server(genre_id,0,20):
                break
            
            text = '\nHow many films do you want to see in the list (from 1 to 1000)'
            send_msg(text,'pr','OK')
            text = "Input number of films: "
            num_films = send_msg(text,'in','OK')
            if check_input_server(num_films,0,1000):
                break
            
            text = '\nInput the year of the oldest film: (from 0 to 2017)'
            send_msg(text,'pr','OK')
            text = "Input year of films: "
            year = send_msg(text,'in','OK')
            if check_input_server(year,0,2017):
                break
                
            films, str_welcome = choose_genre(genre_id,year)
            tab = create_tab(films,num_films)

            send_table(tab,'tab','OK')
            
        elif mode == '2':
            title = send_msg("\nWrite the name of the film: ",'in','OK')
            if title.upper() in UP_titles:
                films = improved_recommendations(title.upper())
                tab = create_tab(films,'10')
                idx = indices[title.upper()]
                
                text = '\nThe similar films for film "' +titles[idx]+'":\n'
                send_msg(text,'pr','OK')
                send_table(tab,'tab','OK')
                
            else:
                text = "\nSorry, I don't know these film"
                send_msg(text,'pr','OK')
        elif mode == '3':
            text = '\nHow many films do you want to see in the list (from 1 to 1000)'
            send_msg(text,'pr','OK')
            text = "Input number of films: "
            num_films = send_msg(text,'in','OK')
            if check_input_server(num_films,0,1000):
                break
                
            text = '\nInput the year of the oldest film: (from 0 to 2017)'
            send_msg(text,'pr','OK')
            text = "Input year of films: "
            year = send_msg(text,'in','OK')
            if check_input_server(year,0,2017):
                break
            
            films = md[md.year >= int(year)]
            films.index = range(len(films))
            rand_index = np.random.randint(0,len(films),int(num_films))
            new_films= films.loc[rand_index,:]
            tab = create_tab(new_films,num_films)
            
            send_table(tab,'tab','OK')
        else:
            text = "Wrong input :(\n"
            send_msg(text,'pr','OK')
            
        text = "\nSomething else? (Y)"
        repeat = send_msg(text,'in','OK')
        if repeat.upper() == 'Y':
            continue
        break
        
    send_msg("Good bye :)",'pr','OK')
    send('bye') 


conn.close()
sock.close()

