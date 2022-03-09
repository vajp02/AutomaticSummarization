#pip install beautifulsoup4
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
import re 
from io import StringIO
from html.parser import HTMLParser
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess(text):
    """
    remove unwanted characters fom a string
    """
    
    text = re.sub(r"\n|\t|”|“|'|\xa0|[Bb]y", " ", text)
    text = re.sub(r'"', " ", text)
    text = re.sub(r" +", " ", text)
    return text

class MLStripper(HTMLParser):
    """
    Strip data scraped from web to get fluent text
    """
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = True
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    """
    apply stripper to remove certain tags
    """
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def get_href(stat):
    """
    Find element which have href attr and 
    print href value
    """
    soup = BeautifulSoup(stat, 'html.parser')
    el = soup.find(href=True)
    return el['href']

def get_statement(stat):
    """
    extract stamenet/claim
    """
    soup = BeautifulSoup(stat, 'html.parser')
    el = soup.find('a').contents[0]
    return preprocess(el)

def get_long_short_texts(url_str):
    
    """
    first method to detect wheter a summary occurs in scraped text, 
    then split text into rulling comments and justification  
    """
    url = "https://www.politifact.com{}".format(url_str)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    statement_footer =  soup.find_all('article',attrs={'class':'m-textblock'})
    text_p =  statement_footer[0].find_all('p')
    
    text=""
    for each in text_p:
        text = text + " {}".format(each.text.strip())
        
    soup_tt = BeautifulSoup(str(text_p), 'html.parser')
    
    # Find element which have href attr
        
    for strong_tag in soup_tt.find_all('strong'):
        if strong_tag.text.strip() == 'Our ruling':
            
            text_splitted =  text.split('Our ruling')
            if len(text_splitted)==2:
                return text_splitted[0], text_splitted[1]
            else:
                return ValueError()
                
        elif strong_tag.text.strip() == 'Our rating':
            
            text_splitted =  text.split('Our rating')
            if len(text_splitted)==2:
                return text_splitted[0], text_splitted[1]
            else:
                return ValueError()
        else:
            return ValueError("Some problem has been shown  by spliting first method")
    

def get_tags(url_str):
    """
    extract tags from javascript
    """
    tags = []
    url = "https://www.politifact.com{}".format(url_str)
    response = requests.get(url)
    #print(response)
    soup = BeautifulSoup(response.text, "html.parser")
    try:
        
        statement_footer =  soup.find_all('li',attrs={'class':'m-list__item'})
        text_p =  statement_footer
        for tag in statement_footer:
            tags.append(tag.find_all('a',title=True)[0]['title'])
        return tags
    
    except:
        
        print("Problems with finding tags for {}".format(url))  
        
    return tags

def get_long_short_second_approach(url_str):  
    
    """
    second method to detect wheter a summary occurs in scraped text, 
    then split text into rulling comments and justification  
    """  
    
    def from_p_to_text(text):
        text_p = BeautifulSoup(text, "html.parser").find_all('p')
        text=""
        for each in text_p:
                text = text + " {}".format(each.text.strip())

        return strip_tags(str(text).strip())
    #url_str = "/factchecks/2021/oct/28/randy-feenstra/biden-administration-predicted-liquid-fuel-cars-ou/"
    url = "https://www.politifact.com{}".format(url_str)
    
    response = requests.get(url)
    #print(response)
    soup_res = BeautifulSoup(response.text, "html.parser")
    statement_foot =  soup_res.find_all('article',attrs={'class':'m-textblock'})
    long_short =  re.split('<div.*>.*Our ruling.*<\/div>', str(statement_foot[0])) # .split('Our ruling')
    if len(long_short) != 2:
        long_short =  re.split('<div.*>.*Our rating.*<\/div>', str(statement_foot[0]))
        if len(long_short) != 2:
            #print("Some problem has been shown the length ater split is {}".format(len(long_short)))
            raise ValueError("Some problem has been shown the length after split is {}".format(len(long_short)))
            #return None
        else:
            return from_p_to_text(preprocess(long_short[0])),preprocess(from_p_to_text(long_short[1]))    
    else:
        return from_p_to_text(preprocess(long_short[0])),preprocess(from_p_to_text(long_short[1]))
    
def get_long_short_third_approach(url_str):
    
    """
    third method to detect wheter a summary occurs in scraped text, 
    then split text into rulling comments and justification  
    """  
    
    url = "https://www.politifact.com{}".format(url_str)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    statement_footer =  soup.find_all('article',attrs={'class':'m-textblock'})
    text_p =  statement_footer[0].find_all('p')
    text=""
    for each in text_p:
            text = text + " {}".format(each.text.strip())
     
    if 'Our ruling' in text:
        text_splitted = text.split('Our ruling')
        if len(text_splitted)==2:
            return text_splitted[0], text_splitted[1]
        else:
            raise ValueError()

    elif 'Our rating' in text:
        text_splitted = text.split('Our rating')
        if len(text_splitted)==2:
            return text_splitted[0], text_splitted[1]
        else:
            raise ValueError()
    else:
        raise ValueError()

def reviewer_date(soup_in):
    """
    extract reviewer and date
    """
    data_lst = []
    statement_footer =  soup_in.find_all('footer',attrs={'class':'m-statement__footer'})
    for i in statement_footer:
        tt = re.sub(r"[Bb]y|\n|\t", " ", str(i.text)).split("•")
        data_lst.append([tt[0].strip(),tt[1].strip()])
    return data_lst

def set_graph(label_str,title, Order = False):
    """
    graph distribution of certain column
    """
    ax = plt.figure(figsize=(80,70)) 
    plt.xlabel('xlabel', fontsize=70)
    plt.ylabel("Krajina",fontsize=70)
    plt.xticks(rotation=90, fontsize=60)
    plt.yticks(fontsize=60)
    plt.title(title, fontsize=80, fontweight='bold', ha='center')
    if Order:
        sns.countplot(x=data_df_splitted[label_str], data = data_df_splitted, order = data_df_splitted[label_str].value_counts()[:50].index)
    else:
        sns.countplot(x = data_df_splitted[label_str], data = data_df_splitted)
    ax.savefig("photo/"+title + '.jpg',bbox_inches='tight')