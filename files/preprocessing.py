import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words= set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text,str):
        return ""
    
    text= text.lower()
    text=re.sub(r"[^a-zA-Z]", "", text)
    text=" ".join([word for word  in text.split() if word not in stop_words])
    
    return text
