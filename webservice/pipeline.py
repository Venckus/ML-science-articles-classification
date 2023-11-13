import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_tags,
    strip_punctuation,
    remove_stopwords,
    strip_multiple_whitespaces
)
from sklearn.feature_extraction.text import TfidfTransformer, HashingVectorizer
from sklearn.pipeline import Pipeline

class FullPipeline:
    def __init__(self, title: str, abstract: str):
        self.text: str = f"{title} {abstract}"
        self.data_preprocess = DataPreprocess()

    def run(self):
        preprocessed = self.data_preprocess(self.text)
        # print(preprocessed)
        self.data = self.get_pipe([preprocessed]).transform([preprocessed])
    
    def get_pipe(self, x_data) -> Pipeline:  #, n_features=2097152):
        hasher = HashingVectorizer(ngram_range=(1,2), n_features=2097152)
        tfidf_transformer = TfidfTransformer(use_idf=True)
        return Pipeline([('hash', hasher), ('tfidf', tfidf_transformer)]).fit(x_data)
    

class DataPreprocess:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

        self.filters = [strip_tags,
                       strip_punctuation,
                       strip_multiple_whitespaces,
                       remove_stopwords]

    def __call__(self, doc) -> str:
    #     clean_words = self.__apply_filter(doc)
    #     return clean_words

    # def __apply_filter(self, doc) -> str:
        try:
            cleanse_words = set(preprocess_string(doc, self.filters))
            filtered_words = set(self.wnl.lemmatize(word, 'v') for word in cleanse_words)
            return ' '.join(filtered_words)
        except TypeError as te:
            raise(TypeError("Not a valid data {}".format(te)))