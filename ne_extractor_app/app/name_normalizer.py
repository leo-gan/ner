import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# # Download required NLTK resources
# nltk.download('stopwords')
# nltk.download('wordnet')


class Normalizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))
        self.org_suffixes = {"co", "corp", "inc", "ltd", "llc", "plc"}

    def compact_name(self, name):
        # Convert to lowercase and remove punctuation
        name = name.lower().translate(str.maketrans("", "", string.punctuation))
        # Tokenize and stem words, remove stop words and organization-type words
        tokens = name.split()
        normalized_tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and token not in self.org_suffixes
        ]
        # Sort the tokens alphabetically to capture names with words in different order
        return " ".join(sorted(normalized_tokens))

    def is_included(self, normalized_to_original, normalized):
        if not normalized:
            return
        normalized_set = set(normalized.split())
        for k in normalized_to_original:
            k_set = set(k.split())
            if normalized_set.issubset(k_set):
                return k
            elif k_set.issubset(normalized_set) and len(k_set) > 1:
                # replace the key with the longer name
                # only for 2+ word similarity
                # only when the long name is the first in the collection
                normalized_to_original[normalized] = normalized_to_original.pop(k)
                return normalized
        return None

    def normalize_names(self, name_list):
        names = name_list.split(";")

        normalized_to_original = {}
        for name in names:
            normalized = self.compact_name(name)
            if k := self.is_included(normalized_to_original, normalized):
                # Keep the longest original name
                if len(name) > len(normalized_to_original[k]):
                    normalized_to_original[k] = name
            else:
                normalized_to_original[normalized] = name
        return ";".join(normalized_to_original.values())
