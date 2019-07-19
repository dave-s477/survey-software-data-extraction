import re

from os import listdir
from os.path import join
from nltk import sent_tokenize
from xml.sax.saxutils import unescape

from data_utils.preprocess import preprocess_complete, preprocess_simple

class Tokenizer:
    """
    Tokenizer: Create BIO formatted text for a Plain Text with XML tagging.
    """
    def __init__(self, data_source=None, data_files=None, preprocessing=False, lower_casing=False):
        """
        Tokenizer init funct.
        
        Keyword Arguments:
            data_source {[str]} -- [base dir with data files] (default: {None})
            data_files {[str]} -- [list of files containing relevant text] (default: {None})
            preprocessing {bool} -- [apply custom preprocessing to text] (default: {False})
            lower_casing {bool} -- [lowercase all words by default] (default: {False})
        """
        self.prepro = preprocessing
        self.lower_casing = lower_casing
        self.data_source = data_source
        self.data_files = data_files
        self.plain_text = ''
        self.sentences = []
        self.processed_text = ''
        self.processed_tags = ''
        self.BIO_format = ''

        self.tag_pattern = re.compile(r"<\w*>")
        self.pmid_pattern = re.compile(r"<b>PMC\d*</b>")
        self.complete_pattern = re.compile(r".?<.*?>.*?</.*?>.?")
        self.complete_pattern_extraction = re.compile(r".?<.*?>.*?</.*?>[^<]*")
        self.start_pattern = re.compile(r"\(?<.*>")
        self.end_pattern = re.compile(r".*</.*>\)?")
        self.likely_error = re.compile(r"\s+\?+\s+")
        self.brackets = {')':'(',']':'[','}':'{'}

    def load_data(self):
        """
        Load data from initialized location, strip basic XML tags and section headings. 
        """
        if not self.data_files:
            self.data_files = listdir(self.data_source)
        if not self.plain_text:
            for f in self.data_files: 
                with open(join(self.data_source, f), 'r') as fo:
                    text = fo.read()
                    text = re.sub(r'</?html>', '', text)
                    text = re.sub(r'</?body.*>', '', text)
                    text = re.sub(r'</?fileFormat>', '', text)
                    text = re.sub(self.pmid_pattern, '', text)
                    text = re.sub(r'<br>', '', text)
                    text = re.sub(r'<p><hr><p>', '', text)
                    text = re.sub(r'Background|Introduction|Results|Discussion|Conclusion|Methods|Supporting\sInformation', '', text)
                    text = re.sub(self.likely_error, ' ', text)
                    self.plain_text = self.plain_text + text + '\n'

    def unescape_text(self):
        if not self.plain_text:
            raise(RuntimeError("No text to preprocess."))
        self.plain_text = unescape(self.plain_text)

    def preprocess_text(self):
        """
        Preprocess text.
        Either simple: Remove coding error from the dataset
        or normal: Regex replace certain tokens. 
        """
        if not self.plain_text:
            raise(RuntimeError("No text to preprocess."))
        if self.prepro:
            self.plain_text = preprocess_complete(self.plain_text)
        else:
            # Just replace dangerous characters
            self.plain_text = preprocess_simple(self.plain_text)

    def strip_xml_tags(self):
        return re.sub(r'<[^>]{1,15}>', '', self.plain_text)

    def split_to_sentences(self):
        """
        Applies NLTK's sentence tokenizer to split input to sentences.
        """
        if not self.sentences:
            sents = sent_tokenize(self.plain_text)
            self.sentences = sents

    def process_sentence(self, sentence, search_annotation=True):
        """
        Split a sentence into individual tokens and labels based on own implementation (step 1 of 2).
        Performs initial white space tokenization.

        Ignores some special tokens that occur in bioNerDS.
        
        Arguments:
            sentence {[str]} -- [plain text sentence to process]
        
        Keyword Arguments:
            search_annotation {bool} -- [indicates whether there is annotation in form of XML tags present in the sentence] (default: {True})
        
        Returns:
            [str],[str] -- [sequence of words],[sequence of tags]
        """
        words = ''
        tags = ''
        sentence = sentence.split()
        in_middle_of_token = False
        middle_tag_name = ''
        for token in sentence:
            if in_middle_of_token and search_annotation:
                if self.end_pattern.search(token):
                    token_split = re.split(r"</|>", token)
                    plain = token_split[0]
                    if token_split[-1].isalpha():
                        print(' '.join(sentence))
                        raise(RuntimeError("Potential inconsistent tagging."))
                    words = words + plain + token_split[-1] + ' '
                    tags = tags + 'I-' + middle_tag_name + ' '
                    in_middle_of_token = False
                    middle_tag_name = ''
                else:
                    words = words + token + ' '
                    tags = tags + 'I-' + middle_tag_name + ' '
            else:
                if not token or "MathType@" in token or "+=feaafiart1ev1" in token or token.count('?') >= 2 or (len(token) > 40 and not self.tag_pattern.search(token)):
                    continue
                if self.tag_pattern.search(token) and search_annotation:
                    if self.complete_pattern.search(token):
                        for token_occurrence in self.complete_pattern_extraction.findall(token):
                            cut_token = re.split(r'</|<|>', token_occurrence)
                            tag_name = cut_token[1]
                            plain = cut_token[2]
                            alpha_end_string = False
                            for i in cut_token[-1]:
                                if i.isalpha():
                                    words = words + cut_token[0] + plain + " " + cut_token[-1] + ' '
                                    tags = tags + 'B-' + tag_name + ' O '
                                    alpha_end_string = True
                                    break
                            if not alpha_end_string:
                                words = words + cut_token[0] + plain + cut_token[-1] + ' '
                                tags = tags + 'B-' + tag_name + ' '
                    elif self.start_pattern.search(token):
                        cut_token = re.split(r'</|<|>', token)
                        middle_tag_name = cut_token[1]
                        plain = cut_token[2]
                        words = words + plain + ' '
                        tags = tags + 'B-' + middle_tag_name + ' '
                        in_middle_of_token = True
                    else:
                        print("Dangerous: " + token)
                else: 
                    words = words + token + ' '
                    tags = tags + 'O ' 
        return words, tags

    def to_BIO(self, words, tags):
        """
        Create a BIO format from sentence split into tokens.
        Further splits up tokens for correct tokenization (step 2 of 2).
        Can be used to keep intact naming conventions, e.g. ACgl-(da20213), would not be split. 
        
        Arguments:
            words {[str]} -- [sentence represented as array of words]
            tags {[str]} -- [tags corresponding to sentence]
        
        Returns:
            [str] -- [sentence in BIO format]
        """
        words = words.split()
        tags = tags.split()
        bio_string = ''
        for word, tag in zip(words, tags):
            if re.search(r'^(\(|"|\')+', word):
                splitted = re.search(r'^(\(|"|\')+', word).group()
                for s in splitted:
                    bio_string = bio_string + s + " O\n"
                word = re.sub(r'^(\(|"|\')+', '', word)
            if re.search(r'(\.|,|;|\:|!|\?|"|\)|\'|\'s|/|-)+$', word) and len(word) > 1: 
                splitted = re.search(r'(\.|,|;|\:|!|\?|"|\)|\'|\'s|/|-)+$', word).group()
                word = re.sub(r'(\.|,|;|\:|!|\?|"|\)|\'|\'s|/|-)+$', '', word)
                bracket_stack = []
                for s in word:
                    if s in ['(', '[', '{']:
                        bracket_stack.append(s)
                    if bracket_stack and s in [')', ']', '}'] and self.brackets[s] == bracket_stack[-1]:
                        bracket_stack.pop()
                if bracket_stack:
                    for idx,s in enumerate(splitted):
                        if s in self.brackets.keys() and self.brackets[s] == bracket_stack[-1]:
                            bracket_stack.pop()
                            word += s
                            splitted = splitted[:idx] + splitted[idx+1:]
                            if not bracket_stack:
                                break
                if word: 
                    bio_string = bio_string + word + " " + tag + "\n"
                for idx,s in enumerate(splitted):
                    bio_string = bio_string + s + " O\n"
            else:
                if word:
                    bio_string = bio_string + word + " " + tag + "\n"
        return bio_string

    def process_data(self, search_annotation=True):
        """
        Transform text into BIO format sentence by sentence. 
        
        Keyword Arguments:
            search_annotation {bool} -- [whether annotations are given as XML in the text] (default: {True})
        """
        if not self.processed_text:
            for sentence in self.sentences:
                words, tags = self.process_sentence(sentence, search_annotation=search_annotation)
                self.processed_text = self.processed_text + words + '\n'
                self.processed_tags = self.processed_tags + tags + '\n'
                bio_string = self.to_BIO(words, tags)
                self.BIO_format = self.BIO_format + bio_string + '\n'

    def write_BIO(self, location, name='BIO_format.txt'):
        """
        Output BIO format.
        
        Arguments:
            location {[type]} -- [path]
        
        Keyword Arguments:
            name {str} -- [file name to write] (default: {'BIO_format.txt'})
        """
        if not self.BIO_format:
            raise(RuntimeError("Data to write has not yet been created."))
        with open(join(location, name), 'w') as bio_out:
            bio_out.write(self.BIO_format)

    def write_BIO_utf8(self, location, name='BIO_format.txt'):
        """
        Output BIO format with UTF-8 encoding.
        
        Arguments:
            location {[type]} -- [path]
        
        Keyword Arguments:
            name {str} -- [file name to write] (default: {'BIO_format.txt'})
        """
        if not self.BIO_format:
            raise(RuntimeError("Data to write has not yet been created."))
        with open(join(location, name), 'w') as bio_out:
            bio_out.write(self.BIO_format.encode('utf8'))

    def process_single_sentence(self, sentence):
        """
        Process a single sentence in the tokenizer.
        
        Arguments:
            sentence {[str]} -- [sentence to process]
        
        Returns:
            [str] -- [list of tokens]
        """
        self.plain_text = sentence
        self.unescape_text()
        self.preprocess_text()
        self.split_to_sentences()
        self.process_data()
        token_list = []
        for idx, line in enumerate(self.BIO_format.split()):
            if idx % 2 == 0:
                token_list.append(line.split()[0])
        return token_list

    def word_tokenize(self, sentence):
        """
        Basic word tokenization.
        Based on white space split and further split up of individual tokens.
        Designed to split restrictive and keep naming conventions intact.
        
        Arguments:
            sentence {[str]} -- [sentence to tokenize]
        
        Returns:
            [str] -- [tokens]
        """
        tokens = []
        sentence = sentence.split()
        for token in sentence:
            if re.search(r'^(\(|"|\')+', token):
                splitted = re.search(r'^(\(|"|\')+', token).group()
                for s in splitted:
                    tokens.append(s)
                token = re.sub(r'^(\(|"|\')+', '', token)
            if re.search(r'(\.|,|;|\:|!|\?|"|\)|\'|\'s)+$', token) and len(token) > 1: 
                splitted = re.search(r'(\.|,|;|\:|!|\?|"|\)|\'|\'s)+$', token).group()
                token = re.sub(r'(\.|,|;|\:|!|\?|"|\)|\'|\'s)+$', '', token)
                bracket_stack = []
                for s in token:
                    if s in ['(', '[', '{']:
                        bracket_stack.append(s)
                    if bracket_stack and s in [')', ']', '}'] and self.brackets[s] == bracket_stack[-1]:
                        bracket_stack.pop()
                if bracket_stack:
                    for idx,s in enumerate(splitted):
                        if s in self.brackets.keys() and self.brackets[s] == bracket_stack[-1]:
                            bracket_stack.pop()
                            token += s
                            splitted = splitted[:idx] + splitted[idx+1:]
                            if not bracket_stack:
                                break
                tokens.append(token)
                for idx,s in enumerate(splitted):
                    tokens.append(s)
            else:
                tokens.append(token)
        return tokens