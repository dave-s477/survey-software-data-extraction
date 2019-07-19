# -*- coding: utf-8 -*-
import re

def preprocess_token(token):
    """ Preprocess a single token """
    chars = {'ö':'oe','ä':'ae','ü':'ue', 'Ö':'Oe','Ä':'Ae','Ü':'Ue', "\[":"[", "\]":"]", '´': "'", '′': "'"}
    greek_alphabet = {u'\u0391': 'Alpha', u'\u0392': 'Beta', u'\u0393': 'Gamma', u'\u0394': 'Delta', u'\u0395': 'Epsilon', u'\u0396': 'Zeta', u'\u0397': 'Eta',
        u'\u0398': 'Theta', u'\u0399': 'Iota', u'\u039A': 'Kappa', u'\u039B': 'Lamda', u'\u039C': 'Mu', u'\u039D': 'Nu', u'\u039E': 'Xi', u'\u039F': 'Omicron',
        u'\u03A0': 'Pi', u'\u03A1': 'Rho', u'\u03A3': 'Sigma', u'\u03A4': 'Tau', u'\u03A5': 'Upsilon', u'\u03A6': 'Phi', u'\u03A7': 'Chi', u'\u03A8': 'Psi',
        u'\u03A9': 'Omega', u'\u03B1': 'alpha', u'\u03B2': 'beta', u'\u03B3': 'gamma', u'\u03B4': 'delta', u'\u03B5': 'epsilon', u'\u03B6': 'zeta', u'\u03B7': 'eta',
        u'\u03B8': 'theta', u'\u03B9': 'iota', u'\u03BA': 'kappa', u'\u03BB': 'lamda', u'\u03BC': 'mu', u'\u03BD': 'nu', u'\u03BE': 'xi', u'\u03BF': 'omicron',
        u'\u03C0': 'pi', u'\u03C1': 'rho', u'\u03C3': 'sigma', u'\u03C4': 'tau', u'\u03C5': 'upsilon', u'\u03C6': 'phi', u'\u03C7': 'chi', u'\u03C8': 'psi', u'\u03C9': 'omega',
    }
    common_unicode_list = [u'\u2013', u'\u201d', u'\u201c']

    harvard_citation = re.compile(r"^\(([\w\&\.\s]+,?\s\d{4}(;\s+[\w\&\.\s]+,?\s\d{4})*)\)$", re.IGNORECASE)
    plain_citation = re.compile(r"^\[(\d{1,3}(,|;|-))*\d{1,3}\]$", re.IGNORECASE)
    paper_mention = re.compile(r"^\w+(\set\sal\.|\sand\s\w+|\s\&\s\w+)\s(\(|\[)\d{4}(\)|\])$", re.IGNORECASE)
    url_link = re.compile(r"^(https?|ftp)://[^\s/$.?#].[^\s),]*$", re.IGNORECASE)
    percentage = re.compile(r"^\d{2,4}([,|.]\d{1,3})?%$", re.IGNORECASE)
    integer_number = re.compile(r"^(\+|-)?([2-9][0-9]|\d{3,7}|(\d{1,3},)*\d{3})$", re.IGNORECASE)
    float_number = re.compile(r"^(\+|-)?\d+\.\d+$", re.IGNORECASE)
    in_paper_ref = re.compile(r"^\((see(\w|\s){2,20}$|^table\s(\w|\d|\s){1,5}$|^tab.\s(\w|\d|\s){1,5}$|^figure\s(\w|\d|\s){1,5}$|^fig.\s(\w|\d|\s){1,5})\)$", re.IGNORECASE)
    gen_sequence = re.compile(r"^(5('|´|′)|3('|´|′))-?([gtcaun]|\[[gtcaun]\/[gtcaun]\]){5,}-?(5('|´|′)|3('|´|′))?$", re.IGNORECASE)
    
    def intrepl(matchobj):
        minus = matchobj.group(0).count(',')
        return 'intertoken' + str(len(matchobj.group(0)) - minus)

    #for char in chars:
    #    token = token.replace(char,chars[char])
    for letter in greek_alphabet:
        token = token.replace(letter,greek_alphabet[letter])
    token = re.sub(harvard_citation, 'citetoken', token)
    token = re.sub(plain_citation, 'citetoken', token)
    token = re.sub(paper_mention, 'papertoken', token)   
    token = re.sub(url_link, 'urltoken', token)    
    token = re.sub(percentage, 'percentagetoken', token)
    token = re.sub(float_number, 'floattoken', token)
    token = re.sub(integer_number, intrepl, token)
    token = re.sub(in_paper_ref, 'referencetoken', token)
    token = re.sub(gen_sequence, "genseqtoken", token)
    token = token.lower()
    return token

def preprocess_complete(text):
    """ 
    Perform regex replacements on a text.
    Regex capture certain umbrella terms.
    Intended for training of new word embeddings, to capture semantic equal words.
    """
    chars = {'ö':'oe','ä':'ae','ü':'ue', 'Ö':'Oe','Ä':'Ae','Ü':'Ue', "\[":"[", "\]":"]", '´': "'", '′': "'"}
    greek_alphabet = {u'\u0391': 'Alpha', u'\u0392': 'Beta', u'\u0393': 'Gamma', u'\u0394': 'Delta', u'\u0395': 'Epsilon', u'\u0396': 'Zeta', u'\u0397': 'Eta',
        u'\u0398': 'Theta', u'\u0399': 'Iota', u'\u039A': 'Kappa', u'\u039B': 'Lamda', u'\u039C': 'Mu', u'\u039D': 'Nu', u'\u039E': 'Xi', u'\u039F': 'Omicron',
        u'\u03A0': 'Pi', u'\u03A1': 'Rho', u'\u03A3': 'Sigma', u'\u03A4': 'Tau', u'\u03A5': 'Upsilon', u'\u03A6': 'Phi', u'\u03A7': 'Chi', u'\u03A8': 'Psi',
        u'\u03A9': 'Omega', u'\u03B1': 'alpha', u'\u03B2': 'beta', u'\u03B3': 'gamma', u'\u03B4': 'delta', u'\u03B5': 'epsilon', u'\u03B6': 'zeta', u'\u03B7': 'eta',
        u'\u03B8': 'theta', u'\u03B9': 'iota', u'\u03BA': 'kappa', u'\u03BB': 'lamda', u'\u03BC': 'mu', u'\u03BD': 'nu', u'\u03BE': 'xi', u'\u03BF': 'omicron',
        u'\u03C0': 'pi', u'\u03C1': 'rho', u'\u03C3': 'sigma', u'\u03C4': 'tau', u'\u03C5': 'upsilon', u'\u03C6': 'phi', u'\u03C7': 'chi', u'\u03C8': 'psi', u'\u03C9': 'omega',
    }
    common_unicode_list = [u'\u2013', u'\u201d', u'\u201c']

    harvard_citation = re.compile(r"\s\(([\w\&\.\s]+,?\s\d{4}(;\s+[\w\&\.\s]+,?\s\d{4})*)\)", re.IGNORECASE)
    plain_citation = re.compile(r"(\s)\[(\d{1,3}(,|;))*\d{1,3}\]", re.IGNORECASE)
    paper_mention = re.compile(r"\w+(\set\sal\.|\sand\s\w+|\s\&\s\w+)\s(\(|\[)\d{4}(\)|\])", re.IGNORECASE)
    url_link = re.compile(r"(https?|ftp)://[^\s/$.?#].[^\s),]*", re.IGNORECASE)
    percentage = re.compile(r"\d{2,4}([,|.]\d{1,3})?%", re.IGNORECASE)
    integer_number = re.compile(r"\s(\+|-)?([2-9][0-9]|(\d{1,3},)*\d{3}|\d{3,7})\s", re.IGNORECASE)
    float_number = re.compile(r"\s(\+|-)?\d+\.\d+\s", re.IGNORECASE)
    in_paper_ref = re.compile(r"\((see(\w|\s){2,20}|table\s(\w|\d|\s){1,5}|tab.\s(\w|\d|\s){1,5}|figure\s(\w|\d|\s){1,5}|fig.\s(\w|\d|\s){1,5})\)", re.IGNORECASE)
    gen_sequence = re.compile(r"(5('|´|′)|3('|´|′))-?([gtcaun]|\[[gtcaun]\/[gtcaun]\]){5,}-?(5('|´|′)|3('|´|′))?", re.IGNORECASE)
    
    def intrepl(matchobj):
        minus = matchobj.group(0).count(',')
        return ' intertoken' + str(len(matchobj.group(0)) - 2 - minus) + ' '

    for char in chars:
        text = text.replace(char,chars[char])
    for letter in greek_alphabet:
        text = text.replace(letter,greek_alphabet[letter])

    text = re.sub(harvard_citation, ' citetoken', text)
    text = re.sub(plain_citation, ' citetoken', text)
    text = re.sub(paper_mention, 'papertoken', text)   
    text = re.sub(url_link, 'urltoken', text)    
    text = re.sub(percentage, 'percentagetoken', text)
    text = re.sub(float_number, ' floattoken ', text)
    text = re.sub(integer_number, intrepl, text)
    text = re.sub(in_paper_ref, 'referencetoken', text)
    text = re.sub(gen_sequence, "genseqtoken", text)
    return text

def preprocess_simple(text):
    """
    Replace certain non-ACII characters likely to appear,
    but relevant for syntax.
    """
    chars = {'ö':'oe','ä':'ae','ü':'ue', 'Ö':'Oe','Ä':'Ae','Ü':'Ue', "\[":"[", "\]":"]", '´': "'", '′': "'"}
    greek_alphabet = {u'\u0391': 'Alpha', u'\u0392': 'Beta', u'\u0393': 'Gamma', u'\u0394': 'Delta', u'\u0395': 'Epsilon', u'\u0396': 'Zeta', u'\u0397': 'Eta',
        u'\u0398': 'Theta', u'\u0399': 'Iota', u'\u039A': 'Kappa', u'\u039B': 'Lamda', u'\u039C': 'Mu', u'\u039D': 'Nu', u'\u039E': 'Xi', u'\u039F': 'Omicron',
        u'\u03A0': 'Pi', u'\u03A1': 'Rho', u'\u03A3': 'Sigma', u'\u03A4': 'Tau', u'\u03A5': 'Upsilon', u'\u03A6': 'Phi', u'\u03A7': 'Chi', u'\u03A8': 'Psi',
        u'\u03A9': 'Omega', u'\u03B1': 'alpha', u'\u03B2': 'beta', u'\u03B3': 'gamma', u'\u03B4': 'delta', u'\u03B5': 'epsilon', u'\u03B6': 'zeta', u'\u03B7': 'eta',
        u'\u03B8': 'theta', u'\u03B9': 'iota', u'\u03BA': 'kappa', u'\u03BB': 'lamda', u'\u03BC': 'mu', u'\u03BD': 'nu', u'\u03BE': 'xi', u'\u03BF': 'omicron',
        u'\u03C0': 'pi', u'\u03C1': 'rho', u'\u03C3': 'sigma', u'\u03C4': 'tau', u'\u03C5': 'upsilon', u'\u03C6': 'phi', u'\u03C7': 'chi', u'\u03C8': 'psi', u'\u03C9': 'omega',
    }

    for char in chars:
        text = text.replace(char,chars[char])
    for letter in greek_alphabet:
        text = text.replace(letter,greek_alphabet[letter])

    return text