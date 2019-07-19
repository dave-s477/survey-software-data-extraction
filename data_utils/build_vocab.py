from collections import Counter
from os.path import join

def build_vocab(output_dir, file_list=['BIO_format.txt'], base_name='data'):
    """
    Build the vocabulary of the BIO formatted dataset. 
    
    Arguments:
        output_dir {[str]} -- [where to write, also has to be the location of the BIO file]
    
    Keyword Arguments:
        file_list {list} -- [one or more BIO files] (default: {['BIO_format.txt']})
        base_name {str} -- [Base string for the created files] (default: {'data'})
    """
    counter_words = Counter()
    counter_tags = Counter()
    for n in file_list:
        with open(join(output_dir, n), 'r') as f:
            for line in f:
                if not line.isspace():
                    counter_words[line.strip().split()[0]] += 1
                    counter_tags[line.strip().split()[1]] += 1

    vocab_words = {w for w, c in counter_words.items()}

    word_file = base_name + '.words.txt'
    with open(join(output_dir, word_file), 'w') as f:
        for w in sorted(list(vocab_words)):
            f.write(w + '\n')

    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    char_file = base_name + '.chars.txt'
    with open(join(output_dir, char_file), 'w') as f:
        for c in sorted(list(vocab_chars)):
            f.write(c + '\n')

    tag_file = base_name + '.tag.txt'
    with open(join(output_dir, tag_file), 'w') as f:
        for t in sorted(list(counter_tags)):
            f.write(t + '\n')
    print("Done creating words, characters and tags. Found the following tags: ")
    print(counter_tags)