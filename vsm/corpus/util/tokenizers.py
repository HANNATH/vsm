import nltk

from base import *



__all__=['toy_corpus', 'file_corpus', 'dir_corpus', 'coll_corpus',
         'word_tokenize', 'sentence_tokenize', 'paragraph_tokenize']




def word_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    lower-case words in this text with numbers and punctuation, except
    for hyphens, removed.

    The core work is done by NLTK's Treebank Word Tokenizer.
    """

    text = rehyph(text)

    text = nltk.TreebankWordTokenizer().tokenize(text)

    tokens = [word.lower() for word in text]

    tokens = strip_punc(tokens)

    tokens = rem_num(tokens)
    
    return tokens



def sentence_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    sentences in this text.

    This is a wrapper for NLTK's pre-trained Punkt Tokenizer.
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    return tokenizer.tokenize(text)



def paragraph_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    paragraphs in this text. It's expected that the text marks
    paragraphs with two consecutive line breaks.
    """

    par_break = re.compile(r'[\r\n]{2,}')
    
    return par_break.split(text)



#######################################################
#
#  Special purpose tokenizers & Corpus generators
#
#######################################################



def toy_corpus(plain_corpus, is_filename=False, compress=True,
               nltk_stop=False, mask_freq=0, add_stop=None, metadata=None):
    """
    `toy_corpus` is a convenience function for generating Corpus or
    MaskedCorpus objects from a given string or a single file.

    `toy_corpus` will perform both word and document-level
    tokenization. It will also strip punctuation and arabic numerals
    outside the range 1-29. All letters are made lowercase.

    Document tokens are delimited by `\n\n`. E.g.,

        <document 0>

        <document 1>

        ...

        <document n>

    where <document i> is any chunk of text to be tokenized by word.

    Parameters
    ----------
    plain_corpus : string-like
        String containing a plain-text corpus or a filename of a file
        containing one.
    is_filename : boolean
        If `True` then `plain_corpus` is treated like a filename.
        Otherwise, `plain_corpus` is presumed to contain the corpus.
        Default is `False`.
    compress : boolean
        If `True` then a Corpus object is returned with all masked
        terms removed. Otherwise, a MaskedCorpus object is returned.
        Default is `True`.
    nltk_stop : boolean
        If `True` then the corpus object is masked using the NLTK
        English stop words. Default is `False`.
    mask_freq : int
        The upper bound for a term to be masked on the basis of its
        collection frequency. Default is 0.
    add_stop : array-like
        A list of stop words. Default is `None`.
    metadata : array-like
        A list of strings providing metadata about the documents. If
        provided, must have length equal to the number of documents.
        Default is `None`.
    Returns
    -------
    c : a Corpus or a MaskedCorpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.

    """
    from vsm.corpus import MaskedCorpus, mask_from_stoplist, mask_freq_t

    if is_filename:

        with open(plain_corpus, 'r') as f:

            plain_corpus = f.read()

    docs = plain_corpus.split('\n\n')

    docs = [word_tokenize(d) for d in docs]

    corpus = sum(docs, [])

    tok = np.cumsum(np.array([len(d) for d in docs]))

    if metadata:

        if not len(metadata) == len(tok):

            msg = 'Metadata mismatch: metadata length is {0} and number'\
                   'of documents is {1}'.format(len(metadata), len(tok))

            raise Exception(msg)

        else:

            dtype = [('idx', np.array(tok).dtype),
                     ('document_label', np.array(metadata).dtype)]

            tok = np.array(zip(tok, metadata), dtype=dtype)

    else:

        dtype = [('idx', np.array(tok).dtype)]

        tok = np.array([(i,) for i in tok], dtype=dtype)
    
    c = MaskedCorpus(corpus, tok_data=[tok], tok_names=['document'])

    c = mask_corpus(c, nltk_stop=nltk_stop,
                    mask_freq=mask_freq, add_stop=add_stop)

    if compress:

        c = c.to_corpus(compress=True)

    return c



def file_tokenize(text):

    words, par_tokens, sent_tokens = [], [], []

    sent_break, par_n, sent_n = 0, 0, 0

    pars = paragraph_tokenize(text)

    for par in pars:
                
        sents = sentence_tokenize(par)

        for sent in sents:
                    
            w = word_tokenize(sent)
                    
            words.extend(w)
                    
            sent_break += len(w)
                    
            sent_tokens.append((sent_break, par_n, sent_n))

            sent_n += 1

        par_tokens.append((sent_break, par_n))
            
        par_n += 1

    idx_dt = ('idx', np.int32)

    sent_label_dt = ('sent_label', np.array(sent_n, np.str_).dtype)

    par_label_dt = ('par_label', np.array(par_n, np.str_).dtype)

    corpus_data = dict()

    dtype = [idx_dt, par_label_dt]

    corpus_data['paragraph'] = np.array(par_tokens, dtype=dtype)

    dtype = [idx_dt, par_label_dt, sent_label_dt]

    corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    return words, corpus_data




def file_corpus(filename, compress=True,
                nltk_stop=True, mask_freq=1, add_stop=None):
    """
    For use with a plain text corpus contained in a single string.
    """

    from vsm.corpus import MaskedCorpus

    with open(filename, mode='r') as f:

        text = f.read()

    words, tok = file_tokenize(text)

    names, data = zip(*tok.items())
    
    c = MaskedCorpus(words, tok_data=data, tok_names=names)

    c = mask_corpus(c, nltk_stop=nltk_stop,
                    mask_freq=mask_freq, add_stop=add_stop)

    if compress:

        c = c.to_corpus(compress=True)

    return c



def dir_tokenize(chunks, labels, chunk_name='article', paragraphs=True):

    words, chk_tokens, sent_tokens = [], [], []

    sent_break, chk_n, sent_n = 0, 0, 0

    if paragraphs:

        par_tokens = []

        par_n = 0
        
        for chk, label in zip(chunks, labels):

            print 'Tokenizing', label

            pars = paragraph_tokenize(chk)

            for par in pars:
                
                sents = sentence_tokenize(par)

                for sent in sents:
                    
                    w = word_tokenize(sent)
                    
                    words.extend(w)
                    
                    sent_break += len(w)
                    
                    sent_tokens.append((sent_break, label, par_n, sent_n))

                    sent_n += 1

                par_tokens.append((sent_break, label, par_n))

                par_n += 1

            chk_tokens.append((sent_break, label))

            chk_n += 1

    else:

        for chk, label in zip(chunks, labels):

            print 'Tokenizing', label

            sents = sentence_tokenize(chk)

            for sent in sents:
                    
                w = word_tokenize(sent)
                    
                words.extend(w)
                    
                sent_break += len(w)

                sent_tokens.append((sent_break, label, sent_n))

                sent_n += 1

            chk_tokens.append((sent_break, label))

            chk_n += 1

    idx_dt = ('idx', np.int32)

    label_dt = (chunk_name + '_label', np.array(labels).dtype)

    sent_label_dt = ('sent_label', np.array(sent_n, np.str_).dtype)

    corpus_data = dict()

    dtype = [idx_dt, label_dt]

    corpus_data[chunk_name] = np.array(chk_tokens, dtype=dtype)

    if paragraphs:

        par_label_dt = ('par_label', np.array(par_n, np.str_).dtype)

        dtype = [idx_dt, label_dt, par_label_dt]

        corpus_data['paragraph'] = np.array(par_tokens, dtype=dtype)

        dtype = [idx_dt, label_dt, par_label_dt, sent_label_dt]

        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    else:

        dtype = [idx_dt, label_dt, sent_label_dt]

        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    return words, corpus_data



def dir_corpus(plain_dir, chunk_name='article', paragraphs=True,
               compress=True, nltk_stop=True, mask_freq=1, add_stop=None):
    """
    `dir_corpus` is a convenience function for generating Corpus or
    MaskedCorpus objects from a directory of plain text files.

    `dir_corpus` will retain file-level tokenization and perform
    sentence and word tokenizations. Optionally, it will provide
    paragraph-level tokenizations.

    It will also strip punctuation and arabic numerals outside the
    range 1-29. All letters are made lowercase.

    Parameters
    ----------
    plain_dir : string-like
        String containing directory containing a plain-text corpus.
    chunk_name : string-line
        The name of the tokenization corresponding to individual
        files. For example, if the files are pages of a book, one
        might set `chunk_name` to `pages`. Default is `articles`.
    paragraphs : boolean
        If `True`, a paragraph-level tokenization is included.
        Defaults to `True`.
    compress : boolean
        If `True` then a Corpus object is returned with all masked
        terms removed. Otherwise, a MaskedCorpus object is returned.
        Default is `True`.
    nltk_stop : boolean
        If `True` then the corpus object is masked using the NLTK
        English stop words. Default is `False`.
    mask_freq : int
        The upper bound for a term to be masked on the basis of its
        collection frequency. Default is 0.
    add_stop : array-like
        A list of stop words. Default is `None`.

    Returns
    -------
    c : a Corpus or a MaskedCorpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.

    """

    from vsm.corpus import MaskedCorpus

    chunks = []
    
    filenames = os.listdir(plain_dir)

    filenames.sort()

    for filename in filenames:
        
        filename = os.path.join(plain_dir, filename)

        with open(filename, mode='r') as f:

            chunks.append(f.read())

    words, tok = dir_tokenize(chunks, filenames, chunk_name=chunk_name,
                              paragraphs=paragraphs)

    names, data = zip(*tok.items())
    
    c = MaskedCorpus(words, tok_data=data, tok_names=names)

    c = mask_corpus(c, nltk_stop=nltk_stop,
                    mask_freq=mask_freq, add_stop=add_stop)

    if compress:

        c = c.to_corpus(compress=True)

    return c



def coll_tokenize(books, book_names):

    words, book_tokens, page_tokens, sent_tokens = [], [], [], []

    sent_break, book_n, page_n, sent_n = 0, 0, 0, 0

    for book, book_label in zip(books, book_names):

        print 'Tokenizing', book_label

        for page in book:
        
            sents = sentence_tokenize(page)

            for sent in sents:
        
                w = word_tokenize(sent)
        
                words.extend(w)
                    
                sent_break += len(w)
            
                sent_tokens.append((sent_break, sent_n,
                                    page_n, book_label))
            
                sent_n += 1

            page_tokens.append((sent_break, page_n, book_label))

            page_n += 1
            
        book_tokens.append((sent_break, book_label))

        book_n += 1

    idx_dt = ('idx', np.int32)

    book_label_dt = ('book_label', np.array(book_names).dtype)

    page_label_dt = ('page_label', np.array(page_n, np.str_).dtype)

    sent_label_dt = ('sent_label', np.array(sent_n, np.str_).dtype)

    corpus_data = dict()

    dtype = [idx_dt, book_label_dt]

    corpus_data['book'] = np.array(book_tokens, dtype=dtype)

    dtype = [idx_dt, page_label_dt, book_label_dt]

    corpus_data['page'] = np.array(page_tokens, dtype=dtype)

    dtype = [idx_dt, sent_label_dt, page_label_dt, book_label_dt]

    corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    return words, corpus_data



def coll_corpus(coll_dir, ignore=['.json', '.log'], compress=True,
                nltk_stop=True, mask_freq=1, add_stop=None):

    from vsm.corpus import MaskedCorpus

    books = []
    
    book_names = os.listdir(coll_dir)

    book_names = filter_by_suffix(book_names, ignore)

    book_names.sort()

    for book_name in book_names:

        pages = []

        book_path = os.path.join(coll_dir, book_name)

        page_names = os.listdir(book_path)

        page_names = filter_by_suffix(page_names, ignore)

        page_names.sort()

        for page_name in page_names:
        
            page_name = os.path.join(book_path, page_name)

            with open(page_name, mode='r') as f:

                pages.append(f.read())

        books.append(pages)

    words, tok = coll_tokenize(books, book_names)

    names, data = zip(*tok.items())
    
    c = MaskedCorpus(words, tok_data=data, tok_names=names)

    c = mask_corpus(c, nltk_stop=nltk_stop,
                    mask_freq=mask_freq, add_stop=add_stop)

    if compress:

        c = c.to_corpus(compress=True)

    return c



########################################
#
#                 Tests
#
########################################



def test_dir_tokenize():

    chunks = ['foo foo foo\n\nfoo foo',
             'Foo bar.  Foo bar.', 
             '',
             'foo\n\nfoo']

    labels = [str(i) for i in xrange(len(chunks))]

    words, tok_data = dir_tokenize(chunks, labels)

    assert len(words) == 11

    assert len(tok_data['article']) == 4

    assert len(tok_data['paragraph']) == 6

    assert len(tok_data['sentence']) == 7
    
    assert (tok_data['article']['idx'] == [5, 9, 9, 11]).all()

    assert (tok_data['article']['article_label'] == ['0', '1', '2', '3']).all()

    assert (tok_data['paragraph']['idx'] == [3, 5, 9, 9, 10, 11]).all()

    assert (tok_data['paragraph']['article_label'] == 
            ['0', '0', '1', '2', '3', '3']).all()

    assert (tok_data['paragraph']['par_label'] == 
            ['0', '1', '2', '3', '4', '5']).all()

    assert (tok_data['sentence']['idx'] == [3, 5, 7, 9, 9, 10, 11]).all()

    assert (tok_data['sentence']['article_label'] == 
            ['0', '0', '1', '1', '2', '3', '3']).all()

    assert (tok_data['sentence']['par_label'] == 
            ['0', '1', '2', '2', '3', '4', '5']).all()

    assert (tok_data['sentence']['sent_label'] == 
            ['0', '1', '2', '3', '4', '5', '6']).all()



def test_coll_tokenize():

    books = [['foo foo foo.\n\nfoo foo',
              'Foo bar.  Foo bar.'], 
             ['',
              'foo.\n\nfoo']]

    book_names = [str(i) for i in xrange(len(books))]

    words, tok_data = coll_tokenize(books, book_names)

    assert len(words) == 11

    assert len(tok_data['book']) == 2

    assert len(tok_data['page']) == 4

    assert len(tok_data['sentence']) == 7
    
    assert (tok_data['book']['idx'] == [9, 11]).all()

    assert (tok_data['book']['book_label'] == ['0', '1']).all()

    assert (tok_data['page']['idx'] == [5, 9, 9, 11]).all()

    assert (tok_data['page']['page_label'] == ['0', '1', '2', '3']).all()

    assert (tok_data['page']['book_label'] == ['0', '0', '1', '1']).all()

    assert (tok_data['sentence']['idx'] == [3, 5, 7, 9, 9, 10, 11]).all()

    assert (tok_data['sentence']['sent_label'] == 
            ['0', '1', '2', '3', '4', '5', '6']).all()

    assert (tok_data['sentence']['page_label'] == 
            ['0', '0', '1', '1', '2', '3', '3']).all()

    assert (tok_data['sentence']['book_label'] == 
            ['0', '0', '0', '0', '1', '1', '1']).all()
    


def test_toy_corpus():

    keats = ('She dwells with Beauty - Beauty that must die;\n\n'
             'And Joy, whose hand is ever at his lips\n\n' 
             'Bidding adieu; and aching Pleasure nigh,\n\n'
             'Turning to poison while the bee-mouth sips:\n\n'
             'Ay, in the very temple of Delight\n\n'
             'Veil\'d Melancholy has her sovran shrine,\n\n'
             'Though seen of none save him whose strenuous tongue\n\n'
             'Can burst Joy\'s grape against his palate fine;\n\n'
             'His soul shall taste the sadness of her might,\n\n'
             'And be among her cloudy trophies hung.')

    assert toy_corpus(keats)

    assert toy_corpus(keats, nltk_stop=True)

    assert toy_corpus(keats, mask_freq=1)

    assert toy_corpus(keats, add_stop=['and', 'with'])

    assert toy_corpus(keats, nltk_stop=True,
                      mask_freq=1, add_stop=['ay'])

    from tempfile import NamedTemporaryFile as NFT

    tmp = NFT(delete=False)

    tmp.write(keats)

    tmp.close()

    c = toy_corpus(tmp.name, is_filename=True, 
                   nltk_stop=True, add_stop=['ay'])
    
    assert c

    os.remove(tmp.name)

    return c
