
import locale
import glob
import os.path
import requests
import tarfile
import sys
import codecs
from smart_open import smart_open
import re

dirname = 'aclImdb'
filename = 'aclImdb_v1.tar.gz'
locale.setlocale(locale.LC_ALL, 'C')
all_lines = []

if sys.version > '3':
    control_chars = [chr(0x85)]
else:
    control_chars = [unichr(0x85)]

# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
    return norm_text

if not os.path.isfile('aclImdb/alldata-id.txt'):
    if not os.path.isdir(dirname):
        if not os.path.isfile(filename):
            # Download IMDB archive
            print("Downloading IMDB archive...")
            url = u'http://ai.stanford.edu/~amaas/data/sentiment/' + filename
            r = requests.get(url)
            with smart_open(filename, 'wb') as f:
                f.write(r.content)
        # if error here, try `tar xfz aclImdb_v1.tar.gz` outside notebook, then re-run this cell
        tar = tarfile.open(filename, mode='r')
        tar.extractall()
        tar.close()
    else:
        print("IMDB archive directory already available without download.")

    # Collect & normalize test/train data
    print("Cleaning up dataset...")
    folders = ['train/pos', 'train/neg', 'test/pos', 'test/neg', 'train/unsup']
    for fol in folders:
        temp = u''
        newline = "\n".encode("utf-8")
        output = fol.replace('/', '-') + '.txt'
        # Is there a better pattern to use?
        txt_files = glob.glob(os.path.join(dirname, fol, '*.txt'))
        print(" %s: %i files" % (fol, len(txt_files)))
        with smart_open(os.path.join(dirname, output), "wb") as n:
            for i, txt in enumerate(txt_files):
                with smart_open(txt, "rb") as t:
                    one_text = t.read().decode("utf-8")
                    for c in control_chars:
                        one_text = one_text.replace(c, ' ')
                    one_text = normalize_text(one_text)
                    all_lines.append(one_text)
                    n.write(one_text.encode("utf-8"))
                    n.write(newline)

    # Save to disk for instant re-use on any future runs
    with smart_open(os.path.join(dirname, 'alldata-id.txt'), 'wb') as f:
        for idx, line in enumerate(all_lines):
            num_line = u"_*{0} {1}\n".format(idx, line)
            f.write(num_line.encode("utf-8"))

assert os.path.isfile("aclImdb/alldata-id.txt"), "alldata-id.txt unavailable"
print("Success, alldata-id.txt is available for next steps.")



# part two of tutorial...
import gensim
from gensim.models.doc2vec import TaggedDocument
from collections import namedtuple

# this data object class suffices as a `TaggedDocument` (with `words` and `tags`)
# plus adds other state helpful for our later evaluation/reporting
SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')

alldocs = []
with smart_open('aclImdb/alldata-id.txt', 'rb', encoding='utf-8') as alldata:
    for line_no, line in enumerate(alldata):
        tokens = gensim.utils.to_unicode(line).split()
        words = tokens[1:]
        tags = [line_no] # 'tags = [tokens[0]]' would also work at extra memory cost
        split = ['train', 'test', 'extra', 'extra'][line_no//25000]  # 25k train, 25k test, 25k extra
        sentiment = [1.0, 0.0, 1.0, 0.0, None, None, None, None][line_no//12500] # [12.5K pos, 12.5K neg]*2 then unknown
        alldocs.append(SentimentDocument(words, tags, split, sentiment))

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']

print('%d docs: %d train-sentiment, %d test-sentiment' % (len(alldocs), len(train_docs), len(test_docs)))


#part three of the tutorial..
from random import shuffle
doc_list = alldocs[:]
shuffle(doc_list)


#part four ..

from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

simple_models = [
    # PV-DBOW plain
    Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores),
    # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
    Doc2Vec(dm=1, vector_size=100, window=10, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores, alpha=0.05, comment='alpha=0.05'),
    # PV-DM w/ concatenation - big, slow, experimental mode
    # window=5 (both sides) approximates paper's apparent 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, vector_size=100, window=5, negative=5, hs=0, min_count=2, sample=0,
            epochs=20, workers=cores),
]

for model in simple_models:
    model.build_vocab(alldocs)
    print("%s vocabulary scanned & state initialized" % model)

models_by_name = OrderedDict((str(model), model) for model in simple_models)


#part five ...


from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[1]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[0], simple_models[2]])


#part 6
import numpy as np
import statsmodels.api as sm
from random import sample


def logistic_predictor_from_data(train_targets, train_regressors):
    """Fit a statsmodel logistic predictor on supplied data"""
    logit = sm.Logit(train_targets, train_regressors)
    predictor = logit.fit(disp=0)
    # print(predictor.summary())
    return predictor


def error_rate_for_model(test_model, train_set, test_set,
                         reinfer_train=False, reinfer_test=False,
                         infer_steps=None, infer_alpha=None, infer_subsample=0.2):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets = [doc.sentiment for doc in train_set]
    if reinfer_train:
        train_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in
                            train_set]
    else:
        train_regressors = [test_model.docvecs[doc.tags[0]] for doc in train_set]
    train_regressors = sm.add_constant(train_regressors)
    predictor = logistic_predictor_from_data(train_targets, train_regressors)

    test_data = test_set
    if reinfer_test:
        if infer_subsample < 1.0:
            test_data = sample(test_data, int(infer_subsample * len(test_data)))
        test_regressors = [test_model.infer_vector(doc.words, steps=infer_steps, alpha=infer_alpha) for doc in
                           test_data]
    else:
        test_regressors = [test_model.docvecs[doc.tags[0]] for doc in test_docs]
    test_regressors = sm.add_constant(test_regressors)

    # Predict & evaluate
    test_predictions = predictor.predict(test_regressors)
    corrects = sum(np.rint(test_predictions) == [doc.sentiment for doc in test_data])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return (error_rate, errors, len(test_predictions), predictor)


