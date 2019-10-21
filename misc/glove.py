import numpy as np
import pickle
import os
import io


def create_glove(path, vocab, opt):
    print("Extracting word vectors form glove...")
    word_vectors = dict()

    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, values = line.split(' ', 1)
            values = values.split()
            try:
                if word in vocab:
                    if word in word_vectors:
                        # Let's use the first occurrence only.
                        continue
                    word_vector = np.array([float(v) for v in values])
                    word_vectors[word] = word_vector
            except ValueError:
                # 840D GloVe file has some encoding errors...
                # I think they can be ignored.
                continue
    norm = 1.0 / np.sqrt(len(vocab))
    glove_weight = np.random.uniform(-norm, norm, (len(vocab), opt.word_size))
    # glove_weight[:] = word_vectors[vocab.unk_word]
    for word in word_vectors:
        word_index = vocab[word]
        glove_weight[word_index, :300] = word_vectors[word]

    print("Find %d word vectors in all %d vocab"%(len(word_vectors), len(vocab)))
    return glove_weight


def load_glove(glove, vocab, opt):
    # assert glove in ["glove.840B.300d", ]
    opt.dataset_split_by = opt.dataset + '_' + opt.split_by
    txt_path = os.path.join(opt.feats_path, 'glove.840B.300d.txt')
    pkl_path = os.path.join(opt.feats_path, opt.dataset_split_by, glove+'.pkl')

    if os.path.isfile(pkl_path):
        glove_weight = pickle.load(open(pkl_path, 'rb'))
    elif os.path.isfile(txt_path):
        glove_weight = create_glove(txt_path, vocab, opt)
        pickle.dump(glove_weight, open(pkl_path, 'wb'))
    else:
        raise IOError("There is no pretrained glove word vectors.")

    return glove_weight
