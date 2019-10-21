from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import os
import json
import argparse
import os.path as osp
import h5py
import numpy as np
from tqdm import tqdm

import sys
import verb
import spacy
import torch

from refer import REFER


def doc_prune(doc):
    new = []
    old = []

    for token in doc:
        if token.pos_ in ['DET', 'SYM', 'SPACE']:
            continue
        else:
            old.append(token.lower_)
            new.append(token.lower_)

    new_sent = " ".join(new)
    new_sent = re.sub(' +',' ', new_sent)

    old_sent = " ".join(old)
    old_sent = re.sub(' +',' ', old_sent)

    return new_sent.rstrip(), new, old_sent.rstrip(), old


def doc_to_tree(doc):
    def traversal(node, sent_list=[]):
        txt = {}
        txt[(node.i, node.text, node.pos_, node.tag_, node.dep_)] = []

        sent_list.append((node.i, node.text, node.pos_, node.tag_, node.dep_))
        for child in node.children:
            tree, sent_list = traversal(child, sent_list)
            txt[(node.i, node.text, node.pos_, node.tag_, node.dep_)].append(tree)
        return txt, sent_list

    # traversal root's every children
    tree = {}
    try:
        root = next(doc.sents).root
    except:
        tree[(0, "UNK", "UNK", "UNK", "UNK")] = []
        print("a NULL sentence: {}".format(doc))
        return tree, [(0, "UNK", "UNK", "UNK", "UNK")]  # cause there a blank sentence in data

    # start traversal
    tree[(root.i, root.text, root.pos_, root.tag_, root.dep_)] = []
    sent_list = []
    sent_list.append((root.i, root.text, root.pos_, root.tag_, root.dep_))
    for child in root.children:
        t, sent_list = traversal(child, sent_list)
        tree[(root.i, root.text, root.pos_, root.tag_, root.dep_)].append(t)

    return tree, sent_list


def transfer_dataset(refer):
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    sentToTokens = refer.sentToTokens  # refer.Sents: 'raw', 'sent', 'tokens'
    if refer.data['dataset'] == 'refcocog':
        error_sents = {268: "a yellow surfboard with a black nose", 29232: "skater in white t - shirt"}
    elif refer.data['dataset'] == 'refcoco':
        error_sents = {115787: "all food on table"}
    elif refer.data['dataset'] == 'refcoco+':
        error_sents = {}

    # extract raw data, for multi-processing
    sents = []
    sents_ids = []
    for sent_id, tokens in sentToTokens.items():
        sents_ids.append(sent_id)

        if sent_id in error_sents:
            sent = error_sents[sent_id]
        else:
            sent = " ".join(tokens)
        sents.append(sent)

    # processing data: prune punct and convert all verb to v-ing
    new_sents = []
    pbar = tqdm(total=len(sents))
    for idx, doc in enumerate(nlp.pipe(sents, batch_size=256, n_threads=16, disable=['parser'])):
        new_sent, new, old_sent, old = doc_prune(doc)
        new_sents.append(new_sent)

        if len(new) == 0:
            print("we must examine the dataset!")
            print(sents[idx])
            raise KeyError

        pbar.update(1)
    pbar.close()

    # processing data: split each sent to multi-branches
    tree_list = []
    token_list = []
    tag_list = []
    dep_list = []
    pbar = tqdm(total=len(new_sents))
    for idx, doc in enumerate(nlp.pipe(new_sents, batch_size=256, n_threads=16)):
        t, s = doc_to_tree(doc)

        tree_list.append(t)

        s = sorted(s, key = lambda x: x[0])
        token = [x[1] for x in s]
        tag = [x[3] for x in s]
        dep = [x[4] for x in s]
        token_list.append(token)
        tag_list.append(tag)
        dep_list.append(dep)

        pbar.update(1)
    pbar.close()

    trees = {}
    for i, idx in enumerate(sents_ids):
        trees[idx] = {'tree': tree_list[i], 'tokens': token_list[i],
                      'tags': tag_list[i], 'deps': dep_list[i]}

    return trees


def build_vocab(trees, vocab_type, count_thr):
    # count up the number of words
    word2count = {}
    for idx, t in trees.items():
        for wd in t[vocab_type]:
            word2count[wd] = word2count.get(wd, 0) + 1

    # print some stats
    total_words = sum(word2count.values())
    bad_words = [wd for wd, n in word2count.items() if n <= count_thr]
    good_words = [wd for wd, n in word2count.items() if n > count_thr]
    bad_count = sum([word2count[wd] for wd in bad_words])
    print('number of good {}: {}'.format(vocab_type, len(good_words)))
    print('number of bad %s: %d/%d = %.2f%%' % ( vocab_type,
        len(bad_words), len(word2count), len(bad_words) * 100.0 / len(word2count)))
    print('number of UNKs in sentences: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))
    vocab = good_words

    # add UNK, PAD
    if bad_count > 0 and 'UNK' not in vocab:
        vocab.append('UNK')
    vocab.insert(0, 'PAD')  # add PAD to the very front

    return vocab


def prepare_data(refer, trees):
    # prepare refs = [{ref_id, ann_id, image_id, box, split, category_id, sent_ids}]
    refs = []
    for ref_id, ref in refer.Refs.items():
        box = refer.refToAnn[ref_id]['bbox']
        refs += [{'ref_id': ref_id, 'split': ref['split'], 'category_id': ref['category_id'], 'ann_id': ref['ann_id'],
                  'sent_ids': ref['sent_ids'], 'box': box, 'image_id': ref['image_id']}]
    print('There in all %s refs.' % len(refs))

    # prepare images = [{'image_id', 'width', 'height', 'file_name', 'ref_ids', 'ann_ids', 'h5_id'}]
    images = []
    h5_id = 0
    for image_id, image in refer.Imgs.items():
        width = image['width']
        height = image['height']
        file_name = image['file_name']
        ref_ids = [ref['ref_id'] for ref in refer.imgToRefs[image_id]]
        ann_ids = [ann['id'] for ann in refer.imgToAnns[image_id]]
        images += [{'image_id': image_id, 'height': height, 'width': width, 'file_name': file_name, 'ref_ids': ref_ids,
                    'ann_ids': ann_ids, 'h5_id': h5_id}]
        h5_id += 1
    print('There are in all %d images.' % h5_id)

    # prepare anns appeared in images, anns = [{ann_id, category_id, image_id, box, h5_id}]
    anns = []
    h5_id = 0
    for image_id in refer.Imgs:
        ann_ids = [ann['id'] for ann in refer.imgToAnns[image_id]]
        for ann_id in ann_ids:
            ann = refer.Anns[ann_id]
            anns += [{'ann_id': ann_id, 'category_id': ann['category_id'], 'box': ann['bbox'], 'image_id': image_id,
                      'h5_id': h5_id}]
            h5_id += 1
    print('There are in all %d anns within the %d images.' % (h5_id, len(images)))

    # prepare sentences = [{sent_id, tokens, image_id}]
    sentences = []
    for sent_id, t in trees.items():
        sent = {'sent_id': sent_id, 'tokens': t['tokens']}
        sentences += [sent]

    return refs, images, anns, sentences


def main(opt):
    # dataset_split_by
    data_root, dataset, split_by = opt.data_root, opt.dataset, opt.split_by

    # load refer
    refer = REFER(data_root, dataset, split_by)

    # transfer dataset to multi-branches
    trees = transfer_dataset(refer)

    # create vocab
    word_vocab = build_vocab(trees, 'tokens', opt.word_count_threshold)
    tag_vocab = build_vocab(trees, 'tags', opt.word_count_threshold)
    dep_vocab = build_vocab(trees, 'deps', opt.word_count_threshold)
    wtoi = {w: i for i, w in enumerate(word_vocab)}
    ttoi = {w: i for i, w in enumerate(tag_vocab)}
    dtoi = {w: i for i, w in enumerate(dep_vocab)}

    # prepare refs, images, anns, sentences and write to json
    refs, images, anns, sentences = prepare_data(refer, trees)

    json.dump({'refs': refs,
               'images': images,
               'anns': anns,
               'sentences': sentences,
               'word_to_ix': wtoi,
               'tag_to_ix': ttoi,
               'dep_to_ix': dtoi,
               'cat_to_ix': {cat_name: cat_id for cat_id, cat_name in refer.Cats.items()}, },
              open(osp.join('data/feats', dataset + '_' + split_by, opt.output_json), 'w'))
    print('%s written.' % osp.join('data/feats', opt.output_json))

    torch.save(trees, osp.join('data/feats', dataset + '_' + split_by, opt.output_pth))
    print('%s written.' % osp.join('data/feats', opt.output_pth))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_json', default='data_dep.json')
    parser.add_argument('--output_pth', default='data_dep.pth')
    parser.add_argument('--data_root', default='data/datasets', type=str)
    parser.add_argument('--dataset', default='refcocog', type=str, help='refcoco/refcoco+/refcocog')
    parser.add_argument('--split_by', default='umd', type=str, help='unc/google/umd')
    parser.add_argument('--word_count_threshold', default=1, type=int)

    opt = parser.parse_args()
    main(opt)
