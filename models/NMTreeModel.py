from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import time

from . import gumbel

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class Normalize_Scale(nn.Module):
    def __init__(self, dim, init_norm=20):
        super(Normalize_Scale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        # input is variable (n, dim)
        bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled


class SingleScore(nn.Module):
    """Compute a unsoftmax similarity between single region and language"""
    def __init__(self, vis_dim, word_size):
        super(SingleScore, self).__init__()

        self.W = nn.Linear(vis_dim, word_size)
        self.fc = nn.Linear(word_size, 1)

    def forward(self, v, w):
        """
        v : vis:    Tensor float (num_bbox, vis_dim)
        w : embed:  Tensor float (word_size, )
        l : logit:  Tensor float (num_bbox, )
        """
        v = self.W(v)
        w = w.expand_as(v)
        f = nn.functional.normalize(v * w, p=2, dim=1)
        l = self.fc(f).squeeze(1)

        return l


class PairScore(nn.Module):
    """Compute a unsoftmax similarity between pairwise regions and language"""
    def __init__(self, vis_dim, word_size):
        super(PairScore, self).__init__()

        self.W = nn.Linear(vis_dim * 2, word_size)
        self.fc = nn.Linear(word_size, 1)

    def forward(self, v, l, w):
        """
        v   : vis:    Tensor float (num_bbox, vis_dim)
        l   : logit:  Tensor float (num_bbox, )
        w   : embed:  Tensor float (word_size, )
        l_2 : logit:  Tensor float (num_bbox, )
        """
        s = F.softmax(l, 0)
        v_2 = torch.mm(s.unsqueeze(0), v)

        v_3 = torch.cat((v, v_2.repeat(v.size(0), 1)), dim=1)
        v_3 = self.W(v_3)
        w = w.expand_as(v_3)
        f = nn.functional.normalize(v_3 * w, p=2, dim=1)
        l_2 = self.fc(f).squeeze(1)

        return l_2


class NMTreeModel(nn.Module):

    def __init__(self, opt, loader):
        super(NMTreeModel, self).__init__()

        # vocab
        self.rnn_size = opt.rnn_size
        self.vis_dim = opt.vis_dim
        self.embed_size = opt.word_size + opt.tag_size + opt.dep_size

        # embedding settings
        self.word_embedding = nn.Embedding(num_embeddings=opt.word_vocab_size,
                                           embedding_dim=opt.word_size)
        self.tag_embedding = nn.Embedding(num_embeddings=opt.tag_vocab_size,
                                           embedding_dim=opt.tag_size)
        self.dep_embedding = nn.Embedding(num_embeddings=opt.dep_vocab_size,
                                           embedding_dim=opt.dep_size)

        self.word_vocab = loader.word_to_ix
        self.tag_vocab = loader.tag_to_ix
        self.dep_vocab = loader.dep_to_ix

        self.dropout = nn.Dropout(opt.drop_prob)

        self.single_score = SingleScore(self.vis_dim, self.embed_size)
        self.pair_score = PairScore(self.vis_dim ,self.embed_size)

        self.up_tree_lstm = UpTreeLSTM(self.embed_size, opt.rnn_size)
        self.down_tree_lstm = DownTreeLSTM(self.embed_size, opt.rnn_size)

        self.sbj_attn_logit = nn.Linear(opt.rnn_size*2, 1)
        self.rlt_attn_logit = nn.Linear(opt.rnn_size*2, 1)
        self.obj_attn_logit = nn.Linear(opt.rnn_size*2, 1)

    def traversal(self, node, v, embedding):
        node.logit = self.node_to_logit(node)
        for idx in range(node.num_children):
            self.traversal(node.children[idx], v, embedding)

        # Case 1: Leaf, update node.sub_word and node.score
        if node.num_children == 0:
            node.sub_word = [node.idx]
            node.sub_logit = [node.logit]
            sbj_embedding = self.attn_embedding(node.sub_word, node.sub_logit, embedding, 'sbj')
            sbj_score = self.single_score(v, sbj_embedding)
            node.score = sbj_score
        # Case 2: Not leaf
        else:
            sub_word = [node.idx]
            sub_logit = [node.logit]
            for c in node.children:
                sub_word = sub_word  + c.sub_word
                sub_logit = sub_logit + c.sub_logit

            # Case 2.1: Not leaf, Language Node
            sbj_score = self.zero_score
            for c in node.children:
                sbj_score = sbj_score + c.score

            # Case 2.2: Not leaf, Visual Node
            obj_embedding = self.attn_embedding(sub_word, sub_logit, embedding, 'obj')
            obj_score = self.single_score(v, obj_embedding)
            for c in node.children:
                obj_score = obj_score + c.score
            node.obj_score = obj_score
            rlt_embedding = self.attn_embedding(sub_word, sub_logit, embedding, 'rlt')
            rlt_score = self.pair_score(v, obj_score, rlt_embedding)

            # Gumbel softmax, [lang, vis]
            node.sub_word = sub_word
            node.sub_logit = sub_logit
            node.score = torch.mm(node.type, torch.stack([sbj_score, rlt_score], dim=0)).squeeze(0)

        return

    def list_to_embedding(self, sent):
        """
        translate a sentence into embedding
        Input: list of sentence, contains tokens, tags, deps
        Output: embedding (num_words * embedding_size)
        """
        word_ids = []
        for word in sent['tokens']:
            word_id = self.word_vocab[word] if word in self.word_vocab else self.word_vocab['UNK']
            word_ids.append(word_id)

        tag_ids = []
        for tag in sent['tags']:
            tag_id = self.tag_vocab[tag] if tag in self.tag_vocab else self.tag_vocab['UNK']
            tag_ids.append(tag_id)

        dep_ids = []
        for dep in sent['deps']:
            dep_id = self.dep_vocab[dep] if dep in self.dep_vocab else self.dep_vocab['UNK']
            dep_ids.append(dep_id)

        word_ids = torch.tensor(word_ids).cuda()
        w_embed = self.word_embedding(word_ids)
        tag_ids = torch.tensor(tag_ids).cuda()
        t_embed = self.tag_embedding(tag_ids)
        dep_ids = torch.tensor(dep_ids).cuda()
        d_embed = self.dep_embedding(dep_ids)

        embedding = torch.cat([w_embed, t_embed, d_embed], dim=-1)
        return self.dropout(embedding)

    def attn_embedding(self, word_list, logit_list, embedding, logit_type):
        embed = embedding[word_list]
        logits = torch.stack([s[logit_type] for s in logit_list], dim=0)

        attn = torch.softmax(logits, dim=0)
        attn_embed = torch.mm(attn.unsqueeze(0), embed)

        return attn_embed

    def node_to_logit(self, node):
        hidden = torch.cat([node.up_state[1], node.down_state[1]], dim=-1)
        sbj_logit = self.sbj_attn_logit(hidden).squeeze()
        rlt_logit = self.rlt_attn_logit(hidden).squeeze()
        obj_logit = self.obj_attn_logit(hidden).squeeze()

        logit = {'sbj': sbj_logit, 'rlt': rlt_logit, 'obj': obj_logit}

        return logit

    def forward(self, data, is_show=False):
        """
        Input:
            vis:      float32 (num_box, vis_dim)
            tree:     dict and list to represent structure
            sents:    list of words
        scores:   float32 (num_sent, num_box)
        """
        vis = data['vis']
        self.num_bbox = vis.size(1)
        self.zero_score = torch.zeros((self.num_bbox, ), dtype=torch.float32, requires_grad=False).cuda()

        scores = []
        for i in range(len(data['trees'])):
            tree_dict = data['trees'][i]
            sent_list = data['sents'][i]
            v = vis[i]

            # list of word embedding
            word_embed = self.list_to_embedding(sent_list)

            # build tree structure to class
            tree = build_bitree(tree_dict)

            self.up_tree_lstm(tree, word_embed)
            self.down_tree_lstm(tree, word_embed)
            self.traversal(tree, v, word_embed)

            # for root
            if tree.type[0, 0]:
                sbj_embedding = self.attn_embedding(tree.sub_word, tree.sub_logit, word_embed, 'sbj')
                sbj_score = self.single_score(v, sbj_embedding)
            else:
                sbj_score = self.zero_score

            s = sbj_score + tree.score

            scores.append(s)

        if is_show:
            print(tree.__vis__(['type_', 'word']))

        scores = torch.stack(scores, dim=0)
        return F.log_softmax(scores, dim=1)

    def word_cloud(self, data, stat_dict={}):
        def loop(node, stat_dict):
            word = node.word
            if len(node.children) == 0:
                node_type = 'l'
            else:
                node_type = node.type_

            stat_dict[word] = stat_dict.get(word, {'l': 0, 'v': 0})
            stat_dict[word][node_type] += 1

            for c in node.children:
                stat_dict = loop(c, stat_dict)
            return stat_dict

        vis = data['vis']
        self.num_bbox = vis.size(1)
        self.zero_score = torch.zeros((self.num_bbox, ), dtype=torch.float32, requires_grad=False).cuda()

        for i in range(len(data['trees'])):
            tree_dict = data['trees'][i]
            sent_list = data['sents'][i]
            v = vis[i]

            # linear leaf lstm to initialize hidden and cell
            word_embed = self.list_to_embedding(sent_list)

            # build tree structure to class
            tree = build_bitree(tree_dict)

            self.up_tree_lstm(tree, word_embed)
            self.down_tree_lstm(tree, word_embed)
            self.traversal(tree, v, word_embed)

            stat_dict = loop(tree, stat_dict)
        return stat_dict

    def tree_map(self, data):
        def loop(node, score_dict={}):
            word = node.word
            if node.type_ == 'v':
                score = node.score.data.cpu().numpy()
                obj_score = node.obj_score.data.cpu().numpy()
                score_dict[word] = [score, obj_score]
            else:
                score = node.score.data.cpu().numpy()
                score_dict[word] = [score]

            for c in node.children:
                score_dict = loop(c, score_dict)
            return score_dict

        vis = data['vis']
        self.num_bbox = vis.size(1)
        self.zero_score = torch.zeros((self.num_bbox, ), dtype=torch.float32, requires_grad=False).cuda()

        scores = []
        score_dict_list = []
        tree_list = []
        for i in range(len(data['trees'])):
            tree_dict = data['trees'][i]
            sent_list = data['sents'][i]
            v = vis[i]

            # linear leaf lstm to initialize hidden and cell
            word_embed = self.list_to_embedding(sent_list)

            # build tree structure to class
            tree = build_bitree(tree_dict)

            self.up_tree_lstm(tree, word_embed)
            self.down_tree_lstm(tree, word_embed)
            self.traversal(tree, v, word_embed)

            # for root
            if tree.type[0, 0]:
                sbj_embedding = self.attn_embedding(tree.sub_word, tree.sub_logit, word_embed, 'sbj')
                sbj_score = self.single_score(v, sbj_embedding)
            else:
                sbj_score = self.zero_score
                print("a visual root!!!")
            s = sbj_score + tree.score
            scores.append(s)

            score_dict = loop(tree, {})
            score_dict['ROOT'] = [s.data.cpu().numpy()]
            score_dict_list.append(score_dict)
            tree_list.append(tree.__vis__(['type_', 'word']))

        scores = torch.stack(scores, dim=0)
        return F.log_softmax(scores, dim=1), score_dict_list, tree_list


def build_bitree(tree):
    def traversal(node):
        (wd, child), = node.items()
        node = BiTree(wd)
        for c in child:
            node.add_child(traversal(c))
        return node

    node = traversal(tree)
    return node


class UpTreeLSTM(nn.Module):
    """
    Adapted from:
    https://github.com/dasguptar/treelstm.pytorch/blob/master/treelstm/model.py
    """
    def __init__(self, in_dim, mem_dim):
        super(UpTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        # gates for visual node
        self.ioux_vis = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_vis = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx_vis = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_vis = nn.Linear(self.mem_dim, self.mem_dim)

        # gates for language node
        self.ioux_lang = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_lang = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx_lang = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_lang = nn.Linear(self.mem_dim, self.mem_dim)

        # Gumbel softmax
        self.type_query = nn.Linear(self.in_dim, 2)
        self.dropout = nn.Dropout()

    def node_forward_vis(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux_vis(inputs) + self.iouh_vis(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh_vis(child_h) +
            self.fx_vis(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def node_forward_lang(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux_lang(inputs) + self.iouh_lang(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh_lang(child_h) +
            self.fx_lang(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].data.new(1, self.mem_dim).zero_()
            child_h = inputs[0].data.new(1, self.mem_dim).zero_()
        else:
            child_c, child_h = zip(* map(lambda x: x.up_state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        c_vis, h_vis = self.node_forward_vis(inputs[tree.idx], child_c, child_h)
        c_lang, h_lang = self.node_forward_lang(inputs[tree.idx], child_c, child_h)

        # Gumbel softmax decision
        type_value = inputs[tree.idx].unsqueeze(0)
        type_weights = self.type_query(type_value)

        if self.training:
            type_mask = gumbel.st_gumbel_softmax(logits=type_weights)
        else:
            type_mask = gumbel.greedy_select(logits=type_weights)
            type_mask = type_mask.float()

        h = torch.mm(type_mask, torch.cat([h_lang, h_vis], dim=0))
        c = torch.mm(type_mask, torch.cat([c_lang, c_vis], dim=0))

        tree.type = type_mask
        tree.type_ = 'l' if type_mask[0, 0].item() else 'v'
        if tree.num_children == 0:
            tree.type_ = 'l'

        tree.up_state = (c, self.dropout(h))

        return tree.up_state


class DownTreeLSTM(nn.Module):
    """
    Adapted from:
    https://github.com/dasguptar/treelstm.pytorch/blob/master/treelstm/model.py
    """
    def __init__(self, in_dim, mem_dim):
        super(DownTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        # gates for visual node
        self.ioux_vis = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_vis = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx_vis = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_vis = nn.Linear(self.mem_dim, self.mem_dim)

        # gates for language node
        self.ioux_lang = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh_lang = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx_lang = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_lang = nn.Linear(self.mem_dim, self.mem_dim)

        # Gumbel softmax
        self.dropout = nn.Dropout()

    def node_forward_vis(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux_vis(inputs) + self.iouh_vis(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh_vis(child_h) +
            self.fx_vis(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def node_forward_lang(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux_lang(inputs) + self.iouh_lang(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh_lang(child_h) +
            self.fx_lang(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        if tree.parent == None:
            child_c = inputs[0].data.new(1, self.mem_dim).zero_()
            child_h = inputs[0].data.new(1, self.mem_dim).zero_()
        else:
            child_c, child_h = tree.parent.down_state

        c_vis, h_vis = self.node_forward_vis(inputs[tree.idx], child_c, child_h)
        c_lang, h_lang = self.node_forward_lang(inputs[tree.idx], child_c, child_h)

        # Gumbel softmax decision
        type_mask = tree.type
        h = torch.mm(type_mask, torch.cat([h_lang, h_vis], dim=0))
        c = torch.mm(type_mask, torch.cat([c_lang, c_vis], dim=0))

        tree.down_state = (c, self.dropout(h))

        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        return tree.down_state


class BiTree(object):
    def __init__(self, wd):
        # structure
        idx, word, pos, tag, dep = wd
        self.pos = pos
        self.tag = tag
        self.dep = dep
        self.word = word
        self.idx = idx

        self.parent = None
        self.num_children = 0
        self.is_leaf = True
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.is_leaf = False
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __str__(self):
        return self.get_ascii()

    def __vis__(self, attributes=None):
        return self.get_ascii(attributes=attributes)

    def __repr__(self):
        return self.word

    def _asciiArt(self, char1='-', show_internal=True, compact=False, attributes=None):
        """
        Returns the ASCII representation of the tree.
        Code based on the PyCogent GPL project.
        """
        if not attributes:
            attributes = ["word"]
        node_name = ', '.join(map(str, [getattr(self, v) for v in attributes if hasattr(self, v)]))

        LEN = max(3, len(node_name) if not self.children or show_internal else 3)
        PAD = ' ' * LEN
        PA = ' ' * (LEN-1)
        if not self.is_leaf:
            mids = []
            result = []
            for c in self.children:
                if len(self.children) == 1:
                    char2 = '-'
                elif c is self.children[0]:
                    char2 = '/'
                elif c is self.children[-1]:
                    char2 = '\\'
                else:
                    char2 = '-'
                (clines, mid) = c._asciiArt(char2, show_internal, compact, attributes)
                mids.append(mid+len(result))
                result.extend(clines)
                if not compact:
                    result.append('')
            if not compact:
                result.pop()
            (lo, hi, end) = (mids[0], mids[-1], len(result))
            prefixes = [PAD] * (lo+1) + [PA+'|'] * (hi-lo-1) + [PAD] * (end-hi)
            mid = int((lo + hi) / 2)
            prefixes[mid] = char1 + '-'*(LEN-1) + prefixes[mid][-1]
            result = [p+l for (p,l) in zip(prefixes, result)]
            if show_internal:
                stem = result[mid]
                result[mid] = stem[0] + node_name + stem[len(node_name)+1:]
            return (result, mid)
        else:
            return ([char1 + '-' + node_name], 0)

    def get_ascii(self, show_internal=True, compact=False, attributes=None):
        """
        Returns a string containing an ascii drawing of the tree.
        :argument show_internal: includes internal edge names.
        :argument compact: use exactly one line per tip.
        :param attributes: A list of node attributes to shown in the
            ASCII representation.
        """
        (lines, mid) = self._asciiArt(show_internal=show_internal,
                                      compact=compact, attributes=attributes)
        return '\n'+'\n'.join(lines)
