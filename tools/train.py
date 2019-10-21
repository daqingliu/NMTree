from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import random
import json
import logging
import numpy as np

import torch

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, '../'))

import models
import utils.eval_utils as eval_utils
import utils.train_utils as train_utils
from utils.gt_loader import GtLoader
import tools.opts as opts
from misc.glove import load_glove


def initialize_logger(output_file):
    formatter = logging.Formatter("<%(asctime)s.%(msecs)03d> %(message)s",
                                  "%m-%d %H:%M:%S")
    handler = logging.FileHandler(output_file, mode='a')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


def main(opt):
    # set random seed
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    # initialize
    opt.dataset_split_by = opt.dataset + '_' + opt.split_by
    checkpoint_dir = os.path.join(opt.checkpoint_path, opt.dataset_split_by + '_' + opt.id)
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    logger = initialize_logger(os.path.join(checkpoint_dir, 'train.log'))
    print = logger.info

    # set up loader
    data_json = os.path.join(opt.feats_path, opt.dataset_split_by, opt.data_file + '.json')
    data_pth = os.path.join(opt.feats_path, opt.dataset_split_by, opt.data_file + '.pth')
    visual_feats_dir = os.path.join(opt.feats_path, opt.dataset_split_by, opt.visual_feat_file)

    if os.path.isfile(data_pth):
        loader = GtLoader(data_json, visual_feats_dir, opt, data_pth)
        opt.tag_vocab_size = loader.tag_vocab_size
        opt.dep_vocab_size = loader.dep_vocab_size
    else:
        loader = GtLoader(data_json, visual_feats_dir, opt)

    opt.word_vocab_size = loader.word_vocab_size
    opt.vis_dim = loader.vis_dim

    # print out the option variables
    print("*" * 20)
    for k, v in opt.__dict__.items():
        print("%r: %r" % (k, v))
    print("*" * 20)

    # load previous checkpoint if possible
    infos = {}
    if opt.start_from:
        assert os.path.isdir(opt.start_from), " %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(
            opt.start_from, "infos.json")), "infos.json file does not exist in path %s" % opt.start_from
        print("Load infos ...")
        with open(os.path.join(opt.start_from, 'infos.json'), 'r') as f:
            infos = json.load(f)

    # resume checkpoint or from scratch
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    loader.iterators = infos.get('iterators', loader.iterators)

    # some histories may useful
    loss_history = infos.get('loss_history', {})
    lr_history = infos.get('lr_history', {})
    val_loss_history = infos.get('val_loss_history', {})
    val_accuracies = infos.get('val_accuracies', [])
    test_accuracies = infos.get('test_accuracies', [])
    test_loss_history = infos.get('test_loss_history', {})
    best_val_score = infos.get('best_val_score', None)
    best_epoch = infos.get('best_epoch', 0)

    # set up model and criterion
    model = models.setup(opt, loader).cuda()
    crit = torch.nn.NLLLoss()

    # set up optimizer
    weights, biases = [], []
    for name, p in model.named_parameters():
       if 'bias' in name:
           biases += [p]
       else:
           weights += [p]
    optimizer = train_utils.build_optimizer(weights, biases, opt)

    # check compatibility if training is continued from previously saved model
    if opt.start_from:
        # check if all necessary files exist 
        assert os.path.isfile(os.path.join(opt.start_from, "model.pth"))
        print("Load model ...")
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    # Load the optimizer
    if opt.start_from:
        assert os.path.isfile(os.path.join(opt.start_from, "optimizer.pth"))
        print("Load optimizer ...")
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    # Load the pretrained word vector
    if opt.glove and not opt.start_from:
        glove_weight = load_glove(glove=opt.glove, vocab=loader.word_to_ix, opt=opt)
        assert glove_weight.shape == model.word_embedding.weight.size()
        model.word_embedding.weight.data.set_(torch.cuda.FloatTensor(glove_weight))
        print("Load word vectors ...")

    # start training
    tic = time.time()
    wrapped = False
    while True:
        model.train()

        # decay the learning rates
        if 0 <= opt.learning_rate_decay_start < epoch:
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            decay_factor = opt.learning_rate_decay_rate ** frac
            opt.current_lr = opt.learning_rate * decay_factor
            train_utils.set_lr(optimizer, opt.current_lr)  # update optimizer's learning rate
        else:
            opt.current_lr = opt.learning_rate

        # start training
        optimizer.zero_grad()
        total_loss = 0.0
        n = 0.0
        acc = 0.0
        for _ in range(opt.batch_size):
            # read data
            data = loader.get_data('train')

            wrapped = True if data['bounds']['wrapped'] else wrapped
            torch.cuda.synchronize()

            # model forward
            scores = model(data)
            target = data['gts']
            loss = crit(scores, target)
            loss.backward()
            total_loss += loss.detach().cpu().numpy()

            # compute accuracy
            scores = scores.data.cpu().numpy()
            gt_ix = target.detach().cpu().numpy()
            pred_ix = np.argmax(scores, axis=1)
            n += len(gt_ix)
            acc += sum(pred_ix == gt_ix)

        total_loss /= opt.batch_size
        train_accuracy = acc / n * 100

        # model backward
        train_utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()

        # write the training loss summary
        if iteration % opt.losses_log_every == 0:
            loss_history[iteration] = total_loss
            lr_history[iteration] = opt.current_lr
            print('epoch=%d, iter=%d, train_loss=%.3f, train_acc=%.2f, time=%.2f' %
                  (epoch, iteration, total_loss, train_accuracy, time.time()-tic))
            tic = time.time()

        # eval loss and save checkpoint
        if wrapped and epoch % opt.save_checkpoint_every == 0:
            # evaluate models
            acc, n, val_loss = eval_utils.eval_gt_split(loader, model, crit, 'val', vars(opt))
            val_accuracy = acc / n * 100
            print("%s set evaluated. val_loss = %.2f, acc = %d / %d = %.2f%%" %
                  ('val', val_loss, acc, n, val_accuracy))
            val_loss_history[iteration] = val_loss
            val_accuracies.append(val_accuracy)

            if opt.split_by == 'unc':
                test_split = 'testA'
            else:
                test_split = 'test'
            test_acc, test_n, test_loss = eval_utils.eval_gt_split(loader, model, crit, test_split, vars(opt))
            test_accuracy = test_acc / test_n * 100
            print("%s set evaluated. test_loss = %.2f, acc = %d / %d = %.2f%%" %
                  (test_split, test_loss, test_acc, test_n, test_accuracy))
            test_loss_history[iteration] = test_loss
            test_accuracies.append(test_accuracy)

            # save model
            checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)
            print("model saved to {}".format(checkpoint_path))

            # save infos
            infos['iter'] = iteration+1
            infos['epoch'] = epoch+1
            infos['iterators'] = loader.iterators
            infos['opt'] = vars(opt)
            infos['word_to_ix'] = loader.word_to_ix

            # save histories
            infos['loss_history'] = loss_history
            infos['lr_history'] = lr_history
            infos['val_accuracies'] = val_accuracies
            infos['val_loss_history'] = val_loss_history
            infos['test_accuracies'] = test_accuracies
            infos['test_loss_history'] = test_loss_history
            infos['best_val_score'] = best_val_score
            infos['best_epoch'] = best_epoch

            # save model if best
            current_score = val_accuracy + test_accuracy
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                model_save_path = os.path.join(checkpoint_dir, 'model-best.pth')
                torch.save(model.state_dict(), model_save_path)
                with open(os.path.join(checkpoint_dir, 'infos-best.json'), 'w') as f:
                    json.dump(infos, f, sort_keys = True, indent = 4)
                print("model saved to {}".format(model_save_path))
            else:
                print("The best model in epoch{}: {}".format(best_epoch, best_val_score))

            with open(os.path.join(checkpoint_dir, 'infos.json'), 'w') as f:
                json.dump(infos, f, sort_keys = True, indent = 4)

        # update iteration and epoch
        iteration += 1
        if wrapped:
            wrapped = False
            epoch += 1
            loader.shuffle(split='train')

        if 0 < opt.max_epochs <= epoch:
            break


if __name__ == '__main__':
    opt = opts.parse_opt()
    main(opt)
