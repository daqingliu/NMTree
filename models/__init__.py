from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def setup(opt, loader):
    if opt.grounding_model == 'NMTree':
        from models.NMTreeModel import NMTreeModel
        model = NMTreeModel(opt, loader)
    else:
        raise Exception("Grounding model not supported: {}".format(opt.grounding_model))

    return model
