from __future__ import division
from builtins import bytes
import os
import argparse
import math
import codecs
import torch

import onmt
import onmt.IO
import opts
from itertools import takewhile, count
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

parser = argparse.ArgumentParser(description='translate.py')
opts.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-cons', required=True,
                    help='Target constraints to decode (one line per sequence)')
parser.add_argument('-src_img_dir',   default="",
                    help='Source image directory')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-attn_debug', action="store_true",
                    help='Print best attn for each word')

parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")
# options most relevant to summarization
parser.add_argument('-dynamic_dict', action='store_true',
                    help="Create dynamic dictionaries")
parser.add_argument('-share_vocab', action='store_true',
                    help="Share source and target vocabulary")


def report_score(name, score_total, words_total):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, score_total / words_total,
        name, math.exp(-score_total/words_total)))


def get_src_words(src_indices, index2str):
    words = []
    raw_words = (index2str[i] for i in src_indices)
    words = takewhile(lambda w: w != onmt.IO.PAD_WORD, raw_words)
    return " ".join(words)


def main():
    opt = parser.parse_args()

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)
    translator = onmt.ConsTranslator(opt, dummy_opt.__dict__)
    out_file = codecs.open(opt.output, 'w', 'utf-8')
    pred_score_total, pred_words_total = 0, 0
    gold_score_total, gold_words_total = 0, 0
    if opt.dump_beam != "":
        import json
        translator.initBeamAccum()
    data = onmt.IO.ONMTDataset(opt.src, opt.tgt, translator.fields, None)

    test_data = onmt.IO.OrderedIterator(
        dataset=data, device=opt.gpu,
        batch_size=opt.batch_size, train=False, sort=False,
        shuffle=False)

    cons_file = codecs.open(opt.cons, encoding='utf-8')
    constraints = cons_file.readlines()
    def parse_constraint(cons):
        return [list(filter(None, part.strip().split(' '))) for part in cons.split('\t')]
    constraints = list(map(parse_constraint, constraints))
    constraints_data = []
    i = 0
    while i < len(constraints):
        constraints_data.append(constraints[i: i + opt.batch_size])
        i += opt.batch_size
    counter = count(1)
    for batch, constraints_batch in zip(test_data, constraints_data):
        best_output = translator.translate(batch, constraints_batch, data)
        best_output_seq = ' '.join(best_output[0][1:])
        out_file.write(best_output_seq + '\n')
        print(best_output_seq)
        # input('Type to continue...')
    out_file.close()

if __name__ == "__main__":
    main()
