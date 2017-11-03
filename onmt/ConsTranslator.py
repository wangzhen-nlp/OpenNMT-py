import torch
from torch.autograd import Variable

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
import onmt.IO
from onmt.Utils import use_gpu

import numpy as np

from onmt.Cons import *

import copy

class ConsTranslator(object):
    def __init__(self, opt, dummy_opt={}):
        # Add in default model arguments, possibly added since training.
        self.opt = opt
        checkpoint = torch.load(opt.model,
                                map_location=lambda storage, loc: storage)
        self.fields = onmt.IO.ONMTDataset.load_fields(checkpoint['vocab'])

        model_opt = checkpoint['opt']
        for arg in dummy_opt:
            if arg not in model_opt:
                model_opt.__dict__[arg] = dummy_opt[arg]

        self._type = model_opt.encoder_type
        self.copy_attn = model_opt.copy_attn

        self.model = onmt.ModelConstructor.make_base_model(
                            model_opt, self.fields, use_gpu(opt), checkpoint)
        self.model.eval()
        self.model.generator.eval()

        # for debugging
        self.beam_accum = None
        self.eos_token_id = self.fields['tgt'].vocab.stoi[onmt.IO.EOS_WORD]
        self.bos_token_id = self.fields['tgt'].vocab.stoi[onmt.IO.BOS_WORD]
        self.decoder = self.create_constrained_decoder()

    def create_constrained_decoder(self):
        decoder = ConstrainedDecoder(hyp_generation_func=self.generate,
                                     constraint_generation_func=self.generate_constrained,
                                     continue_constraint_func=self.continue_constrained,
                                     beam_implementation=Beam)
        return decoder

    def initBeamAccum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}

    def generate_blacks(self, constraints, is_str=False):
        blacks = [float('inf') if '_copy_' in token else 0
            for token in self.fields['tgt'].vocab.itos]

        def flatten(array):
            if isinstance(array, list):
                for item in array:
                    yield from flatten(item)
            else:
                yield array
        constraints = list(flatten(constraints))

        def constraint_to_idx(x):
            return self.fields['tgt'].vocab.stoi[x]
        if is_str:
            constraints = map(constraint_to_idx, constraints)

        for cons in constraints:
            blacks[cons] = 0.
        return blacks 
 
    def translateBatch(self, batch, constraints, dataset, length_factor=1.3):
        beam_size = self.opt.beam_size
        batch_size = batch.batch_size
        assert batch_size == 1
        constraints = constraints[0]

        # (1) Run the encoder on the src.
        _, src_lengths = batch.src
        src = onmt.IO.make_features(batch, 'src')
        encStates, contexts = self.model.encoder(src, src_lengths)
        decStates = self.model.decoder.init_decoder_state(
                                        src, contexts, encStates)

        # Note: tile 1x because minibatch size is effectively 1
        target_prefix = self.bos_token_id

        blacks = self.generate_blacks(constraints)

        start_hyp = self.start_hypothesis(contexts=contexts, states=decStates,
                                          target_prefix=target_prefix,
                                          constraints=constraints, blacks=blacks)
        search_grid = self.decoder.search(start_hyp=start_hyp, constraints=constraints,
                                     max_hyp_len=100,
                                     beam_size=5)
        best_output = self.decoder.best_n(search_grid, onmt.IO.EOS_WORD, n_best=1)

        return best_output

    def translate(self, batch, constraints, data):
        #  (1) convert words to indexes
        batch_size = batch.batch_size

        def convert_constraint_idx(cons):
            return self.fields['tgt'].vocab.stoi[cons]
        constraints = [[[convert_constraint_idx(i) for i in j] for j in k] for k in constraints] 
        #  (2) translate
        best_output = self.translateBatch(batch, constraints, data)
        return best_output

    def start_hypothesis(self, contexts, states, target_prefix, constraints, coverage=None, blacks=None):
        coverage = [np.zeros(l, dtype='int16') for l in [len(s) for s in constraints]]
        payload = {
            'contexts': contexts,
            'states': states,
            'input_values': target_prefix,
        }
        payload['input_values'] = \
            Variable(torch.LongTensor(np.asarray([[[payload['input_values']]]], dtype=np.int64)))
        if contexts.is_cuda:
            new_payload['input_values'] = new_payload['input_values'].cuda()
        start_hyp = ConstraintHypothesis(
            token=onmt.IO.BOS_WORD,
            score=0,
            coverage=coverage,
            constraints=constraints,
            payload=payload,
            backpointer=None,
            constraint_index=None,
            unfinished_constraint=False,
            blacks=blacks
        )
        return start_hyp

    def generate(self, hyp, n_best):
        """
        Note: the `n_best` parameter here is only used to limit the number of hypothesis objects that are generated
        from the input hyp, the beam implementation may specify a different `n_best`
        """
        # if we already generated EOS, there's only one option -- just continue it and copy the cost
        if hyp.token == onmt.IO.EOS_WORD:
            new_hyp = ConstraintHypothesis(
                token=hyp.token,
                score=hyp.score,
                coverage=copy.deepcopy(hyp.coverage),
                constraints=hyp.constraints,
                payload=hyp.payload,
                backpointer=hyp,
                constraint_index=None,
                unfinished_constraint=False,
                blacks=hyp.blacks
            )
            return [new_hyp]
            # return []
 
        # print(hyp.payload['input_values'])
        # print(hyp.payload['states'].hidden.repeat(1, 2, 1))
        # print(hyp.payload['states'].input_feed.repeat(1, 2, 1))
        decOut, decStates, attn = \
            self.model.decoder(hyp.payload['input_values'],
                               hyp.payload['contexts'],
                               hyp.payload['states'])
        decOut = decOut.squeeze(0)
        assert not self.copy_attn
        out = self.model.generator.forward(decOut).data

        logprobs = out.view(-1).cpu().numpy() - hyp.blacks
        n_best_outputs = np.argsort(logprobs.flatten())[::-1][:n_best]
        chosen_costs = logprobs.flatten()[n_best_outputs]

        payload = hyp.payload

        # create ContstrainedHypothesis objects from these states (tile back down to one)
        new_hyps = []
        for hyp_idx in range(n_best):
            new_payload = {}
            new_payload['contexts'] = payload['contexts']
            new_payload['states'] = decStates
            new_payload['input_values'] = n_best_outputs[hyp_idx]

            # TODO: account for EOS continuations -- i.e. make other costs infinite
            if hyp.score is not None:
                next_score = hyp.score + chosen_costs[hyp_idx]
            else:
                # hyp.score is None for the start hyp
                next_score = chosen_costs[hyp_idx]

            new_payload['input_values'] = \
                Variable(torch.LongTensor(np.asarray([[[new_payload['input_values']]]], dtype=np.int64)))
            if out.is_cuda:
                new_payload['input_values'] = new_payload['input_values'].cuda()
            new_hyp = ConstraintHypothesis(
                token=self.fields['tgt'].vocab.itos[n_best_outputs[hyp_idx]],
                score=next_score,
                coverage=copy.deepcopy(hyp.coverage),
                constraints=hyp.constraints,
                payload=new_payload,
                backpointer=hyp,
                constraint_index=None,
                unfinished_constraint=False,
                blacks=hyp.blacks
            )

            new_hyps.append(new_hyp)

        return new_hyps

    def generate_constrained(self, hyp):
        """Use hyp.constraints and hyp.coverage to return new hypothesis which start constraints
        that are not yet covered by this hypothesis.

        """
        assert hyp.unfinished_constraint is not True, 'hyp must not be part of an unfinished constraint'

        new_constraint_hyps = []
        available_constraints = hyp.constraint_candidates()

        # TODO: if the model knows about constraints, getting the score from the model must be done differently
        # TODO: currently, according to the model, there is no difference between generating and choosing from constraints
        decOut, decStates, attn = \
            self.model.decoder(hyp.payload['input_values'],
                               hyp.payload['contexts'],
                               hyp.payload['states'])
        decOut = decOut.squeeze(0)
        assert not self.copy_attn
        out = self.model.generator.forward(decOut).data

        logprobs = out.view(-1).cpu().numpy()

        for idx in available_constraints:
            # start new constraints
            constraint_idx = hyp.constraints[idx][0]
            new_payload = {}
            new_payload['contexts'] = hyp.payload['contexts']
            new_payload['states'] = decStates
            new_payload['input_values'] = constraint_idx

            # get the score for this token from the logprobs
            if hyp.score is not None:
                next_score = hyp.score + logprobs[constraint_idx]
            else:
                # hyp.score is None for the start hyp
                next_score = logprobs[constraint_idx]

            coverage = copy.deepcopy(hyp.coverage)
            coverage[idx][0] = 1

            if len(coverage[idx]) > 1:
                unfinished_constraint = True
            else:
                unfinished_constraint = False
            new_payload['input_values'] = \
                Variable(torch.LongTensor(np.asarray([[[new_payload['input_values']]]], dtype=np.int64)))
            if out.is_cuda:
                new_payload['input_values'] = new_payload['input_values'].cuda()
            # TODO: if the model knows about constraints, getting the score from the model must be done differently
            new_hyp = ConstraintHypothesis(token=self.fields['tgt'].vocab.itos[constraint_idx],
                                           score=next_score,
                                           coverage=coverage,
                                           constraints=hyp.constraints,
                                           payload=new_payload,
                                           backpointer=hyp,
                                           constraint_index=(idx, 0),
                                           unfinished_constraint=unfinished_constraint,
                                           blacks=hyp.blacks
                                          )
            new_constraint_hyps.append(new_hyp)

        return new_constraint_hyps

    def continue_constrained(self, hyp):
        assert hyp.unfinished_constraint is True, 'hyp must be part of an unfinished constraint'

        # Note: if the model knows about constraints, getting the score from the model must be done differently
        # Note: according to this model, there is no difference between generating and choosing from constraints
        decOut, decStates, attn = \
            self.model.decoder(hyp.payload['input_values'],
                               hyp.payload['contexts'],
                               hyp.payload['states'])
        decOut = decOut.squeeze(0)
        assert not self.copy_attn
        out = self.model.generator.forward(decOut).data

        logprobs = out.view(-1).cpu().numpy()

        constraint_row_index = hyp.constraint_index[0]
        # the index of the next token in the constraint
        constraint_tok_index = hyp.constraint_index[1] + 1
        constraint_index = (constraint_row_index, constraint_tok_index)

        continued_constraint_token = hyp.constraints[constraint_index[0]][constraint_index[1]]

        # get the score for this token from the logprobs
        if hyp.score is not None:
            next_score = hyp.score + logprobs[continued_constraint_token]
        else:
            # hyp.score is None for the start hyp
            next_score = logprobs[continued_constraint_token]

        coverage = copy.deepcopy(hyp.coverage)
        coverage[constraint_row_index][constraint_tok_index] = 1

        if len(hyp.constraints[constraint_row_index]) > constraint_tok_index + 1:
            unfinished_constraint = True
        else:
            unfinished_constraint = False

        new_payload = defaultdict(OrderedDict)
        new_payload['contexts'] = hyp.payload['contexts']
        new_payload['states'] = decStates
        new_payload['input_values'] = continued_constraint_token

        new_payload['input_values'] = \
             Variable(torch.LongTensor(np.asarray([[[new_payload['input_values']]]], dtype=np.int64)))
        if out.is_cuda:
            new_payload['input_values'] = new_payload['input_values'].cuda()
        new_hyp = ConstraintHypothesis(token=self.fields['tgt'].vocab.itos[continued_constraint_token],
                                       score=next_score,
                                       coverage=coverage,
                                       constraints=hyp.constraints,
                                       payload=new_payload,
                                       backpointer=hyp,
                                       constraint_index=constraint_index,
                                       unfinished_constraint=unfinished_constraint,
                                       blacks=hyp.blacks)

        return new_hyp
