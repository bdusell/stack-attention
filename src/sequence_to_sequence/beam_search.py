import more_itertools
import torch

def beam_search(initial_state, beam_size, eos_symbol, max_length, device):
    batch_size = initial_state.batch_size()
    return [
        beam_search_single(
            initial_state.transform_tensors(lambda x: x[i:i+1, ...]),
            beam_size,
            eos_symbol,
            max_length,
            device
        )
        for i in range(batch_size)
    ]

def beam_search_single(initial_state, beam_size, eos_symbol, max_length, device):
    if initial_state.batch_size() != 1:
        raise ValueError
    # At any given point in time, the beam contains at most `beam_size` items.
    # Each item can either be an unfinished sequence that has not yet generated
    # EOS, or a finished sequence that has already generated EOS. In this
    # implementation, the unfinished and finished items are stored in separate
    # tensors.
    # This is a single State object whose batch dimension is used to hold
    # multiple decoder states (at most `beam_size`). It represents a list of
    # all the decoder states corresponding to unfinished beam items, in
    # decreasing order of score.
    beam_unfinished_states = initial_state
    # This contains the log-probabilities for all items on the beam, both
    # unfinished and finished, in decreasing order.
    beam_log_probs = torch.tensor([0.0], device=device)
    # This is a bool tensor of the same size as `beam_log_probs`. It is true
    # for finished items and false for unfinished items.
    beam_is_finished = torch.tensor([False], device=device)
    finished_backpointers = torch.empty((0, 2), dtype=torch.long, device=device)
    unfinished_backpointers = []
    symbols = []
    did_finish = False
    for t in range(max_length):
        # output_log_probs : old_unfinished_beam_size x output_vocab_size
        # TODO It should be possible to take the top k of the logits and only
        # compute log probs for the top k. But it's necessary to pass all the
        # logits to log_softmax() to get the denominators right.
        output_log_probs = torch.nn.functional.log_softmax(
            beam_unfinished_states.output(),
            dim=1
        )
        output_vocab_size = output_log_probs.size(1)
        k = min(beam_size, output_vocab_size)
        # top_output_log_probs : old_unfinished_beam_size x k
        # top_output_indexes : old_unfinished_beam_size x k of ints in [0, output_vocab_size)
        top_output_log_probs, top_output_symbols = torch.topk(
            output_log_probs,
            k=k,
            dim=1,
            sorted=False
        )
        # unfinished_log_probs : old_unfinished_beam_size
        unfinished_log_probs = beam_log_probs[~beam_is_finished]
        # unfinished_candidate_log_probs : old_unfinished_beam_size x k
        unfinished_candidate_log_probs = unfinished_log_probs[:, None] + top_output_log_probs
        # flat_unfinished_candidate_log_probs : old_unfinished_beam_size * k
        flat_unfinished_candidate_log_probs = unfinished_candidate_log_probs.view(-1)
        num_unfinished_candidates, = flat_unfinished_candidate_log_probs.size()
        # finished_log_probs : old_finished_beam_size
        finished_log_probs = beam_log_probs[beam_is_finished]
        # candidate_log_probs : num_candidates = old_unfinished_beam_size * k + old_finished_beam_size
        candidate_log_probs = torch.concat([
            flat_unfinished_candidate_log_probs,
            finished_log_probs
        ])
        # NOTE This implements length normalization.
        candidate_scores = torch.concat([
            flat_unfinished_candidate_log_probs / (t + 1),
            finished_log_probs / (finished_backpointers[:, 0] + 1)
        ])
        # beam_scores : k
        # top_k_indexes : k of ints in [0, num_candidates)
        beam_scores, top_k_indexes = torch.topk(
            candidate_scores,
            k=k,
            dim=0,
            sorted=True
        )
        # beam_log_probs : k
        beam_log_probs = candidate_log_probs[top_k_indexes]
        # is_from_unfinished : k of bool
        is_from_unfinished = top_k_indexes < num_unfinished_candidates
        # This makes sure that any indexes that would be invalid for
        # top_output_indexes are set to 0.
        # masked_top_k_indexes : k of ints in [0, num_unfinished_candidates)
        masked_top_k_indexes = top_k_indexes * is_from_unfinished
        # top_backpointers : k of ints in [0, old_unfinished_beam_size)
        top_backpointers = torch.div(masked_top_k_indexes, k, rounding_mode='floor')
        # top_output_indexes : k of ints in [0, k)
        top_output_indexes = torch.remainder(masked_top_k_indexes, k)
        # new_symbols : k of ints in [0, output_vocab_size)
        new_symbols = top_output_symbols[top_backpointers, top_output_indexes]
        # just_generated_eos : k of bool
        just_generated_eos = is_from_unfinished & (new_symbols == eos_symbol)
        # is_from_finished : k of bool
        is_from_finished = ~is_from_unfinished
        # beam_is_finished : k of bool
        beam_is_finished = just_generated_eos | is_from_finished
        # was_already_finished : new_finished_beam_size of bool
        was_already_finished = is_from_finished[beam_is_finished]
        new_finished_beam_size, = was_already_finished.size()
        # new_finished_backpointers[:, 0] : new_finished_beam_size of ints in [0, t]
        # new_finished_backpointers[:, 1] : new_finished_beam_size of ints in [0, old_unfinished_beam_size)
        new_finished_backpointers = torch.empty((new_finished_beam_size, 2), dtype=torch.long, device=device)
        # For beam items that were already finished, copy their backpointers
        # from the previous timestep.
        # indexes_from : k of ints in [0, old_finished_beam_size)
        indexes_from = top_k_indexes[is_from_finished] - num_unfinished_candidates
        # indexes_to : (num_already_finished,) of ints in [0, new_finished_beam_size)
        indexes_to = torch.nonzero(was_already_finished, as_tuple=True)
        new_finished_backpointers[indexes_to] = finished_backpointers[indexes_from]
        # For beam items that just finished, set their backpointers and timestep.
        # indexes_to : num_just_finished of ints in [0, new_finished_beam_size)
        indexes_to, = torch.nonzero(~was_already_finished, as_tuple=True)
        # just_finished_backpointers : num_just_finished of ints in [0, old_unfinished_beam_size)
        just_finished_backpointers = top_backpointers[just_generated_eos]
        new_finished_backpointers[indexes_to, 0] = t
        new_finished_backpointers[indexes_to, 1] = just_finished_backpointers
        finished_backpointers = new_finished_backpointers
        # If the top of the beam is finished, stop.
        if beam_is_finished[0].item():
            did_finish = True
            break
        # unfinished_beam_indexes : new_unfinished_beam_size of ints in [0, k)
        unfinished_beam_indexes, = torch.nonzero(~beam_is_finished, as_tuple=True)
        # rearranged_backpointers : new_unfinished_beam_size of ints in [0, old_unfinished_beam_size)
        rearranged_backpointers = top_backpointers[unfinished_beam_indexes,]
        # The first set of backpointers always points to 0, so there's no need
        # to save it.
        if t > 0:
            unfinished_backpointers.append(rearranged_backpointers)
        def rearrange(x):
            # x : old_unfinished_beam_size x ...
            # return : new_unfinished_beam_size x ...
            return x[rearranged_backpointers, ...]
        # rearranged_states : State with batch size new_unfinished_beam_size
        rearranged_states = beam_unfinished_states.transform_tensors(rearrange)
        # rearranged_input_symbols : new_unfinished_beam_size of ints in [0, output_vocab_size)
        rearranged_input_symbols = new_symbols[unfinished_beam_indexes,]
        symbols.append(rearranged_input_symbols)
        if t == max_length - 1:
            # If we've reached the last timestep, don't bother updating the
            # state.
            break
        # beam_unfinished_states : State with batch size new_unfinished_beam_size
        beam_unfinished_states = rearranged_states.next(rearranged_input_symbols)
    if did_finish:
        length, start_index = finished_backpointers[0].tolist()
    else:
        # Beam search did not end with a sequence ending in EOS, because the
        # max length was reached. In this case, just start decoding from the
        # best item at the last timestep.
        length = len(symbols)
        start_index = 0
    return follow_backpointers(unfinished_backpointers, symbols, length, start_index)

def follow_backpointers(backpointers, symbols, length, start_index):
    result = []
    curr_backpointer = start_index
    for t in reversed(range(length)):
        symbol = symbols[t][curr_backpointer].item()
        result.append(symbol)
        if t > 0:
            curr_backpointer = backpointers[t-1][curr_backpointer]
    result.reverse()
    return result
