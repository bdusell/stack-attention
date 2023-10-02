import dataclasses
import datetime
import random

import humanfriendly
import torch

from lib.ticker import TimedTicker
from utils.profile_torch import reset_memory_profiler, get_peak_memory
from torch_extras.early_stopping import UpdatesWithoutImprovement

from .batcher import add_batching_arguments, get_batcher

def add_train_arguments(parser):
    group = parser.add_argument_group('Training options')
    group.add_argument('--no-progress', action='store_true', default=False)
    group.add_argument('--epochs', type=int, required=True)
    group.add_argument('--random-shuffling-seed', type=int)
    add_filtering_arguments(group)
    group.add_argument('--filter-validation-data', action='store_true', default=False)
    add_batching_arguments(group)
    add_optimizer_arguments(group)
    add_parameter_update_arguments(group)
    group.add_argument('--early-stopping-patience', type=int, required=True)
    group.add_argument('--learning-rate-patience', type=int, required=True)
    group.add_argument('--learning-rate-decay-factor', type=float, required=True)
    group.add_argument('--checkpoint-interval-sequences', type=int, required=True)

def add_filtering_arguments(group):
    group.add_argument('--max-length', type=int)

def add_optimizer_arguments(group):
    group.add_argument('--optimizer', choices=['SGD', 'Adam'], default='Adam')
    group.add_argument('--learning-rate', type=float, required=True)

def add_parameter_update_arguments(group):
    group.add_argument('--label-smoothing', type=float, default=0.0)
    group.add_argument('--gradient-clipping-threshold', type=float)

def filter_pairs(data, batcher, max_length):
    if max_length is not None:
        data = filter(
            lambda pair: len(pair[0]) <= max_length and len(pair[1]) <= max_length,
            data
        )
    return [
        pair
        for pair in data
        if batcher.is_small_enough(1, len(pair[0]), len(pair[1]))
    ]

def train(parser, args, saver, data, model_interface, events, logger):
    """
    NOTE: When this function returns, the model's parameters will be those of
    the *last* epoch, not necessarily the *best* epoch.
    """
    do_show_progress = not args.no_progress
    device = model_interface.get_device(args)
    do_profile_memory = device.type == 'cuda'
    random_shuffling_generator, random_shuffling_seed = \
        get_random_generator_and_seed(args.random_shuffling_seed)
    logger.info(f'random shuffling seed: {random_shuffling_seed}')
    optimizer = get_optimizer(saver, args)
    validation_criterion = 'cross_entropy_per_token'
    early_stopping = UpdatesWithoutImprovement(
        'min',
        patience=args.early_stopping_patience
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=early_stopping.mode,
        patience=args.learning_rate_patience,
        factor=args.learning_rate_decay_factor
    )
    num_training_examples_before = len(data.training_data)
    logger.info(f'training examples before filtering: {num_training_examples_before}')
    batcher = get_batcher(parser, args, model_interface)
    data.training_data = filter_pairs(
        data.training_data,
        batcher,
        args.max_length
    )
    num_training_examples_after = len(data.training_data)
    logger.info(f'training examples after filtering: {num_training_examples_after}')
    logger.info(f'training examples filtered: {num_training_examples_before - num_training_examples_after}')
    num_validation_examples = len(data.validation_data)
    logger.info(f'validation examples: {num_validation_examples}')
    if args.filter_validation_data:
        data.validation_data = filter_pairs(
            data.validation_data,
            batcher,
            args.max_length
        )
    validation_batches = list(batcher.generate_batches(data.validation_data))
    logger.info(f'validation batches: {len(validation_batches)}')
    model_interface.on_before_process_pairs(
        saver,
        [data.training_data, data.validation_data]
    )
    data.validation_data = None
    events.log('start_training', dict(
        training_examples_before_filtering=num_training_examples_before,
        training_examples_after_filtering=num_training_examples_after,
        validation_examples=num_validation_examples,
        validation_batches=len(validation_batches),
        epochs=args.epochs,
        random_shuffling_seed=random_shuffling_seed,
        batching_max_memory=args.batching_max_memory,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=args.early_stopping_patience,
        learning_rate_patience=args.learning_rate_patience,
        learning_rate_decay_factor=args.learning_rate_decay_factor,
        gradient_clipping_threshold=args.gradient_clipping_threshold,
        checkpoint_interval_sequences=args.checkpoint_interval_sequences
    ))
    epoch_no = 0
    sequences_since_checkpoint = 0
    checkpoint_no = 0
    best_validation_scores = None
    best_checkpoint_no = None
    best_epoch_no = None
    total_start_time = datetime.datetime.now()
    for _ in range(args.epochs):
        epoch_start_time = datetime.datetime.now()
        logger.info(f'epoch #{epoch_no + 1}')
        random_shuffling_generator.shuffle(data.training_data)
        batches = list(batcher.generate_batches(data.training_data))
        random_shuffling_generator.shuffle(batches)
        epoch_loss = LossAccumulator()
        if do_show_progress:
            progress_loss = LossAccumulator()
            progress_num_examples = 0
            progress_start_time = datetime.datetime.now()
            ticker = TimedTicker(len(batches), 1)
        if do_profile_memory:
            reset_memory_profiler(device)
        should_stop = False
        for batch_no, batch in enumerate(batches):
            try:
                result = run_parameter_update(
                    saver,
                    optimizer,
                    batch,
                    model_interface,
                    data,
                    device,
                    args
                )
                epoch_loss.update(result.loss_numer, result.num_symbols)
                if do_show_progress:
                    progress_loss.update(result.loss_numer, result.num_symbols)
            except OutOfCUDAMemoryError as e:
                handle_out_of_cuda_memory(data, events, logger, device, batch, e.parts)
                raise
            batch_size = len(batch)
            if do_show_progress:
                progress_num_examples += batch_size
                ticker.progress = batch_no + 1
                if ticker.tick():
                    progress_loss_value = progress_loss.get_value()
                    progress_duration = datetime.datetime.now() - progress_start_time
                    progress_examples_per_second = progress_num_examples / progress_duration.total_seconds()
                    logger.info(
                        f'  {ticker.int_percent}% '
                        f'| loss: {progress_loss_value:.2f} '
                        f'| examples/s: {progress_examples_per_second:.2f}'
                    )
                    progress_loss = LossAccumulator()
                    progress_start_time = datetime.datetime.now()
                    progress_num_examples = 0
            sequences_since_checkpoint += batch_size
            if sequences_since_checkpoint >= args.checkpoint_interval_sequences:
                logger.info(f'  checkpoint #{checkpoint_no + 1}')
                validation_scores = evaluate(
                    saver.model,
                    validation_batches,
                    data,
                    model_interface,
                    device
                )
                validation_score = validation_scores[validation_criterion]
                logger.info(f'    validation cross entropy: {validation_score:.2f}')
                # Update the learning rate.
                lr_scheduler.step(validation_score)
                # Show the current learning rate.
                curr_learning_rate = optimizer.param_groups[0]['lr']
                logger.info(f'    learning rate: {curr_learning_rate}')
                # Decide whether to save the model parameters and whether to
                # stop early.
                is_best, should_stop = early_stopping.update(validation_score)
                if is_best:
                    logger.info('    saving parameters')
                    saver.save()
                    best_validation_scores = validation_scores
                    best_checkpoint_no = checkpoint_no
                    best_epoch_no = epoch_no
                events.log('checkpoint', dict(
                    is_best=is_best,
                    scores=validation_scores
                ))
                # Reset the count of sequences seen since the last checkpoint.
                # If `sequences_since_checkpoint` is not exactly equal to
                # `args.checkpoint_interval_sequences` after `batch_size` is
                # added to it, but is greater than it, include the extra
                # sequences in the updated count.
                sequences_since_checkpoint %= args.checkpoint_interval_sequences
                checkpoint_no += 1
                if should_stop:
                    logger.info('  stopping early')
                    break
        if should_stop:
            break
        epoch_loss_value = epoch_loss.get_value()
        epoch_duration = datetime.datetime.now() - epoch_start_time
        logger.info(f'  epoch loss: {epoch_loss_value:.2f}')
        logger.info(f'  epoch duration: {epoch_duration}')
        if do_profile_memory:
            peak_memory = get_peak_memory(device)
            logger.info(f'  peak CUDA memory: {humanfriendly.format_size(peak_memory)}')
        else:
            peak_memory = None
        events.log('epoch', dict(
            loss=epoch_loss_value,
            duration=epoch_duration.total_seconds(),
            peak_memory=peak_memory
        ))
        epoch_no += 1
    total_duration = datetime.datetime.now() - total_start_time
    if best_validation_scores is None:
        raise ValueError(
            'the maximum number of epochs has been reached, but no '
            'checkpoints have been made'
        )
    best_validation_score = best_validation_scores[validation_criterion]
    logger.info(f'best validation cross entropy: {best_validation_score:.2f}')
    logger.info(f'completed epochs: {epoch_no}')
    logger.info(f'best epoch: #{best_epoch_no+1}')
    logger.info(f'completed checkpoints: {checkpoint_no}')
    logger.info(f'best checkpoint: #{best_checkpoint_no+1}')
    logger.info(f'checkpoints since improvement: {early_stopping.updates_since_improvement}')
    logger.info(f'total training duration: {total_duration}')
    events.log('train', dict(
        best_validation_scores=best_validation_scores,
        num_epochs=epoch_no,
        best_epoch=best_epoch_no,
        num_checkpoints=checkpoint_no,
        best_checkpoint=best_checkpoint_no,
        checkpoints_since_improvement=early_stopping.updates_since_improvement,
        duration=total_duration.total_seconds()
    ))

def evaluate(model, batches, data, model_interface, device):
    model.eval()
    with torch.no_grad():
        cumulative_loss = LossAccumulator()
        for batch in batches:
            parts = get_loss_parts(
                model,
                batch,
                data,
                model_interface,
                device,
                reduction='sum',
                label_smoothing=0.0
            )
            cumulative_loss.update(parts.cross_entropy.item(), parts.num_symbols)
    return dict(cross_entropy_per_token=cumulative_loss.get_value())

def get_optimizer(saver, args):
    OptimizerClass = getattr(torch.optim, args.optimizer)
    return OptimizerClass(
        saver.model.parameters(),
        lr=args.learning_rate
    )

@dataclasses.dataclass
class LossParts:
    cross_entropy: torch.Tensor
    num_symbols: int
    source_input_size: torch.Size
    target_input_size: torch.Size
    target_output_size: torch.Size

def get_loss_parts(model, batch, data, model_interface, device, reduction, label_smoothing):
    pad_index = len(data.target_output_vocab)
    model_input, correct_target = model_interface.prepare_batch(batch, device, data)
    logits = model_interface.get_logits(model, model_input)
    cross_entropy = torch.nn.functional.cross_entropy(
        logits.permute(0, 2, 1),
        correct_target,
        ignore_index=pad_index,
        reduction=reduction,
        label_smoothing=label_smoothing
    )
    num_symbols = torch.sum(correct_target != pad_index).item()
    return LossParts(
        cross_entropy,
        num_symbols,
        model_input.source.size(),
        model_input.target.size(),
        correct_target.size()
    )

class OutOfCUDAMemoryError(RuntimeError):

    def __init__(self, parts):
        super().__init__()
        self.parts = parts

def run_parameter_update(
    saver,
    optimizer,
    batch,
    model_interface,
    data,
    device,
    args
):
    parts = None
    try:
        optimizer.zero_grad()
        saver.model.train()
        parts = get_loss_parts(
            saver.model,
            batch,
            data,
            model_interface,
            device,
            reduction='none',
            label_smoothing=args.label_smoothing
        )
        sequence_loss = torch.sum(parts.cross_entropy, dim=1)
        parts.cross_entropy = None
        loss = torch.mean(sequence_loss)
        loss_numer = torch.sum(sequence_loss.detach()).item()
        del sequence_loss
        loss.backward()
        del loss
        if args.gradient_clipping_threshold is not None:
            torch.nn.utils.clip_grad_norm_(
                saver.model.parameters(),
                args.gradient_clipping_threshold
            )
        optimizer.step()
        return ParameterUpdateResult(loss_numer, parts.num_symbols)
    except torch.cuda.OutOfMemoryError as e:
        raise OutOfCUDAMemoryError(parts) from e

@dataclasses.dataclass
class ParameterUpdateResult:
    loss_numer: float
    num_symbols: int

class LossAccumulator:

    def __init__(self):
        super().__init__()
        self.numerator = 0.0
        self.denominator = 0

    def update(self, numerator, denominator):
        self.numerator += numerator
        self.denominator += denominator

    def get_value(self):
        return self.numerator / self.denominator

def handle_out_of_cuda_memory(data, events, logger, device, batch, parts):
    logger.info('  out of CUDA memory')
    logger.info(torch.cuda.memory_summary(device))
    peak_memory = get_peak_memory(device)
    logger.info(f'  peak CUDA memory: {humanfriendly.format_size(peak_memory)}')
    logger.info(f'  source input size: {tuple(parts.source_input_size) if parts is not None else None}')
    logger.info(f'  target input size: {tuple(parts.target_input_size) if parts is not None else None}')
    logger.info(f'  target output size: {tuple(parts.target_output_size) if parts is not None else None}')
    source_tokens = sum(len(s) for s, t in batch)
    logger.info(f'  source tokens: {source_tokens}')
    target_tokens = sum(len(t) for s, t in batch)
    logger.info(f'  target tokens: {target_tokens}')
    lengths = [(len(s), len(t)) for s, t in batch]
    logger.info(f'  sequence lengths: {lengths}')
    token_strs = [
        (
            [data.source_vocab.to_string(w) for w in s],
            [data.target_output_vocab.to_string(w) for w in t]
        )
        for s, t in batch
    ]
    sequences_str = '\n'.join(f'{" ".join(s)}\t{" ".join(t)}' for s, t in token_strs)
    logger.info(f'  sequences:\n{sequences_str}')
    events.log('out_of_cuda_memory', dict(
        peak_memory=peak_memory,
        source_input_size=list(parts.source_input_size) if parts is not None else None,
        target_input_size=list(parts.target_input_size) if parts is not None else None,
        target_output_size=list(parts.target_output_size) if parts is not None else None,
        sequences=token_strs
    ))

def get_random_generator_and_seed(random_seed):
    random_seed = get_random_seed(random_seed)
    return random.Random(random_seed), random_seed

def get_random_seed(random_seed):
    return random.getrandbits(32) if random_seed is None else random_seed
