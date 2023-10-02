import copy
import datetime
import itertools
import random

import attr
import numpy
import torch

from torch_extras.early_stopping import UpdatesWithoutImprovement
from lib.pytorch_tools.curriculum import RandomShuffling
from lib.ticker import TimedTicker
from utils.cli_util import parse_interval
from .data_util import get_random_seed, get_random_generator_and_seed, generate_batches

def add_train_arguments(parser):
    group = parser.add_argument_group('Training options')
    parser.add_argument('--no-progress', action='store_true', default=False,
        help='Do not print progress messages.')
    group.add_argument('--save-model', action='store_true', default=False,
        help='Save the model parameters to the output directory so that it '
             'can be re-loaded and evaluated on a test set later. The '
             'parameters will NOT be saved if this flag is not used.')
    group.add_argument('--shuffle-seed', type=int,
        help='Random seed used for random shuffling.')
    add_optimizer_args(group, default='Adam')
    group.add_argument('--learning-rate', type=float, required=True,
        help='Initial learning rate.')
    group.add_argument('--learning-rate-patience', type=int,
        default=5,
        help='Number of epochs of no improvement before the learning rate is '
             'decreased.')
    group.add_argument('--learning-rate-decay', type=float,
        default=0.9,
        help='The learning rate will be multiplied by this factor when '
             'patience runs out. Should be between 0 and 1.')
    group.add_argument('--l2-regularizer-lambda', type=float,
        default=0,
        help='Coefficient for the L2 regularizer.')
    group.add_argument('--gradient-clipping', type=float,
        default=5,
        help='Gradient clipping threshold.')
    group.add_argument('--epochs', type=int,
        default=200,
        help='Maximum number of epochs to run.')
    group.add_argument('--early-stopping-patience', type=int,
        default=10,
        help='Number of epochs of no improvement before training stops.')
    group.add_argument('--vis-data-seed', type=int)
    group.add_argument('--vis-length-range', type=parse_interval)
    group.add_argument('--vis-data-size', type=int, default=5)
    group.add_argument('--vis-updates', type=int)
    group.add_argument('--show-outputs', action='store_true', default=False)
    group.add_argument('--log-stack-signals', action='store_true', default=False)

def parse_optimizer_class(s):
    return getattr(torch.optim, s)

def add_optimizer_args(parser, default='SGD'):
    parser.add_argument('--optimizer', type=parse_optimizer_class,
        default=parse_optimizer_class(default),
        help=f'Type of optimizer to use for training. Default is {default}.')

@attr.s
class TrainState:
    epoch_no = attr.ib()
    update_no = attr.ib()
    batch_no = attr.ib()

def train(args, saver, data, model_interface, events, logger):

    model = saver.model
    show_progress = not args.no_progress

    # Seed the RNG used to shuffle the data during training.
    shuffle_seed = get_random_seed(args.shuffle_seed)
    logger.info(f'shuffle random seed: {shuffle_seed}')
    shuffle_generator = numpy.random.RandomState(shuffle_seed)

    # Generate some data used for visualization.
    if args.vis_length_range is not None:
        vis_valid_lengths = data.sampler.valid_lengths(*args.vis_length_range)
    else:
        vis_valid_lengths = data.train_lengths
    if args.show_outputs or args.log_stack_signals:
        vis_data_size = args.vis_data_size
    else:
        vis_data_size = 0
    vis_data_generator, vis_data_seed = get_random_generator_and_seed(args.vis_data_seed)
    logger.info(f'visualization data random seed: {vis_data_seed}')
    vis_data = list(generate_batches(
        sampler=data.sampler,
        valid_lengths=vis_valid_lengths,
        num_samples=vis_data_size,
        batch_size=1,
        generator=vis_data_generator
    ))
    events.log('vis_data', { 'data' : vis_data })
    device = next(model.parameters()).device
    vis_data = model_interface.batches_to_tensors(vis_data, device, data.input_vocab, data.output_vocab)
    if args.vis_updates:
        vis_updates = args.vis_updates
    else:
        vis_updates = len(data.train_data)

    regularizer = model_interface.get_regularizer(args, saver)

    # Configure the optimization process.
    OptimizerClass = args.optimizer
    optimizer = OptimizerClass(
        itertools.chain(model.parameters(), regularizer.parameters()),
        lr=args.learning_rate,
        weight_decay=args.l2_regularizer_lambda
    )
    early_stopping = UpdatesWithoutImprovement(
        'min',
        patience=args.early_stopping_patience
    )
    early_stopping_metric = 'perplexity'
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=early_stopping.mode,
        patience=args.learning_rate_patience,
        factor=args.learning_rate_decay
    )
    learning_curriculum = RandomShuffling(data.train_data, shuffle_generator)

    # Configure the loss function.
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    train_state = TrainState(epoch_no=None, update_no=0, batch_no=None)

    total_start_time = datetime.datetime.now()
    epoch_no = None
    best_epoch = None
    best_parameters = None
    best_metrics = None
    for epoch_no in range(args.epochs):
        train_state.epoch_no = epoch_no
        epoch_start_time = datetime.datetime.now()
        logger.info(f'epoch #{epoch_no + 1}')

        # Update the learning curriculum (e.g. randomly shuffle the data).
        learning_curriculum.step()

        # Show the current learning rate.
        curr_learning_rate = optimizer.param_groups[0]['lr']
        logger.info(f'  learning rate: {curr_learning_rate}')

        ticker = TimedTicker(len(learning_curriculum.data()), 1)
        train_loss = 0.0
        curr_loss = 0.0
        train_perp_numer = 0.0
        curr_perp_numer = 0.0
        train_num_symbols = 0
        curr_num_symbols = 0

        for batch_no, (x, y) in enumerate(learning_curriculum.data()):
            train_state.batch_no = batch_no
            optimizer.zero_grad()
            # Evaluate the model (forward pass).
            # logits : B x n x V
            model.train()
            logits, regularization_term = model_interface.get_logits_and_regularization_term(
                saver,
                x,
                regularizer,
                train_state
            )
            # Get the loss term for each symbol.
            # y : B x n
            # symbol_losses : B x n
            symbol_losses = criterion(logits.transpose(1, 2), y)
            # Sum over time steps to get the loss for each sequence in the
            # batch.
            # sequence_losses : B
            sequence_losses = torch.sum(symbol_losses, dim=1)
            if regularization_term is not None:
                batch_element_losses = sequence_losses + regularization_term
            else:
                batch_element_losses = sequence_losses
            # Average across batch elements to get the loss.
            loss = torch.mean(batch_element_losses, dim=0)
            # Update statistics.
            detached_loss = loss.detach()
            with torch.no_grad():
                perp_numer = torch.sum(sequence_losses.detach())
            num_symbols = y.numel()
            train_loss += detached_loss
            curr_loss += detached_loss
            train_perp_numer += perp_numer
            curr_perp_numer += perp_numer
            train_num_symbols += num_symbols
            curr_num_symbols += num_symbols
            # Run backprop.
            loss.backward()
            # Gradient clipping.
            if args.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clipping)
            # Update parameters.
            optimizer.step()
            train_state.update_no += 1
            # Show training progress.
            ticker.progress = batch_no + 1
            if show_progress and ticker.tick():
                avg_curr_loss = curr_loss / curr_num_symbols
                avg_curr_perplexity = torch.exp(curr_perp_numer / curr_num_symbols)
                avg_curr_loss_item = avg_curr_loss.item()
                avg_curr_perplexity_item = avg_curr_perplexity.item()
                logger.info(
                    f'  {ticker.int_percent}% | '
                    f'loss: {avg_curr_loss_item:.2f} | '
                    f'perplexity: {avg_curr_perplexity_item:.2f}')
                curr_loss = 0.0
                curr_perp_numer = 0.0
                curr_num_symbols = 0
            # Log some information on some example data for analysis.
            if train_state.update_no % vis_updates == 0:
                if args.show_outputs:
                    for vis_batch in vis_data:
                        model_interface.print_example(saver, vis_batch, data.vocab, logger)
                if args.log_stack_signals:
                    signals_list = []
                    for x, y in vis_data:
                        signals = model_interface.get_signals(saver, x, train_state)
                        signals_list.append(signals)
                    events.log('signals', { 'values' : signals_list })
        # Summarize training progress for this epoch.
        avg_train_loss = train_loss / train_num_symbols
        avg_train_perplexity = torch.exp(train_perp_numer / train_num_symbols)
        avg_train_loss_item = avg_train_loss.item()
        avg_train_perplexity_item = avg_train_perplexity.item()
        logger.info(
            f'  average loss: {avg_train_loss_item:.2f} | '
            f'average perplexity: {avg_train_perplexity_item:.2f}')
        events.log('epoch', {
            'loss' : avg_train_loss_item,
            'perplexity' : avg_train_perplexity_item
        })

        # Evaluate the model on the validation set.
        valid_scores = evaluate(
            saver,
            data.valid_data,
            model_interface,
            show_progress,
            logger,
            logger_indent='  ',
            compute_accuracy=False
        )
        events.log('validate', valid_scores)
        score = valid_scores[early_stopping_metric]
        logger.info(f'  dev perplexity: {valid_scores["perplexity"]:.3f}')
        # Adjust the learning rate based on the score.
        if learning_curriculum.done():
            lr_scheduler.step(score)
        is_best, should_stop = early_stopping.update(score)
        # Save the best model seen so far.
        if is_best:
            if args.save_model:
                logger.info('  saving parameters')
                saver.save()
            best_parameters = copy.deepcopy(model.state_dict())
            best_epoch = epoch_no
            best_metrics = valid_scores
        # Early stopping.
        if should_stop:
            last_epoch = epoch_no
            break
        epoch_duration = datetime.datetime.now() - epoch_start_time
        logger.info(f'  epoch duration: {epoch_duration}')
    else:
        last_epoch = epoch_no
    total_duration = datetime.datetime.now() - total_start_time
    logger.info(f'total duration: {total_duration}')
    # Restore the parameters of the best model.
    if best_parameters is not None:
        model.load_state_dict(best_parameters)
    logger.info(f'best epoch: {best_epoch + 1}')
    if best_metrics is not None:
        logger.info(f'best dev perplexity: {best_metrics["perplexity"]:.3f}')
    result = {
        'best_epoch' : best_epoch + 1,
        'last_epoch' : last_epoch + 1,
        'best_validation_metrics' : best_metrics,
        'epochs_since_improvement' : early_stopping.updates_since_improvement
    }
    events.log('train', result)
    return result

def evaluate(saver, batches, model_interface, show_progress, logger,
        logger_indent='', compute_accuracy=True):
    s = logger_indent
    perp_numer = 0.0
    acc_numer = 0
    num_symbols = 0
    saver.model.eval()
    with torch.no_grad():
        ticker = TimedTicker(len(batches), 1)
        for batch_no, (x, y_target) in enumerate(batches):
            # Let B be batch size, n be sequence length, and V be vocabulary size.
            # y_logits : B x n x V
            # y_target : B x n of int in [0, V-1]
            y_logits = model_interface.get_logits(saver, x, None)
            # Compute the numerator for the perplexity score.
            # -log(p_M(y_target)) = -\sum_t log(p_M(y_target_{t} | y_target_{<t}))
            y_neg_log_prob = torch.nn.functional.cross_entropy(
                y_logits.transpose(1, 2),
                y_target,
                reduction='sum')
            perp_numer += y_neg_log_prob
            if compute_accuracy:
                # Compute the numerator for the accuracy score.
                # Get the symbol with the highest score at each timestep.
                y_pred = torch.argmax(y_logits, dim=2)
                num_correct = torch.eq(y_pred, y_target).sum()
                acc_numer += num_correct
            num_symbols += y_target.numel()
            ticker.progress = batch_no + 1
            if show_progress and ticker.tick():
                logger.info(f'{s}{ticker.int_percent}%')
    perplexity = torch.exp(perp_numer / num_symbols)
    result = {
        'perplexity' : perplexity.item(),
        'perplexity_numerator' : perp_numer.item(),
        'num_symbols' : num_symbols
    }
    if compute_accuracy:
        accuracy = acc_numer / num_symbols
        result.update({
            'accuracy' : accuracy.item(),
            'accuracy_numerator' : acc_numer.item()
        })
    return result
