"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import glob
import math
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler

sys.path.append('./unilm-v1')
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering

from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from nn.data_parallel import DataParallelImbalance
import biunilm.seq2seq_loader as seq2seq_loader
import torch.distributed as dist


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# bert输入：query+passage
# 改成unilm的输入格式： passage + query
# utils_tweet_passage+query(label的bug未改)
# from utils_tweet_para_query import (read_tweet_examples, convert_examples_to_features,
#                          RawResult, write_predictions,
#                          RawResultExtended, write_predictions_extended)

from script.utils_tweet import (read_tweet_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)

from script.tweetqa_eval import evaluate as evaluate_on_tweetqa

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

# 解决乱码输出
def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)

# 设置随机种子
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def train(args, train_dataset, ex_model, tokenizer, ge_model=None,):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer
    param_optimizer = list(ex_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.fp16:
        try:
            # from apex.optimizers import FP16_Optimizer
            from pytorch_pretrained_bert.optimization_fp16 import FP16_Optimizer_State
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer_State(
                optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer_State(
                optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
                             
    recover_step = _get_max_epoch_model(args.output_dir)
    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)), map_location='cpu')
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        if args.loss_scale == 0:
            logger.info("***** Recover optimizer: dynamic_loss_scale *****")
            optimizer.dynamic_loss_scale = True

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        ex_model = torch.nn.DataParallel(ex_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        ex_model = torch.nn.parallel.DistributedDataParallel(ex_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    
    # 显存释放
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    ex_model.zero_grad()
    if recover_step:
        start_epoch = recover_step+1
    else:
        start_epoch = 1
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            ex_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            ex_inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1],
                      'token_type_ids':  None if args.model_type == 'xlm' else batch[2],
                      'start_positions': batch[3],
                      'end_positions':   batch[4]}
            if args.model_type in ['xlnet', 'xlm']:
                ex_inputs.update({'cls_index': batch[5],
                               'p_mask':       batch[6]})
        
            outputs = ex_model(**ex_inputs)
            ex_loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                ex_loss = ex_loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                ex_loss = ex_loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(ex_loss)
                if amp_handle:
                    amp_handle._clear_cache()
            else:
                ex_loss.backward()

            tr_loss += ex_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                lr_this_step = args.learning_rate * \
                    warmup_linear(global_step/t_total,
                                    args.warmup_proportion)
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # 训练时评估
                # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                #     # Log metrics
                #     if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                #         results = evaluate(args, ex_model, tokenizer)
                #         for key, value in results.items():
                #             tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                #     tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                #     tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                #     logging_loss = tr_loss
        
                if args.do_save and args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = ex_model.module if hasattr(ex_model, 'module') else ex_model  # Take care of distributed/parallel training
                    # 
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    model_to_save.config.to_json_file(os.path.join(output_dir, 'bert_config.json'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args, ex_model, tokenizer, prefix="", ge_model=None):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    torch.cuda.empty_cache()

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        ex_model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            ex_inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
                      }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                ex_inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            outputs = ex_model(**ex_inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            if args.model_type in ['xlnet', 'xlm']:
                # XLNet uses a more complex post-processing procedure
                result = RawResultExtended(unique_id            = unique_id,
                                           start_top_log_probs  = to_list(outputs[0][i]),
                                           start_top_index      = to_list(outputs[1][i]),
                                           end_top_log_probs    = to_list(outputs[2][i]),
                                           end_top_index        = to_list(outputs[3][i]),
                                           cls_logits           = to_list(outputs[4][i]))
            else:
                result = RawResult(unique_id    = unique_id,
                                   start_logits = to_list(outputs[0][i]),
                                   end_logits   = to_list(outputs[1][i]))
            all_results.append(result)

    # Compute predictions
    if args.do_test:
        output_prediction_file = os.path.join(args.output_dir, "predictions_ex_test{}.json".format(prefix))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_ex_test{}.json".format(prefix))
    else:
        output_prediction_file = os.path.join(args.output_dir, "predictions_ex{}.json".format(prefix))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_ex{}.json".format(prefix))
    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        # XLNet uses a more complex post-processing procedure
        write_predictions_extended(examples, features, all_results, args.n_best_size,
                        args.max_answer_length_extract, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.ex_predict_file,
                        ex_model.config.start_n_top, ex_model.config.end_n_top,
                        args.version_2_with_negative, tokenizer, args.verbose_logging)
    else:
        write_predictions(examples, features, all_results, args.n_best_size,
                        args.max_answer_length_extract, args.do_lower_case, output_prediction_file,
                        output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                        args.version_2_with_negative, args.null_score_diff_threshold)
    
    # Evaluate with the official TweetQA script
    results = evaluate_on_tweetqa(gold_file=args.ex_predict_file, 
                                    pred_file=output_prediction_file) 
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.ex_predict_file if evaluate else args.ex_train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.bert_model.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        # 仅训练时去噪
        do_denoising = not evaluate and args.do_denoising
        examples = read_tweet_examples(input_file=input_file,
                                       is_training=not evaluate,
                                       is_pre_process=args.do_pre_process,
                                       do_denoising=do_denoising,
                                       version_2_with_negative=args.version_2_with_negative)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    if output_examples:
        return dataset, examples, features
    return dataset


def main():    
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    
    # bert
    parser.add_argument("--ex_train_file", default=None, type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--ex_predict_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    # unilm
    parser.add_argument("--ge_data_dir",
                        default=None,
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--src_file", default=None, type=str,
                        help="The input data file name.")
    parser.add_argument("--tgt_file", default=None, type=str,
                        help="The output data file name.")
    # ?
    parser.add_argument("--optim_recover_path",
                        default=None,
                        type=str,
                        help="The file of pretraining optimizer.")

    # Other parameters
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        # required=True,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--model_type", default='bert', type=str,
                        help="Model type. Defualt bert.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cuda_visible_devices", default=None, type=str,
                        help="os.environ['CUDA_VISIBLE_DEVICES']")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    # bert
    parser.add_argument("--do_pre_process", action='store_true',
                        help="Whether to pre_process the data.")
    parser.add_argument("--do_denoising", action='store_true',
                        help="Whether to denoising the data.")
    parser.add_argument("--do_save", action='store_true',
                        help="Whether to save ch.")
    
    # Test测试
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run testing.")

    parser.add_argument('--version_2_with_negative', action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="BERT+UNILM " 
                             "The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=64, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length_extract", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    # unilm
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for attention probabilities.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',
                        help="Whether to use 32-bit float precision instead of 16-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--max_len_a', type=int, default=0,
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_b', type=int, default=0,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--trunc_seg', default='',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.7, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument("--mask_prob_eos", default=0, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=24,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Number of workers for the data loader.")

    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument('--skipgram_prb', type=float, default=0.0,
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1,
                        help='the max size of ngram mask')
    parser.add_argument('--mask_whole_word', action='store_true',
                        help="Whether masking a whole word.")
    parser.add_argument('--do_l2r_training', action='store_true',
                        help="Whether to do left to right training")
    parser.add_argument('--has_sentence_oracle', action='store_true',
                        help="Whether to have sentence level oracle for training. "
                             "Only useful for summary generation")

    parser.add_argument('--relax_projection', action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")

    args = parser.parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # assert Path(args.model_recover_path).exists(
    # ), "--model_recover_path doesn't exist"

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl')
    args.device = device
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    # Set seed
    set_seed(args)
    
    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)
    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings
    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
    if args.local_rank == 0:
        dist.barrier()

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    
    # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    cls_num_labels = 2
    type_vocab_size = 6 + \
        (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    num_sentlvl_labels = 2 if args.has_sentence_oracle else 0
    relax_projection = 4 if args.relax_projection else 0

    # t_total = int(len(train_dataloader) * args.num_train_epochs /
                # args.gradient_accumulation_steps)

    # Prepare model
    if args.do_train:
        if (recover_step is None) and (args.model_recover_path is None):
            # if _state_dict == {}, the parameters are randomly initialized
            # if _state_dict == None, the parameters are initialized with bert-init
            _state_dict = {} if args.from_scratch else None
            ex_model = BertForQuestionAnswering.from_pretrained(args.bert_model, state_dict=_state_dict)
            global_step = 0
        else:
            if recover_step:
                logger.info("***** Recover model: %d *****", recover_step)
                model_recover = torch.load(os.path.join(
                    args.output_dir, "model.{0}.bin".format(recover_step)), map_location='cpu')
                # recover_step == number of epochs
                # global_step = math.floor(
                    # recover_step * t_total / args.num_train_epochs)
            # 执行
            elif args.model_recover_path:
                logger.info("***** Recover model: %s *****",
                            args.model_recover_path)
                model_recover = torch.load(
                    args.model_recover_path, map_location='cpu')
                global_step = 0

            ex_model = BertForQuestionAnswering.from_pretrained(args.bert_model, state_dict=model_recover)

        ex_model.to(args.device)

    # Training
    if args.do_train:
        print("Loading Train Dataset", args.ex_train_file)
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)

        global_step, tr_loss = train(args, train_dataset, ex_model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving final model to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer
        # They can then be reloaded using `from_pretrained()`
        model_to_save = ex_model.module if hasattr(ex_model, 'module') else ex_model  # Take care of distributed/parallel training
        # 保存model    
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))
        # 保存config
        model_to_save.config.to_json_file(os.path.join(args.output_dir, 'bert_config.json'))
        # 保存vocab.txt
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        ex_model =BertForQuestionAnswering.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        ex_model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + "pytorch_model.bin", recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            ex_model = BertForQuestionAnswering.from_pretrained(checkpoint, type_vocab_size=type_vocab_size)
            ex_model.to(args.device)

            # Evaluate
            result = evaluate(args, ex_model, tokenizer, prefix=global_step)
            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in result.items())
            results.update(result)
    
    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()
