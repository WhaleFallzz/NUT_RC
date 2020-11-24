import os
import json

def convert_ans(pred_file, qid_file, output_path):
    with open(pred_file, 'r') as f_pred, open(qid_file, 'r') as f_qid:
        preds = f_pred.readlines()
        qids = f_qid.readlines()
        print(len(preds), len(qids))
        predictions = []
        for ans, qid in zip(preds, qids):
            predictions.append({
                'qid': qid.strip(),
                'Answer': ans.strip(),
            })
        # print(predictions)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = os.path.join(output_path, 'predictions_ge.json')
    with open(output_file, 'w') as writer:
        writer.write(json.dumps(predictions, indent=4) + '\n')


if __name__ == '__main__':
    pred_file = '/home/rthuang/TweetQA/TweetQA_data/gqa_unilm/dev/pred_ans_multitask.txt'
    qid_file = '/home/rthuang/TweetQA/TweetQA_data/gqa_unilm/dev/dev.qid.txt'
    output_path = '/home/rthuang/debug/multi_task'
    convert_ans(pred_file, qid_file, output_path)

