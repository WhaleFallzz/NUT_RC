import json

def get_final_result_v1(ex_file, ge_file, qid_file):
    """简单的标签判断   若标签一致，则取ex"""
    with open(ex_file, 'r') as f_obj:
        ex_results = f_obj.readlines()
    with open(ge_file, 'r') as f_obj:
        ge_results = f_obj.readlines()
    with open(qid_file, 'r') as f_obj:
        qids = f_obj.readlines()
    assert len(ex_results) == len(ge_results) == len(qids)
    final_results = []
    for ex_result, ge_result, qid in zip(ex_results, ge_results, qids):
        ex_ans, ex_label = ex_result.strip().split('\t')[:2]
        ge_ans, ge_label = ge_result.strip().split('\t')[:2]
        qid = qid.strip()
        # 0:正例    1:负例
        if ex_label == ge_label:
            final_results.append({
                'qid': qid,
                'Answer': ex_ans
            })
        elif ge_label == '0':
            final_results.append({
                'qid': qid,
                'Answer': ge_ans
            })
        else:
            final_results.append({
                'qid': qid,
                'Answer': ex_ans
            })
    return final_results

def get_final_result_v2(ex_file, ge_file, qid_file):
    """标签一致时，看得分"""
    with open(ex_file, 'r') as f_obj:
        ex_results = f_obj.readlines()
    with open(ge_file, 'r') as f_obj:
        ge_results = f_obj.readlines()
    with open(qid_file, 'r') as f_obj:
        qids = f_obj.readlines()
    # print(len(ex_results), len(ge_results), len(qids))
    assert len(ex_results) == len(ge_results) == len(qids)

    final_results = []
    for ex_result, ge_result, qid in zip(ex_results, ge_results, qids):
        ex_ans, ex_label, ex_score = ex_result.strip().split('\t')
        ge_ans, ge_label, ge_score = ge_result.strip().split('\t')
        ex_score, ge_score = float(ex_score), float(ge_score)
        qid = qid.strip()
        # 0:正例    1:负例
        if ex_label == ge_label == '0':
            if ex_score > ge_score:
                final_results.append({
                    'qid': qid,
                    'Answer': ex_ans
                })
            else:
                final_results.append({
                    'qid': qid,
                    'Answer': ge_ans
                })
            continue
        if ex_label == ge_label == '1':
            if ex_score > ge_score:
                final_results.append({
                    'qid': qid,
                    'Answer': ge_ans
                })
            else:
                final_results.append({
                    'qid': qid,
                    'Answer': ex_ans
                })
            continue
        if ex_label == '0':
            final_results.append({
                'qid': qid,
                'Answer': ex_ans
            })
            continue
        if ge_label == '0':
            final_results.append({
                'qid': qid,
                'Answer': ge_ans
            })
            continue
    return final_results


if __name__ == '__main__':
    ex_cls_result_file = '/home/rthuang/debug/cls_large_wwm/Ext_RC_dev_cls_results.txt'
    ge_cls_result_file = '/home/rthuang/debug/cls_large_wwm/Gen_RC_dev_cls_results.txt'
    qid_file = '/home/rthuang/Experimental_result_TweetQA/cls_data/generation_ans/qid.txt'
    # 标签判断：v1 or v2
    final_results = get_final_result_v2(ex_cls_result_file, ge_cls_result_file, qid_file)

    same_results = json.load(open('/home/rthuang/Experimental_result_TweetQA/cls_data/same.json'))
    final_results.extend(same_results)
    print(len(final_results))
    
    # final_predictions:v1
    # with open('/home/rthuang/debug/cls/final_predictions.json', 'w') as writer:
    with open('/home/rthuang/NUT_RC/final_predictions.json', 'w') as writer:
        writer.write(json.dumps(final_results, indent=4) + '\n')
    
