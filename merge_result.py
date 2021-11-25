import json
def max_length(*lst):
    return max(*lst, key=lambda v: len(v))

def span_equal(span1, span2):
    for k, v in span1.items():
        v2 = span2[k]
        if v != v2:
            return False
    return True

def spans_merge(spans_list):
    ori_spans = []
    for add_spans in spans_list:
        for add_span in add_spans:
            for ori_span in ori_spans:
                if span_equal(add_span, ori_span):
                    break
            else:
                ori_spans.append(add_span)
    return ori_spans
    

if __name__ == "__main__":
    BiLSTM_path = '/home/aipf/work/WYY-JC/NER-BERT-BiLSTM-CRF-/result/testB_result_BILSTM.json'
    crf_small_path = '/home/aipf/work/WYY-JC/BERT-NER-Pytorch-1123/result/testB_result_crf.json'
    crf_large_path = '/home/aipf/work/WYY-JC/BERT-NER-Pytorch-1123/result/testB_result_crf_large.json'
    drop03_path = '/home/aipf/work/WYY-JC/BERT-NER-Pytorch-1123/result/testB_result_drop03.json'
    input_path = '/home/aipf/work/建行杯数据集/舆情预警/TestB/testB.json'
    output_path = 'result/mergeB.json'
    with open(input_path) as f:
        datas = json.load(f)
    with open(BiLSTM_path) as BiLSTM_f:
        BiLSTM_results = json.load(BiLSTM_f)
    with open(crf_small_path) as small_f:
        small_results = json.load(small_f)
    with open(crf_large_path) as large_f:
        large_results = json.load(large_f)
    with open(drop03_path) as drop03_f:
        drop03_results = json.load(drop03_f)
        
    
    for data, bilstm_result, s_result, l_result, d_result in zip(datas["result"], BiLSTM_results["result"], small_results["result"], large_results["result"], drop03_results["result"]):
        print("=================Four results=================")
        print(bilstm_result["spans"])
        print(s_result["spans"])
        print(l_result["spans"])
        print(d_result["spans"])
        print("=================After merge=================")
        data["spans"] = spans_merge([bilstm_result["spans"], s_result["spans"], l_result["spans"], d_result["spans"]])
        print(data["spans"])
    with open(output_path, "w") as f:
            json.dump(datas, f, ensure_ascii=False)
