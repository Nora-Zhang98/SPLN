from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import load_word_vectors
import torch
# 按首字母排序
pred_classes = ["__background__", "above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between",            # 0-10
             "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has",    # 11-20
             "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of",      # 21-30
             "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on",       # 31-40
             "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]         # 41-50
# 用这个 按样本数目排序
# pred_classes = ['__background__', 'on', 'has', 'wearing', 'of', 'in', 'near', 'behind', 'with', 'holding', 'above', # 0-10
#                 'sitting on', 'wears', 'under', 'riding', 'in front of', 'standing on', 'at', 'carrying', 'attached to', 'walking on', # 11-20
#                 'over', 'for', 'looking at', 'watching', 'hanging from', 'laying on', 'eating', 'and', 'belonging to', 'parked on', # 21-30
#                 'using', 'covering', 'between', 'along', 'covered in', 'part of', 'lying on', 'on back of', 'to', 'walking in', # 31-40
#                 'mounted on', 'across', 'against', 'from', 'growing on', 'painted on', 'playing', 'made of', 'says', 'flying in'] # 41-50

# GQA 按样本数目排序
# pred_classes = ['__background__', 'on', 'wearing', 'of', 'near', 'in', 'behind', 'in front of', 'holding', 'next to', 'above', # 0-10
#                 'on top of', 'below', 'by', 'with', 'sitting on', 'on the side of', 'under', 'riding', 'standing on', 'beside', # 11-20
#                 'carrying', 'walking on', 'standing in', 'lying on', 'eating', 'covered by', 'looking at', 'hanging on', 'at', 'covering', # 21-30
#                 'on the front of', 'around', 'sitting in', 'parked on', 'watching', 'flying in', 'hanging from', 'using', 'sitting at', 'covered in', # 31-40
#                 'crossing', 'standing next to', 'playing with', 'walking in', 'on the back of', 'reflected in', 'flying', 'touching', 'surrounded by', 'covered with', # 41-50
#                 'standing by', 'driving on', 'leaning on', 'lying in', 'swinging', 'full of', 'taking on', 'walking down', 'throwing', 'surrounding', # 51-60
#                 'standing near', 'standing behind', 'hitting', 'printed on', 'filled with', 'catching', 'growing on', 'grazing on', 'mounted on', 'facing',  # 61-70
#                 'leaning against', 'cutting', 'growing in', 'floating in', 'driving', 'beneath', 'contain', 'resting on', 'worn on', 'walking with',  # 71-80
#                 'driving down', 'on the bottom of', 'playing on', 'playing in', 'feeding', 'standing in front of', 'waiting for', 'running on', 'close to', 'sitting next to',  # 81-90
#                 'swimming in', 'talking to', 'grazing in', 'pulling', 'pulled by', 'reaching for', 'attached to', 'skiing on', 'parked along', 'hang on'] # 91-100

# 原来是__background__，对应的embedding是normal(0,1)的结果，现在把它换成background，对应这个单词的word embedding
root, wv_type, wv_dim = '.', 'glove.6B', 300
wv_dict, wv_arr, wv_size = load_word_vectors(root , wv_type, wv_dim)

vectors = torch.Tensor(len(pred_classes), wv_dim)
vectors.normal_(0, 1)

fusion_rate = 0.7 # 0.7 # 三个词重点在中间，两个词重点在前

for i, token in enumerate(pred_classes):
    cur_list = token.split(' ') # 分词后是一个list
    cur_vec = torch.Tensor(wv_dim)
    cur_vec.normal_(0, 1)

    # 为每个分词赋权重
    if len(cur_list) == 1:
        rate_list = [1.]
    elif len(cur_list) == 2:
        rate_list = [fusion_rate, 1-fusion_rate]
    elif len(cur_list) == 3:
        rate_list = [(1-fusion_rate)/2. ,fusion_rate, (1-fusion_rate)/2.]
    elif len(cur_list) == 4:
        rate_list = [0.25, 0.25, 0.25, 0.25]
    else:
        print('wrong length!')

    for id, word in enumerate(cur_list):
        wv_index = wv_dict.get(word, None)

        if wv_index is not None:
            cur_vec += rate_list[id] * wv_arr[wv_index]
            print(id, word)
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            # print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))

    vectors[i] = cur_vec

#torch.save(vectors, 'gqa_pred_embedding.pt')
torch.save(vectors, 'pred_embedding.pt')
print('ready saving pred_embedding')
