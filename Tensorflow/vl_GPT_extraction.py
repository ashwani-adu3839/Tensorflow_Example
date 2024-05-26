import pandas as pd

from vl_utility import *
from vl_ner_model_def import Build_model, NER
import transformers
import numpy as np
from collections import Counter
from vl_model_args import ModelArguments
import torch
from vl_ner_label_list import label_dict

import numpy as np
############### #####################################################################

device = ModelArguments.device
tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=ModelArguments.tokenizer_path)
####################################################################################

word_emb_obj = word_emb(tokenizer)
config = transformers.BertConfig.from_pretrained("bert-base-uncased", cache_dir=ModelArguments.tokenizer_path)
BERT = transformers.BertModel(config)
Bert_out = BERT(torch.ones((1, 512), dtype=torch.int64))

ner=NER(BERT, ModelArguments.dense_layer, ModelArguments.drop_rate,ModelArguments.num_class)

ner = ner.to(device)
##################################################################################################


##################################################################################################
###################################################################################################

def create_features(height, data_frame):
    ids_matrix = np.zeros((height), dtype=int)
    token_type_ids_matrix = np.zeros((height), dtype=int)
    attention_matrix = np.zeros((height), dtype=int)

    l_matrix = np.ones((height), dtype=float) * (ModelArguments.pad_token)

    ids_matrix[0] = tokenizer.cls_token_id
    l_matrix[0] = ModelArguments.pad_token  # label_dict['O']
    token_type_ids_matrix[0] = 0
    attention_matrix[0] = 1
    ind = 1
    for i in range(data_frame.shape[0]):
        res_list = (data_frame['ids'][i])
        ids_matrix[ind] = res_list
        token_type_ids_matrix[ind] = data_frame['token_type_ids'][i]
        attention_matrix[ind] = data_frame['attention_mask'][i]
        l_matrix[ind] = 0  # int(data_frame['labels'][i])
        ind = ind + 1
    ids_matrix[ind] = tokenizer.sep_token_id
    l_matrix[ind] = ModelArguments.pad_token  # label_dict['O']
    token_type_ids_matrix[ind] = 0
    attention_matrix[ind] = 1
    return ids_matrix, token_type_ids_matrix, attention_matrix, l_matrix


class classify:

    def __init__(self):
        self.model = Build_model(ner, num_class = ModelArguments.num_class)
        self.model.load_weights(ModelArguments.fine_tuned_weights)

    def update_the_data_frame(self, data_frame):
        lis = []
        for idx, mapping in enumerate(data_frame["offset"]):
            if mapping[0] != 0 and mapping[0] != data_frame['offset'][idx - 1][1]:
                pass
            else:
                if idx >= 1:
                    lis.append(idx)
        words_all = [[]]
        label_all = [[]]
        confidence_all = [[]]
        for ind in data_frame.index:
            tokens = data_frame['tokens'][ind]
            confidance = data_frame['confidence'][ind]
            label = data_frame['labels'][ind]
            if ind in lis:
                words_all[-1].append(tokens)
                label_all[-1].append(label)
                confidence_all[-1].append(confidance)
            else:
                words_all.append([])
                label_all.append([])
                confidence_all.append([])
                words_all[-1].append(tokens)
                label_all[-1].append(label)
                confidence_all[-1].append(confidance)
        pre_words = []
        pre_labelx = []
        pre_confx = []
        for i, (x, y, z) in enumerate(zip(words_all, label_all, confidence_all)):
            if i > 0:
                # print(x,y,z)
                mydf = pd.DataFrame(list(zip(x, y, z)), columns=['tokens', 'labels', 'confidence'])
                List_labels = mydf['labels'].to_list()

                unique_labels = set(List_labels) - set('O')
                if len(unique_labels) >= 2:
                    pre_token = mydf['tokens'][0]
                    pre_conf = mydf['confidence'][0]
                    pre_label = mydf['labels'][0]
                    clean_list = []
                    clean_list.append([pre_token, pre_label, pre_conf])
                    for ind in range(1, len(mydf)):
                        tokens = mydf['tokens'][ind]
                        confidance = mydf['confidence'][ind]
                        label = mydf['labels'][ind]
                        if tokens.find('##') != -1 or label == pre_label:
                            pre_token = pre_token + tokens.replace('##', '')
                            pre_conf = 0.5 * pre_conf + 0.5 * confidance
                            clean_list.pop()
                            clean_list.append([pre_token, pre_label, pre_conf])
                        else:
                            clean_list.append([tokens, label, confidance])
                            pre_token = tokens
                            pre_conf = confidance
                            pre_label = label
                    words_all = [x[0] for x in clean_list]
                    label_all = [x[1] for x in clean_list]
                    confidance_all = [x[2] for x in clean_list]
                    pre_words.extend(words_all)
                    pre_labelx.extend(label_all)
                    pre_confx.extend(confidance_all)
                else:
                    count = Counter(y)
                    pre_words.append(''.join(element.replace('##', '') for element in x))
                    max_num = 0
                    labels = set([*count.keys()])
                    # label = 'O'
                    # for x in labels:
                    #     if x!= 'O':
                    #         label = x
                    for x_label, y_count in zip([*count.keys()], [*count.values()]):

                        if y_count == max_num and x_label != 'O':
                            label = x_label
                            max_num = y_count
                        elif y_count > max_num:
                            label = x_label
                            max_num = y_count
                    pre_labelx.append(label)
                    pre_confx.append(sum(z) / len(z))

        df = pd.DataFrame(list(zip(pre_words, pre_labelx, pre_confx)))
        df.columns = ['tokens', 'labels', 'confidence']
        return df


    def proces_input(self, data_frames, feat_hieght=256):
        data_frame_orig = data_frames.copy(deep=True)
        data_frame = dataframe_add_emb_all(data_frames, word_emb_obj)
        index = 0
        no_items = data_frame.shape[0]
        inputs = []
        token_ids = []
        attention = []
        targets = []
        # NameInsuedDesciption = [9]
        while index < no_items:
            df_framex = data_frame[index:index + feat_hieght - 2]
            df_framex.reset_index(inplace=True)
            ids_matrix, token_type_ids_matrix, attention_matrix, label = create_features(feat_hieght, df_framex)
            index = index + feat_hieght - 2
            inputs.append(ids_matrix)
            targets.append(label)
            token_ids.append(token_type_ids_matrix)
            attention.append(attention_matrix)
        input_ids = np.stack(inputs)
        attention_ids = np.stack(attention)
        token_type_ids = np.stack(token_ids)
        
        input_ids = torch.tensor(input_ids)
        attention_ids = torch.tensor(attention_ids)
        token_type_ids = torch.tensor(token_type_ids)
        
        X = {"input_ids":input_ids.to(device),'token_type_ids':token_type_ids.to(device), 'attention_mask':attention_ids.to(device)}
        prediction = self.model.predict(X)[:, 1:-1, :].cpu()
        index_list = []
        probability_list = []
        for pred in prediction:
            for items in pred:
                items = torch.nn.functional.softmax(items)
                m_index = np.argmax(items.numpy())
                indices = (-items.numpy()).argsort()[0:3]
                if indices[0] == 0 and indices[1] == 9 and items[
                    indices[1]].numpy() > 0.01:  # NamedInsuredDescriptionofOperations
                    m_index = indices[1]

                index_list.append(m_index)
                probability_list.append(items[m_index].numpy())
        labels = []
        confidance_scores = []
        label_dict_rev = dict([(value, key) for key, value in label_dict.items()])
        k = 0
        for ind in data_frame.index:
            labels.append(label_dict_rev[index_list[k]])
            confidance_scores.append(probability_list[k])
            k += 1

        data_frame['labels'] = labels
        data_frame['confidence'] = confidance_scores
        # print(data_frame[['labels', 'confidence', 'tokens']])

        data_frame.ids.astype('int64')
        data_frame = self.update_the_data_frame(data_frame)

        return data_frame


classify_request = classify()


def process_input_po(datframe, feat_hieght=256):
    data_frame = classify_request.proces_input(datframe, feat_hieght)

    return data_frame
