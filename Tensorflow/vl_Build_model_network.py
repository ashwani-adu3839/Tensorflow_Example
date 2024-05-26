import torch.nn as nn 
import torch
import time
from tqdm import tqdm
import os
# from vl_utility import *
# from transformers import BertTokenizerFast
from vl_model_args import ModelArguments


###########################################################

device = ModelArguments.device



###########################################################
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
#         cache_dir=ModelArguments.tokenizer_path)
###########################################################
Feature = Feature_map(tokenizer, ModelArguments.pad_token)

class Build_model(nn.Module):    
    def __init__(self, model, num_class):
            super(Build_model, self).__init__()
            self.model = model
            self.num_class = num_class

    def loss_object(self, real, pred):

            loss = self.loss_fct(pred.view(-1, self.num_class), real.view(-1))
            return loss 
    def accuracy_function(self, label, logits):
        
        
        masked_active_acc_one = torch.ne(label.view(-1,),ModelArguments.pad_token)
        masked_labels = torch.masked_select(label.view(-1), masked_active_acc_one)
        masked_active_acc_one = torch.unsqueeze(masked_active_acc_one, 1)
        masked_active_acc = masked_active_acc_one.repeat(1,logits.size(dim=2))
        logits = logits.view(-1,logits.size(dim=2))
        new_logits = logits[masked_active_acc]
        new_logits = new_logits.view(-1, self.num_class)
        masked_pred = torch.argmax(new_logits, dim=-1)
        result = (masked_pred == masked_labels).float().mean()
        return result
        
 
    def compiler(self, optimizer, loss_fct):
        
            self.optimizer = optimizer
            self.loss_fct = loss_fct

    def fit(self,training_data_list, validation_data_list,epochs,BS):
        history = {
        "epoch": [],
        "loss": [],
        "Accuracy" :[],
        "val_loss" :[],
        "val_Accuracy" :[]
        }
        print("Training Started ........ ")
        for epoch in range(epochs):
            start = time.time()
                    
            total_loss = 0
            val_loss = 0
            total_accuracy = 0
            val_accuracy = 0            
            print(f"Epoch : {epoch +1}\n")

            training_generator = data_generator(training_data_list, BS)
            iterator = iter(training_generator)
            batch =0 
            try:  
                start_iter_train = time.time()
                while True:
                    inp, label = next(iterator)
                    input_ids=inp['input_ids']
                    token_type_ids=inp['token_type_ids']
                    attention_mask=inp['attention_mask']
                    self.model.train()  # make sure we are in .train() mode                
                    prediction = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)

                    loss = self.loss_object(label, prediction)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    acc = self.accuracy_function(label, prediction)
                    if (batch+1) % 5 == 0:    
                        print(f"Train iter : {batch+1} |  accuracy : {acc.detach().cpu().numpy():.4f} | loss : {loss.detach().cpu().numpy():.4f} | time : {time.time() - start_iter_train:.2f} secs")  
                    total_accuracy = total_accuracy+acc.detach().cpu().numpy() 
                    total_loss = total_loss + loss.detach().cpu().numpy()
                    batch=batch+1
                    del inp
                    del label
                    del loss
                    del prediction
                    del acc
            except StopIteration:
                    pass
            print(f'\nTotal Loss: {total_loss/(batch):.4f} | Accuracy : {total_accuracy/(batch):.4f}\n')
            validation_generator = data_generator(validation_data_list, BS)
            val_iterator = iter(validation_generator)
            with torch.no_grad():
                    batc =0 
                    try:
                        start_iter_val = time.time()

                        while True:
                            inp, label = next(val_iterator)
                            input_ids=inp['input_ids']
                            token_type_ids=inp['token_type_ids']
                            attention_mask=inp['attention_mask']
                            self.model.eval()  # make sure we are in .eval() mode
                            prediction = self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
                            loss = self.loss_object(label, prediction)
                            acc = self.accuracy_function(label, prediction)
                            if (batc+1) % 5 == 0:    
                                print(f"Validation iter : {batc+1}  |  accuracy : {acc.detach().cpu().numpy():.4f} | loss : {loss.detach().cpu().numpy():.4f} | time : {time.time() - start_iter_val:.2f} secs")  
                            val_accuracy = val_accuracy+acc.detach().cpu().numpy() 
                            val_loss = val_loss + loss.detach().cpu().numpy()
                            batc=batc+1
                            del inp
                            del label
                            del loss
                            del prediction
                            del acc
                    except StopIteration:
                          pass

            print(f'\nTotal Validation Loss : {val_loss/(batc):.4f} | Validation Accuracy : {val_accuracy/(batc):.4f}') 
            print(f'\nTime taken for 1 epoch : {time.time() - start:.2f} secs\n')  
        return history 
    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
    def predict(self, inp):
        with torch.no_grad():
                    input_ids=inp['input_ids']
                    token_type_ids=inp['token_type_ids']
                    attention_mask=inp['attention_mask']
                    self.model.eval()
                    output =  self.model(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        return output
                            
