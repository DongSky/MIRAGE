
import os
from tqdm import tqdm
import pandas as pd
import re
import ast
import pandas as pd

from Levenshtein import distance
from copy import deepcopy
from retry import retry
from rich.console import Console
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,f1_score

from ..utils.file import import_file
from ..LLM.LLM_state import get_eval_api_client

from .eval_prompts import *


console = Console()

class benchmark_eval():
    
    def __init__(self,
                 data : pd.DataFrame,
                 path : str) -> None:
        
        self.df = data
        self.path = path
        self.response = self.find_response()
        self.answer = self.find_answer()
        self.response_answer_pair = list(zip(self.response,self.answer))
        self.model = get_eval_api_client()
        
        self.type = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        return self.response_answer_pair[index]
    @property
    def basename(self):
        return '.'.join(os.path.basename(self.path).split('.')[0:-1])
    
    @property
    def dirname(self):
        return os.path.dirname(self.path)+'/eval_results'
    @property
    def result_path(self):
        basename = self.basename +'_eval.tsv'
        results_path = os.path.join(self.dirname,basename)
        return results_path
    
    @property
    def score_name(self):
        score_df = self.basename + '_eval.xlsx'
        results_path = os.path.join(self.dirname,score_df)
        return results_path
        
    
    def cut(self,index):
        return self.__class__(self.df.iloc[index:].reset_index(drop=True),self.path)
    
    def itterows(self):
        length = len(self)
        for i in range(0,length):
            yield i,self[i]
            
    def get(self,index):
        return self.df.iloc[[index],:]
    
    def get_prompt(self,index):
        return self.df['prompt'][index]
            
    def find_response(self):
        pass
    
    def find_answer(self):
        pass
    
    def find_choice(self,
                    under_eval_bench:'benchmark_eval',
                    index:int) -> str:
        pass
    
    def score(self):
        pass
    
    def eval(self):
        self.Eval_main()
        self.score()
    
    def checking_exiting_file(self):
        if os.path.exists(self.result_path):
            df = import_file(self.result_path)
            console.print(f"Loaded {len(df)} results from {self.result_path}",style="bold green")
            return len(df)
        else:
            console.print(f"Created {self.result_path}",style="bold green")
            return 0
    
    def Eval_main(self):
        # type of evaluation: YORN, MCQ, etc
        
        existing_length = self.checking_exiting_file()
        if existing_length == len(self):
            console.print(f'All response evaluations have been saved. No further evaluations needed.',style='bold green')
        
        else:
            
            eval_benchmark = self.cut(existing_length)
            
            for index,row in tqdm(eval_benchmark.itterows(),total=len(eval_benchmark)):
                response,answer = row
                
                if self.type == 'YORN':
                    LLM_parse,sign = self.YORN(response,answer)
                elif self.type == 'MCQ':
                    LLM_parse,sign = self.MCQ(response,answer,eval_benchmark,index)
                elif self.type == 'MIX':
                    LLM_parse,sign = self.MIX(response,answer,eval_benchmark,index)
                elif self.type == 'Free_Form':
                    LLM_parse,sign = self.Free_Form(response,answer,eval_benchmark,index)
                elif self.type == 'VQA':
                    LLM_parse,sign = self.VQA(response,answer)
                else:
                    raise ValueError('Invalid evaluation type')
                
                answer_df= deepcopy(eval_benchmark.get(index))
                answer_df.loc[:,'LLM_parse'] = LLM_parse
                answer_df.loc[:,'sign'] = sign
                
                self.saved(answer_df)
                
            results = import_file(self.result_path)
            assert len(results) == len(self.df)
            print(f'All response evaluations saved.')
            
    def saved(self,answer_df):
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
        if not os.path.exists(self.result_path):
            answer_df.to_csv(self.result_path,sep='\t')
        else:
            answer_df.to_csv(self.result_path,sep='\t',mode='a',header=False)
        console.print(f"Stored 1 result in {self.result_path}",style="bold green")
        
        
        
    
    def YORN(self,response,answer):
        
        prompt = "You will be provided with an response to a yes/no question. Your task is to interpret the response and output either 'yes' or 'no,' matching the meaning of the response. If the response shows high confidence in the answer, output 'yes.' If the response shows some confidence in the answer, output 'yes.' If the response shows low confidence in answer, like 'likely,' 'probably,' 'maybe,' 'possibly,' etc., output 'yes.' If the response shows high confidence in the answer, output 'no.' If the response shows some confidence in the answer, output 'no.' If the response shows some confidence in the answer, like 'likely,' 'probably,' conclude in corresponding yes or no. only if the response is not present or very ambiguous, output 'unclear'."
        
        response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    
    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the letter (A, B, C, D, etc.) that corresponds to the option chosen by the user. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        choice = self.find_choice(under_eval_bench,index)
        
        response = f"Choises:{choice}\nResponse:{response}"
        # response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    def number(self,response,answer):
        
        prompt = """You will be provided with an response to a question that requires a numerical answer. It can be a int, float or a list. Your task is to indentify and parse the final numerical answer. You answer should only contain the answer of number(or number list) with no punctuations. Make sure your answer is in the correct format and is as accurate as possible. if you can not find the number or it is unclear, output 'unclear'
        Please read the following example. Then extract the answer from the model response and type it at the end of the prompt. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer.
        Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
        Question: Which number is missing?

        Model response: The number missing in the sequence is 14.

        Extracted answer: 14

        Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
        Question: What is the fraction of females facing the camera?

        Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.        

        Extracted answer: 0.6

        Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
        Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

        Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

        Extracted answer: 1.45

        Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.      
        Question: Between which two years does the line  graph saw its maximum peak?

        Model response: The line graph saw its maximum peak between 2007 and 2008.

        Extracted answer: [2007, 2008]
        """
        #print(response, type(response), flush=True)
        try:
            response = 'response:'+response+'Extracted answer:'
        except:
            response = "none"
            response = 'response:'+response+'Extracted answer:'
        
        
        LLM_parse = self.LLM(response,prompt)
        print(LLM_parse, flush=True)
        
        
        sign = self.extract_answer(LLM_parse,answer,num=True)
        
        return LLM_parse,sign
    
    def extract_answer(self,LLM_parse,answer,num=False):
        
        if num:
            try:
                answer = int(answer)
                num_type = 'int'
            except ValueError:
                try:
                    answer = float(answer)
                    num_type = 'float'
                except ValueError:
                    try:
                        answer = ast.literal_eval(answer)
                        num_type = 'list'
                    except:
                        num_type = 'str'
                    
            if num_type == 'int':
                try:
                    LLM_parse = int(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'float':
                try:
                    LLM_parse = float(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'list':
                try:
                    LLM_parse = ast.literal_eval(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except:
                    return 0
            else:
                return 0
            
        else:
            
            LLM_parse = LLM_parse.lower()
            answer = answer.lower()
            
            char='.()[],:;!*#{}'
            for c in char:
                LLM_parse=LLM_parse.replace(c,' ')
                
            LLM_parse = re.sub(r'\s+', ' ', LLM_parse).strip()
                
            LLM_parse_list = LLM_parse.split(' ')
            
            
            if answer in LLM_parse_list:
                
                return 1
            
            else:
                return 0
    
    def VQA(self,response,answer):
            
            prompt = """You will be provided a real answer and a user's response to a visual question. Your task is to determine whether the user's response is correct or incorrect. If the user's response is correct, output 'correct'. If the user's response is incorrect, output 'incorrect'. If the user's response is unclear or ambiguous, output 'unclear'."""
            
            response = f"Real Answer:{answer},Response:{response}"
            
            def is_response_unclear(response):
                # define the maximum number of times a word can be repeated in the response
                max_word_repetition = 10
                
                # split the response into words
                words = response.split(' ')
                
                # count the number of times each word appears in the response
                word_count = {}
                for word in words:
                    if word in word_count:
                        word_count[word] += 1
                    else:
                        word_count[word] = 1
                
                # check if any word is repeated more than the maximum number of times
                for count in word_count.values():
                    if count > max_word_repetition:
                        return "unclear"
                
                # check if the response contains any words that are not alphanumeric
                return response
            
            if is_response_unclear(response) == 'unclear':
                LLM_parse = 'unclear'
            else:
                LLM_parse = self.LLM(response,prompt)
            
            
            
            print(LLM_parse)
            
            sign = 1 if LLM_parse == 'correct' else 0
                
            return LLM_parse,sign
    
    
    @retry(tries=3,delay=2)
    def LLM(self,response,prompt=None):
        if prompt:
        
            data = {'query':response,'system_prompt':prompt}
        
        else:
            data = {'query':response}
        
        output = self.model.request(data)
        
        return output
        
    @classmethod
    def from_tsv(cls,path):
        data = import_file(path)
        return cls(data,path)
    
class Hallucination_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        # Initialize the type of evaluation, support for 'YORN' and 'MCQ' right now.
        self.type = 'YORN'
        
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        # you must rewrite this function to find the answer in your data and return it as a list
        answer_list = self.df['gt_answer'].tolist()
        answer = ['yes' if ans == 1 else 'no' for ans in answer_list]
        self.df['answer'] = answer
        return answer
    
    def score(self):
        # you must define how to score the evaluation results. This is a example for hallucination benchmark.
        try:
            df = import_file(self.result_path)
        except:
            raise ValueError('No evaluation results found.')
        # calculate the related score
        score_dict = {}
        # correct is the number of correct responses
        score_dict['correct'] = df['sign'].sum()
        # total is the total number of responses
        score_dict['total'] = len(df)
        # accuracy is the ratio of correct responses to total responses
        score_dict['accuracy'] = score_dict['correct']/score_dict['total']
        # # if all the answer of a fig is correct, then return is correct (an another statistics)
        # fig_accuracy = df.groupby(['category', "subcategory", "set_id", "figure_id"]).apply(lambda x: 1 if x['sign'].sum()==len(x) else 0)
        # fig_accuracy = fig_accuracy.sum()/len(fig_accuracy)
        # score_dict['fig_accu'] = fig_accuracy
        # save the score to a excel file
        
        #fine-grained evaluation
        category = df.groupby('subcategory').apply(lambda x: x['sign'].sum()/len(x))
        category = category.to_dict()
        score_dict.update(category)
        y_true , y_pred = self.label_parse(df)
        precision = precision_score(y_true,y_pred)
        recall = recall_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)
        yes_rate = y_pred.count(1)/len(y_pred)
        score_dict['precision'] = precision
        score_dict['recall'] = recall
        score_dict['f1'] = f1
        score_dict['yes_rate'] = yes_rate
        
        
        score_df = pd.DataFrame([score_dict])
        score_df.to_excel(self.score_name)
        
        #print the score
        for key,value in score_dict.items():
            print(f'{key}:{value}')
        
    def label_parse(self,df):
        y_pred = df['LLM_parse'].tolist()
        y_true = df['gt_answer'].tolist()
        y_pred = [1 if 'yes' in pred.lower() else 0 for pred in y_pred]
        return y_true,y_pred
        
        
    
    
        


class MME_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        # Initialize the type of evaluation, support for 'YORN' and 'MCQ' right now.
        self.type = 'YORN'
        
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        # you must rewrite this function to find the answer in your data and return it as a list
        # answer_list = self.df['ground_truth'].tolist()
        # answer = ['yes' if ans == 1 else 'no' for ans in answer_list]
        # self.df['answer'] = answer
        return self.df['ground_truth'].tolist()
    
    def score(self):
        eval_type_dict = {
            "Perception": ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"],
            "Cognition": ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning"]
        }

        eval_df = import_file(self.result_path)
        MME_score = dict()

        # you must define how to score the evaluation results. This is a example for hallucination benchmark.
        for eval_type, task_name_list in eval_type_dict.items():
            print("===========", eval_type, "===========")
           
            scores = 0
            task_score_dict = dict()

            for task_name in task_name_list:
                task_df = eval_df[eval_df['subset'] == task_name]

                task_other_ans_num = 0
                task_score = 0
                acc_plus_correct_num = 0
                gts = []
                preds = []

                # response_column = [col for col in self.task_df.columns if 'response' in col and 'intermediate' not in col]
                gts = task_df['ground_truth'].squeeze().tolist()
                gts = list(map(lambda x: x.lower(), gts))
                preds = task_df['LLM_parse'].tolist()
                preds = list(map(lambda x: x.lower(), preds))

                metric_dict = self.compute_metric(gts, preds)
                acc_plus = self.get_accplus(task_df) / len(task_df)
                metric_dict["acc_plus"] = acc_plus
                
                for k, v in metric_dict.items():
                    if k in ["acc", "acc_plus"]:
                        task_score += v*100
                
                task_score_dict[task_name] = task_score
                
                scores += task_score

            print("total score:", scores, "\n")
            for task_name, score in task_score_dict.items():
                print("\t", task_name, " score:", score)
            print("\n")
            
            MME_score[f"{eval_type}_totol"] = scores
            MME_score.update(task_score_dict)
            
        MME_score_df = pd.DataFrame([MME_score])
        MME_score_df.to_excel(self.score_name, index=False)
            
    
    
    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)

        label_map = {
            "yes": 1,
            "no": 0,
            "unclear": -1,
        }
        
        gts = [label_map[x] for x in gts]
        preds = [label_map[x.strip('.')] for x in preds]

        acc = accuracy_score(gts, preds) 

        clean_gts = []
        clean_preds = []
        other_num = 0 
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)
        

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1,0])
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        metric_dict = dict()
        metric_dict = {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }

        return metric_dict
    
    def get_accplus(self, task_df):
        temp_df = task_df.groupby('picture_name')['sign'].sum()

        acc_plus = (temp_df == 2).sum()

        return acc_plus


            
class Pope_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        self.type = 'YORN'
        
    def find_response(self):
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = self.df['ground_truth'].tolist()
        answer_list = [x.lower().strip() for x in answer_list]
        return answer_list
    
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        
        # Initialize the score dictionaries
        pope_score = {'total':len(eval_results)}
        pope_score['correct'] = eval_results['sign'].sum()
        pope_score['accuracy'] = pope_score['correct']/pope_score['total']
        
        #fine-grained evaluation
        category = eval_results.groupby('subset').apply(lambda x: x['sign'].sum()/len(x))
        category = category.to_dict()
        pope_score.update(category)
        
        y_true, y_pred = self.lable_parse(eval_results)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        yes_rate = y_pred.count(1)/len(y_pred)
        pope_score['f1'] = f1
        pope_score['precision'] = precision
        pope_score['recall'] = recall
        pope_score['yes_rate'] = yes_rate
        
        
        # Print and save the scores
        pope_score_df = pd.DataFrame([pope_score])
        pope_score_df.to_excel(self.score_name, index=False)
        
        for key, value in pope_score.items():
            print(f'{key}: {value:.4f}'+ '\n')
        
    def lable_parse(self,eval_results):
        y_true = eval_results['ground_truth'].tolist()
        y_pred = eval_results['LLM_parse'].tolist()
        y_true = [1 if 'yes' in x.lower() else 0 for x in y_true]
        y_pred = [1 if 'yes' in x.lower() else 0 for x in y_pred]
        return y_true, y_pred
        


class Seed_bench_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        self.type = 'MCQ'
        
    def find_response(self):
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = self.df['answer'].tolist()
        answer_list = [f'({ans.lower()})' for ans in answer_list]
        return answer_list
    
    def find_choice(self,index):
        return self.df.loc[index,'choices']
    
    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the upper letter with parentheses (A), (B),(C),(D) etc. that corresponds to the option chosen by the response. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        choice = under_eval_bench.find_choice(index)
        
        response = f"Choises:{choice}\nResponse:{response}"
        # response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    def extract_answer(self, LLM_parse, answer):
        if answer in LLM_parse.lower():
            return 1
        else:
            return 0
    
    def score(self):
            
        # Load the evaluation
        eval_results = import_file(self.result_path)
        
        
        # Initialize the score dictionaries
        seed_bench_score = {'total':len(eval_results)}
        seed_bench_score['correct'] = eval_results['sign'].sum()
        seed_bench_score['accuracy'] = seed_bench_score['correct']/seed_bench_score['total']
        
        #fine-grained evaluation
        category = eval_results.groupby('question_type').apply(lambda x: x['sign'].sum()/len(x))
        category = category.to_dict()
        seed_bench_score.update(category)
        
        # Print and save the scores
        seed_bench_score_df = pd.DataFrame([seed_bench_score])
        seed_bench_score_df.to_excel(self.score_name, index=False)
        
        for key, value in seed_bench_score.items():
            print(f'{key}: {value:.4f}'+ '\n')
            
class mathvista_eval(benchmark_eval):
    
    def __init__(self, data, path):
        super().__init__(data, path)
        self.type = 'MIX'
    
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = []
        
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        
        for i in range(len(self)):
            if self.df['question_type'][i] == 'multi_choice':
                answer = self.df['answer'][i]
                distance_list = [distance(answer,choice) for choice in ast.literal_eval(self.df['choices'][i])]
                distance_index = distance_list.index(min(distance_list))
                answer_list.append(options[distance_index])
            else:
                answer_list.append(self.df['answer'][i])
                
        assert len(answer_list) == len(self)
        return answer_list
    
    def find_choice(self,under_eval_bench,index):
        choices = ast.literal_eval(under_eval_bench.df['choices'][index])
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        choice = ''
        for i in range(len(choices)):
            choice += options[i]+'. '+choices[i]+'\n'
        return choice
            
    
    
    def MIX(self, response, answer, under_eval_bench, index):
        
        if under_eval_bench.df['question_type'][index] == 'multi_choice':
            
            LLM_parse,sign = self.MCQ(response,answer,under_eval_bench,index)
            
        else:
            LLM_parse,sign = self.number(response,answer)
            
        return LLM_parse,sign
        
                
    
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        
        # Initialize the score dictionaries
        mathvista_score = {'total':len(eval_results)}
        mathvista_score['correct'] = eval_results['sign'].sum()
        mathvista_score['accuracy'] = mathvista_score['correct']/mathvista_score['total']
        
        skills = []
        
        for i in range(len(eval_results)):
            mata_data = ast.literal_eval(eval_results['metadata'][i])
            # print(mata_data)
            skill = mata_data['skills']
            skills.extend(skill)
        skills = list(set(skills))
        
        for skill in skills:
            df =eval_results[eval_results['metadata'].apply(lambda x: skill in ast.literal_eval(x)['skills'])]
            skill_score = df['sign'].sum()
            skill_total = len(df)
            mathvista_score[skill] = skill_score/skill_total
        
        # Print and save the scores
        mathvista_score_df = pd.DataFrame([mathvista_score])
        mathvista_score_df.to_excel(self.score_name, index=False)
            
    
class mathverse_eval(benchmark_eval):
    
    def __init__(self, data, path):
        super().__init__(data, path)
        self.type = 'MIX'
    
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = []
        
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        
        for i in range(len(self)):
            # if self.df['question_type'][i] == 'multi-choice':
            #     answer = self.df['answer'][i]
            #     distance_list = [distance(answer,choice) for choice in ast.literal_eval(self.df['choices'][i])]
            #     distance_index = distance_list.index(min(distance_list))
            #     answer_list.append(options[distance_index])
            # else:
            answer_list.append(self.df['answer'][i])
                
        assert len(answer_list) == len(self)
        return answer_list
    
    def find_choice(self,under_eval_bench,index):
        choices = ast.literal_eval(under_eval_bench.df['choices'][index])
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        choice = ''
        for i in range(len(choices)):
            choice += options[i]+'. '+choices[i]+'\n'
        return choice

    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the letter (A, B, C, D, etc.) that corresponds to the option chosen by the user. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        # choice = self.find_choice(under_eval_bench,index)
        
        # response = f"Choises:{choice}\nResponse:{response}"
        response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign   
    
    
    def MIX(self, response, answer, under_eval_bench, index):
        
        if under_eval_bench.df['question_type'][index] == 'multi-choice':
            
            LLM_parse,sign = self.MCQ(response,answer,under_eval_bench,index)
            
        else:
            LLM_parse,sign = self.number(response,answer)
            
        return LLM_parse,sign
        
                
    
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        
        # Initialize the score dictionaries
        mathvista_score = {'total':len(eval_results)}
        mathvista_score['correct'] = eval_results['sign'].sum()
        mathvista_score['accuracy'] = mathvista_score['correct']/mathvista_score['total']
        
        skills = []
        
        # for i in range(len(eval_results)):
        #     mata_data = ast.literal_eval(eval_results['metadata'][i])
        #     print(mata_data, flush=True)
        #     skill = mata_data['skills']
        #     skills.extend(skill)
        # skills = list(set(skills))
        
        # for skill in skills:
        #     df =eval_results[eval_results['metadata'].apply(lambda x: skill in ast.literal_eval(x)['skills'])]
        #     skill_score = df['sign'].sum()
        #     skill_total = len(df)
        #     mathvista_score[skill] = skill_score/skill_total
        
        # Print and save the scores
        mathvista_score_df = pd.DataFrame([mathvista_score])
        mathvista_score_df.to_excel(self.score_name, index=False)
        

class mathvision_eval(benchmark_eval):
    
    def __init__(self, data, path):
        super().__init__(data, path)
        self.type = 'MIX'
    
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = []
        
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        
        for i in range(len(self)):
            # if self.df['question_type'][i] == 'multi-choice':
            #     answer = self.df['answer'][i]
            #     distance_list = [distance(answer,choice) for choice in ast.literal_eval(self.df['choices'][i])]
            #     distance_index = distance_list.index(min(distance_list))
            #     answer_list.append(options[distance_index])
            # else:
            answer_list.append(self.df['answer'][i])
                
        assert len(answer_list) == len(self)
        return answer_list
    
    def find_choice(self,under_eval_bench,index):
        choices = ast.literal_eval(under_eval_bench.df['choices'][index])
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        choice = ''
        for i in range(len(choices)):
            choice += options[i]+'. '+choices[i]+'\n'
        return choice

    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the letter (A, B, C, D, etc.) that corresponds to the option chosen by the user. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        # choice = self.find_choice(under_eval_bench,index)
        
        # response = f"Choises:{choice}\nResponse:{response}"
        response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign   
    
    
    def MIX(self, response, answer, under_eval_bench, index):
        
        if under_eval_bench.df['question_type'][index] == 'multi-choice':
            
            LLM_parse,sign = self.MCQ(response,answer,under_eval_bench,index)
            
        else:
            LLM_parse,sign = self.number(response,answer)
            
        return LLM_parse,sign
        
                
    
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        
        # Initialize the score dictionaries
        mathvista_score = {'total':len(eval_results)}
        mathvista_score['correct'] = eval_results['sign'].sum()
        mathvista_score['accuracy'] = mathvista_score['correct']/mathvista_score['total']
        
        skills = []
        
        # for i in range(len(eval_results)):
        #     mata_data = ast.literal_eval(eval_results['metadata'][i])
        #     print(mata_data, flush=True)
        #     skill = mata_data['skills']
        #     skills.extend(skill)
        # skills = list(set(skills))
        
        # for skill in skills:
        #     df =eval_results[eval_results['metadata'].apply(lambda x: skill in ast.literal_eval(x)['skills'])]
        #     skill_score = df['sign'].sum()
        #     skill_total = len(df)
        #     mathvista_score[skill] = skill_score/skill_total
        
        # Print and save the scores
        mathvista_score_df = pd.DataFrame([mathvista_score])
        mathvista_score_df.to_excel(self.score_name, index=False)

class mmiq_eval(benchmark_eval):
    
    def __init__(self, data, path):
        super().__init__(data, path)
        self.type = 'MIX'
    
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = []
        
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        
        for i in range(len(self)):
            # if self.df['question_type'][i] == 'multi-choice':
            #     answer = self.df['answer'][i]
            #     distance_list = [distance(answer,choice) for choice in ast.literal_eval(self.df['choices'][i])]
            #     distance_index = distance_list.index(min(distance_list))
            #     answer_list.append(options[distance_index])
            # else:
            answer_list.append(self.df['answer'][i])
                
        assert len(answer_list) == len(self)
        return answer_list
    
    def find_choice(self,under_eval_bench,index):
        choices = ast.literal_eval(under_eval_bench.df['choices'][index])
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        choice = ''
        for i in range(len(choices)):
            choice += options[i]+'. '+choices[i]+'\n'
        return choice

    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the letter (A, B, C, D, etc.) that corresponds to the option chosen by the user. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        # choice = self.find_choice(under_eval_bench,index)
        
        # response = f"Choises:{choice}\nResponse:{response}"
        response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign   
    
    
    def MIX(self, response, answer, under_eval_bench, index):
        
        # if under_eval_bench.df['question_type'][index] == 'multi-choice':
            
        LLM_parse,sign = self.MCQ(response,answer,under_eval_bench,index)
            
        # else:
        #     LLM_parse,sign = self.number(response,answer)
            
        return LLM_parse,sign
        
                
    
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        
        # Initialize the score dictionaries
        mathvista_score = {'total':len(eval_results)}
        mathvista_score['correct'] = eval_results['sign'].sum()
        mathvista_score['accuracy'] = mathvista_score['correct']/mathvista_score['total']
        
        skills = []
        
        # for i in range(len(eval_results)):
        #     mata_data = ast.literal_eval(eval_results['metadata'][i])
        #     print(mata_data, flush=True)
        #     skill = mata_data['skills']
        #     skills.extend(skill)
        # skills = list(set(skills))
        
        # for skill in skills:
        #     df =eval_results[eval_results['metadata'].apply(lambda x: skill in ast.literal_eval(x)['skills'])]
        #     skill_score = df['sign'].sum()
        #     skill_total = len(df)
        #     mathvista_score[skill] = skill_score/skill_total
        
        # Print and save the scores
        mathvista_score_df = pd.DataFrame([mathvista_score])
        mathvista_score_df.to_excel(self.score_name, index=False)
        
class MMVP_eval(benchmark_eval):
    
    def __init__(self,data,path):
        super().__init__(data,path)
        self.type = 'MCQ'
        
    def find_response(self):
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    def find_answer(self):
        answer_list = self.df['Correct Answer'].tolist()
            
        return answer_list 
    
    def find_choice(self,index) -> str:
        return self.df.loc[index,'Options']
    
    def MCQ(self, response, answer, under_eval_bench, index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the lower letter with parentheses (a), (b), etc. that corresponds to the option chosen by the response. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        choice = under_eval_bench.find_choice(index)
        
        response = f"Choises:{choice}\nResponse:{response}"
        # response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    def extract_answer(self, LLM_parse, answer):
        if answer in LLM_parse.lower():
            return 1
        else:
            return 0
        
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        
        # Initialize the score dictionaries
        MMVP_score = {'total':len(eval_results)/2}
        correct = 0
        for i in range(0,len(eval_results),2):
            if eval_results['sign'][i] == 1 and eval_results['sign'][i+1] == 1:
                correct += 1
        MMVP_score['correct'] = correct
        MMVP_score['accuracy'] = MMVP_score['correct']/MMVP_score['total']
        
        #save the scores to excel
        MMVP_score_df = pd.DataFrame([MMVP_score])
        MMVP_score_df.to_excel(self.score_name, index=False)
        
        #print the scores
        for key, value in MMVP_score.items():
            print(f'{key}: {value:.4f}')
        
        
class mirage_eval(benchmark_eval):
    
    def __init__(self, data, path):
        super().__init__(data, path)
        self.type = 'MIX'

    @retry(tries=3,delay=2)
    def LLM(self,response,prompt=None):
        if prompt:
        
            data = {'query':response,'system_prompt':prompt}
        
        else:
            data = {'query':response}
        
        output = self.model.request(data)
        
        return output
    
    def MCQ(self,response,answer,question,under_eval_bench,index):
        
        prompt = f"Please read the user's response to a multiple-choice question. Your task is to identify and output the letter (A, B, C, D, etc.) that corresponds to the option chosen by the user. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'. To avoid the model forget to answer the question by the letter, we provide the original question here: {question}. Now directly extract the answer from the response."
        
        # choice = self.find_choice(under_eval_bench,index)
        
        # response = f"Choises:{choice}\nResponse:{response}"
        # # response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    def number(self,response,answer):
        
        prompt = """You will be provided with an response to a question that requires a numerical answer. It can be a int, float or a list. Your task is to indentify and parse the final numerical answer. You answer should only contain the answer of number(or number list) with no punctuations. Make sure your answer is in the correct format and is as accurate as possible. if you can not find the number or it is unclear, output 'unclear'
        Please read the following example. Then extract the answer from the model response and type it at the end of the prompt. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer.
        Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
        Question: Which number is missing?

        Model response: The number missing in the sequence is 14.

        Extracted answer: 14

        Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
        Question: What is the fraction of females facing the camera?

        Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.        

        Extracted answer: 0.6

        Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
        Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

        Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

        Extracted answer: 1.45

        Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.      
        Question: Between which two years does the line  graph saw its maximum peak?

        Model response: The line graph saw its maximum peak between 2007 and 2008.

        Extracted answer: [2007, 2008]
        """
        #print(response, type(response), flush=True)
        try:
            response = 'response:'+response+'Extracted answer:'
        except:
            response = "none"
            response = 'response:'+response+'Extracted answer:'
        
        
        LLM_parse = self.LLM(response,prompt)
        print(LLM_parse, flush=True)
        
        
        sign = self.extract_answer(LLM_parse,answer,num=True)
        
        return LLM_parse,sign
    
    def number_approx(self,response,answer):
        
        prompt = """You will be provided with an response to a question that requires a numerical answer. It can be a int, float or a list. Your task is to indentify and parse the final numerical answer. You answer should only contain the answer of number(or number list) with no punctuations. Make sure your answer is in the correct format and is as accurate as possible. if you can not find the number or it is unclear, output 'unclear'
        Please read the following example. Then extract the answer from the model response and type it at the end of the prompt. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer.
        Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
        Question: Which number is missing?

        Model response: The number missing in the sequence is 14.

        Extracted answer: 14

        Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
        Question: What is the fraction of females facing the camera?

        Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.        

        Extracted answer: 0.6

        Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
        Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

        Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

        Extracted answer: 1.45

        Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.      
        Question: Between which two years does the line  graph saw its maximum peak?

        Model response: The line graph saw its maximum peak between 2007 and 2008.

        Extracted answer: [2007, 2008]
        """
        #print(response, type(response), flush=True)
        try:
            response = 'response:'+response+'Extracted answer:'
        except:
            response = "none"
            response = 'response:'+response+'Extracted answer:'
        
        
        LLM_parse = self.LLM(response,prompt)
        print(LLM_parse, flush=True)
        
        
        sign = self.extract_answer_approx(LLM_parse,answer,num=True)
        
        return LLM_parse,sign
    
    def extract_answer(self,LLM_parse,answer,num=False):
        
        if num:
            try:
                answer = int(answer)
                num_type = 'int'
            except ValueError:
                try:
                    answer = float(answer)
                    num_type = 'float'
                except ValueError:
                    try:
                        answer = ast.literal_eval(answer)
                        num_type = 'list'
                    except:
                        num_type = 'str'
                    
            if num_type == 'int':
                try:
                    LLM_parse = int(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'float':
                try:
                    LLM_parse = float(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'list':
                try:
                    LLM_parse = ast.literal_eval(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except:
                    return 0
            else:
                return 0
            
        else:
            
            LLM_parse = LLM_parse.lower()
            answer = answer.lower()
            
            char='.()[],:;!*#{}'
            for c in char:
                LLM_parse=LLM_parse.replace(c,' ')
                
            LLM_parse = re.sub(r'\s+', ' ', LLM_parse).strip()
                
            LLM_parse_list = LLM_parse.split(' ')
            
            
            if answer in LLM_parse_list:
                
                return 1
            
            else:
                return 0
    
    def extract_answer_approx(self,LLM_parse,answer,num=False):
        
        if num:
            try:
                answer = int(answer)
                num_type = 'int'
            except ValueError:
                try:
                    answer = float(answer)
                    num_type = 'float'
                except ValueError:
                    try:
                        answer = ast.literal_eval(answer)
                        num_type = 'list'
                    except:
                        num_type = 'str'
                    
            if num_type == 'int':
                try:
                    LLM_parse = int(LLM_parse)
                    approx_rate = min(int(LLM_parse)/int(answer),int(answer)/int(LLM_parse))
                    if LLM_parse == answer or approx_rate >= 0.95:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'float':
                try:
                    LLM_parse = float(LLM_parse)
                    approx_rate = min(float(LLM_parse)/float(answer),float(answer)/float(LLM_parse))
                    if LLM_parse == answer or approx_rate >= 0.95:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'list':
                try:
                    LLM_parse = ast.literal_eval(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except:
                    return 0
            else:
                return 0
            
        else:
            
            LLM_parse = LLM_parse.lower()
            answer = answer.lower()
            
            char='.()[],:;!*#{}'
            for c in char:
                LLM_parse=LLM_parse.replace(c,' ')
                
            LLM_parse = re.sub(r'\s+', ' ', LLM_parse).strip()
                
            LLM_parse_list = LLM_parse.split(' ')
            
            
            if answer in LLM_parse_list:
                
                return 1
            
            else:
                return 0
    
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()
    
    # def find_answer(self):
    #     answer_list = []
        
    #     options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        
    #     for i in range(len(self)):
    #         if self.df['question_type'][i] == 'multi_choice':
    #             answer = self.df['answer'][i]
    #             distance_list = [distance(answer,choice) for choice in ast.literal_eval(self.df['choices'][i])]
    #             distance_index = distance_list.index(min(distance_list))
    #             answer_list.append(options[distance_index])
    #         elif self.df['question_type'][i] == 'approx':
    #             answer_list.append(self.df['answer'][i])
    #         else:
    #             answer_list.append(self.df['answer'][i])
                
    #     assert len(answer_list) == len(self)
    #     return answer_list
    def find_answer(self):
        answer_list = []
        
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        
        for i in range(len(self)):
            # if self.df['question_type'][i] == 'multi-choice':
            #     answer = self.df['answer'][i]
            #     distance_list = [distance(answer,choice) for choice in ast.literal_eval(self.df['choices'][i])]
            #     distance_index = distance_list.index(min(distance_list))
            #     answer_list.append(options[distance_index])
            # else:
            answer_list.append(self.df['answer'][i])
                
        assert len(answer_list) == len(self)
        return answer_list
    
    def find_choice(self,under_eval_bench,index):
        choices = ast.literal_eval(under_eval_bench.df['choices'][index])
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        choice = ''
        for i in range(len(choices)):
            choice += options[i]+'. '+choices[i]+'\n'
        return choice
            
    
    
    def MIX(self, response, answer, under_eval_bench, index):
        
        if under_eval_bench.df['question_type'][index] == 'multi_choice':
            
            LLM_parse,sign = self.MCQ(response,answer, under_eval_bench.df['prompt'][index],under_eval_bench,index)
        
        elif under_eval_bench.df['question_type'][index] == 'approx':
            
            LLM_parse,sign = self.number_approx(response,answer)
            
        else:
            LLM_parse,sign = self.number(response,answer)
            
        return LLM_parse,sign
        
                
    
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        candidate_classes = ['Algebraic', 'Arithmetic', 'Geometry', 'Logical', 'Scientific', 'Spatial', 'Statistical']
        # Initialize the score dictionaries
        mathvista_score = {'total':len(eval_results)}
        mathvista_score['correct'] = eval_results['sign'].sum()
        mathvista_score['accuracy'] = mathvista_score['correct']/mathvista_score['total']

        cache_results = {}
        for i in candidate_classes:
            cache_results[i] = {'total':0, 'correct': 0, 'accuracy': 0.0}

        for i in range(len(eval_results)):
            # single_result = eval_results[i]
            classification = eval_results['normalized_classification'][i]
            cache_results[classification]['total'] += 1
            cache_results[classification]['correct'] += int(eval_results['sign'][i])


        
        # skills = []
        
        # for i in range(len(eval_results)):
        #     mata_data = ast.literal_eval(eval_results['metadata'][i])
        #     # print(mata_data)
        #     skill = mata_data['skills']
        #     skills.extend(skill)
        # skills = list(set(skills))
        
        for skill in candidate_classes:
            # df =eval_results[eval_results['metadata'].apply(lambda x: skill in ast.literal_eval(x)['skills'])]
            skill_score = cache_results[skill]['correct']#df['sign'].sum()
            skill_total = cache_results[skill]['total']#len(df)
            mathvista_score[skill] = skill_score/skill_total
        
        # Print and save the scores
        mathvista_score_df = pd.DataFrame([mathvista_score])
        mathvista_score_df.to_excel(self.score_name, index=False)
                
        
class mirage_cot_eval(benchmark_eval):
    
    def __init__(self, data, path):
        super().__init__(data, path)
        self.type = 'MIX'

    @retry(tries=3,delay=2)
    def LLM(self,response,prompt=None):
        if prompt:
        
            data = {'query':response,'system_prompt':prompt}
        
        else:
            data = {'query':response}
        
        output = self.model.request(data)
        
        return output
    
    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the letter (A, B, C, D, etc.) that corresponds to the option chosen by the user. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        # choice = self.find_choice(under_eval_bench,index)
        
        # response = f"Choises:{choice}\nResponse:{response}"
        # # response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    def number(self,response,answer):
        
        prompt = """You will be provided with an response to a question that requires a numerical answer. It can be a int, float or a list. Your task is to indentify and parse the final numerical answer. You answer should only contain the answer of number(or number list) with no punctuations. Make sure your answer is in the correct format and is as accurate as possible. if you can not find the number or it is unclear, output 'unclear'
        Please read the following example. Then extract the answer from the model response and type it at the end of the prompt. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer.
        Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
        Question: Which number is missing?

        Model response: The number missing in the sequence is 14.

        Extracted answer: 14

        Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
        Question: What is the fraction of females facing the camera?

        Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.        

        Extracted answer: 0.6

        Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
        Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

        Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

        Extracted answer: 1.45

        Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.      
        Question: Between which two years does the line  graph saw its maximum peak?

        Model response: The line graph saw its maximum peak between 2007 and 2008.

        Extracted answer: [2007, 2008]
        """
        #print(response, type(response), flush=True)
        try:
            response = 'response:'+response+'Extracted answer:'
        except:
            response = "none"
            response = 'response:'+response+'Extracted answer:'
        
        
        LLM_parse = self.LLM(response,prompt)
        print(LLM_parse, flush=True)
        
        
        sign = self.extract_answer(LLM_parse,answer,num=True)
        
        return LLM_parse,sign
    
    def number_approx(self,response,answer):
        
        prompt = """You will be provided with an response to a question that requires a numerical answer. It can be a int, float or a list. Your task is to indentify and parse the final numerical answer. You answer should only contain the answer of number(or number list) with no punctuations. Make sure your answer is in the correct format and is as accurate as possible. if you can not find the number or it is unclear, output 'unclear'
        Please read the following example. Then extract the answer from the model response and type it at the end of the prompt. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer.
        Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
        Question: Which number is missing?

        Model response: The number missing in the sequence is 14.

        Extracted answer: 14

        Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
        Question: What is the fraction of females facing the camera?

        Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.        

        Extracted answer: 0.6

        Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
        Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

        Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

        Extracted answer: 1.45

        Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.      
        Question: Between which two years does the line  graph saw its maximum peak?

        Model response: The line graph saw its maximum peak between 2007 and 2008.

        Extracted answer: [2007, 2008]
        """
        #print(response, type(response), flush=True)
        try:
            response = 'response:'+response+'Extracted answer:'
        except:
            response = "none"
            response = 'response:'+response+'Extracted answer:'
        
        
        LLM_parse = self.LLM(response,prompt)
        print(LLM_parse, flush=True)
        
        
        sign = self.extract_answer_approx(LLM_parse,answer,num=True)
        
        return LLM_parse,sign
    
    def extract_answer(self,LLM_parse,answer,num=False):
        
        if num:
            try:
                answer = int(answer)
                num_type = 'int'
            except ValueError:
                try:
                    answer = float(answer)
                    num_type = 'float'
                except ValueError:
                    try:
                        answer = ast.literal_eval(answer)
                        num_type = 'list'
                    except:
                        num_type = 'str'
                    
            if num_type == 'int':
                try:
                    LLM_parse = int(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'float':
                try:
                    LLM_parse = float(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'list':
                try:
                    LLM_parse = ast.literal_eval(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except:
                    return 0
            else:
                return 0
            
        else:
            
            LLM_parse = LLM_parse.lower()
            answer = answer.lower()
            
            char='.()[],:;!*#{}'
            for c in char:
                LLM_parse=LLM_parse.replace(c,' ')
                
            LLM_parse = re.sub(r'\s+', ' ', LLM_parse).strip()
                
            LLM_parse_list = LLM_parse.split(' ')
            
            
            if answer in LLM_parse_list:
                
                return 1
            
            else:
                return 0
    
    def extract_answer_approx(self,LLM_parse,answer,num=False):
        
        if num:
            try:
                answer = int(answer)
                num_type = 'int'
            except ValueError:
                try:
                    answer = float(answer)
                    num_type = 'float'
                except ValueError:
                    try:
                        answer = ast.literal_eval(answer)
                        num_type = 'list'
                    except:
                        num_type = 'str'
                    
            if num_type == 'int':
                try:
                    LLM_parse = int(LLM_parse)
                    approx_rate = min(int(LLM_parse)/int(answer),int(answer)/int(LLM_parse))
                    if LLM_parse == answer or approx_rate >= 0.95:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'float':
                try:
                    LLM_parse = float(LLM_parse)
                    approx_rate = min(float(LLM_parse)/float(answer),float(answer)/float(LLM_parse))
                    if LLM_parse == answer or approx_rate >= 0.95:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'list':
                try:
                    LLM_parse = ast.literal_eval(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except:
                    return 0
            else:
                return 0
            
        else:
            
            LLM_parse = LLM_parse.lower()
            answer = answer.lower()
            
            char='.()[],:;!*#{}'
            for c in char:
                LLM_parse=LLM_parse.replace(c,' ')
                
            LLM_parse = re.sub(r'\s+', ' ', LLM_parse).strip()
                
            LLM_parse_list = LLM_parse.split(' ')
            
            
            if answer in LLM_parse_list:
                
                return 1
            
            else:
                return 0
    
    def find_response(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col]
        return self.df[response_column].squeeze().tolist()

    def find_answer(self):
        answer_list = []
        
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        
        for i in range(len(self)):
            answer_list.append(self.df['answer'][i])
                
        assert len(answer_list) == len(self)
        return answer_list
    
    def find_choice(self,under_eval_bench,index):
        choices = ast.literal_eval(under_eval_bench.df['choices'][index])
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        choice = ''
        for i in range(len(choices)):
            choice += options[i]+'. '+choices[i]+'\n'
        return choice
            
    
    
    def MIX(self, response, answer, under_eval_bench, index):
        
        if under_eval_bench.df['question_type'][index] == 'multi_choice':
            
            LLM_parse,sign = self.MCQ(response,answer,under_eval_bench,index)
        
        elif under_eval_bench.df['question_type'][index] == 'approx':
            
            LLM_parse,sign = self.number_approx(response,answer)
            
        else:
            LLM_parse,sign = self.number(response,answer)
            
        return LLM_parse,sign
        
                
    
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        candidate_classes = ['Algebraic', 'Arithmetic', 'Geometry', 'Logical', 'Scientific', 'Spatial', 'Statistical']
        # Initialize the score dictionaries
        mathvista_score = {'total':len(eval_results)}
        mathvista_score['correct'] = eval_results['sign'].sum()
        mathvista_score['accuracy'] = mathvista_score['correct']/mathvista_score['total']

        cache_results = {}
        for i in candidate_classes:
            cache_results[i] = {'total':0, 'correct': 0, 'accuracy': 0.0}

        for i in range(len(eval_results)):
            # single_result = eval_results[i]
            classification = eval_results['normalized_classification'][i]
            cache_results[classification]['total'] += 1
            cache_results[classification]['correct'] += int(eval_results['sign'][i])
        
        for skill in candidate_classes:
            df =eval_results[eval_results['metadata'].apply(lambda x: skill in ast.literal_eval(x)['skills'])]
            skill_score = cache_results[skill]['correct']#df['sign'].sum()
            skill_total = cache_results[skill]['total']#len(df)
            mathvista_score[skill] = skill_score/skill_total
        
        # Print and save the scores
        mathvista_score_df = pd.DataFrame([mathvista_score])
        mathvista_score_df.to_excel(self.score_name, index=False)


class mirage_f1_eval(benchmark_eval):
    
    def __init__(self, data, path):
        super().__init__(data, path)
        self.type = 'F1'
    
    def Eval_main(self):
        # type of evaluation: YORN, MCQ, etc
        
        existing_length = self.checking_exiting_file()
        if existing_length == len(self):
            console.print(f'All response evaluations have been saved. No further evaluations needed.',style='bold green')
        
        else:
            
            eval_benchmark = self.cut(existing_length)
            
            for index,row in tqdm(eval_benchmark.itterows(),total=len(eval_benchmark)):
                response,answer = row
                
                if self.type == 'F1':
                    claim_analysis, step_analysis, claim_precision, claim_recall, claim_f1, step_precision, step_recall, step_f1 = self.F1(response, answer, eval_benchmark, index)
                else:
                    raise ValueError('Invalid evaluation type')
                
                answer_df= deepcopy(eval_benchmark.get(index))
                answer_df.loc[:,'claim_analysis'] = claim_analysis
                answer_df.loc[:,'step_analysis'] = step_analysis
                answer_df.loc[:,'claim_precision'] = claim_precision
                answer_df.loc[:,'claim_recall'] = claim_recall
                answer_df.loc[:,'claim_f1'] = claim_f1
                answer_df.loc[:,'step_precision'] = step_precision
                answer_df.loc[:,'step_recall'] = step_recall
                answer_df.loc[:,'step_f1'] = step_f1
                
                self.saved(answer_df)
                
            results = import_file(self.result_path)
            assert len(results) == len(self.df)
            print(f'All response evaluations saved.')

    @retry(tries=3,delay=2)
    def LLM(self,response,prompt=None):
        if prompt:
        
            data = {'query':response,'system_prompt':prompt}
        
        else:
            data = {'query':response}
        
        output = self.model.request(data)
        
        return output
    
    def MCQ(self,response,answer,under_eval_bench,index):
        
        prompt = "Please read the user's response to a multiple-choice question. Your task is to identify and output the letter (A, B, C, D, etc.) that corresponds to the option chosen by the user. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer. If the response does not clearly indicate a choice or no option is mentioned, output 'unclear'. Only output a single letter or 'unclear'."
        
        # choice = self.find_choice(under_eval_bench,index)
        
        # response = f"Choises:{choice}\nResponse:{response}"
        # # response = f"Response:{response}"
        
        LLM_parse = self.LLM(response,prompt)
        # print(LLM_parse)
        
        sign = self.extract_answer(LLM_parse,answer)
            
        return LLM_parse,sign
    
    def number(self,response,answer):
        
        prompt = """You will be provided with an response to a question that requires a numerical answer. It can be a int, float or a list. Your task is to indentify and parse the final numerical answer. You answer should only contain the answer of number(or number list) with no punctuations. Make sure your answer is in the correct format and is as accurate as possible. if you can not find the number or it is unclear, output 'unclear'
        Please read the following example. Then extract the answer from the model response and type it at the end of the prompt. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer.
        Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
        Question: Which number is missing?

        Model response: The number missing in the sequence is 14.

        Extracted answer: 14

        Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
        Question: What is the fraction of females facing the camera?

        Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.        

        Extracted answer: 0.6

        Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
        Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

        Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

        Extracted answer: 1.45

        Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.      
        Question: Between which two years does the line  graph saw its maximum peak?

        Model response: The line graph saw its maximum peak between 2007 and 2008.

        Extracted answer: [2007, 2008]
        """
        #print(response, type(response), flush=True)
        try:
            response = 'response:'+response+'Extracted answer:'
        except:
            response = "none"
            response = 'response:'+response+'Extracted answer:'
        
        
        LLM_parse = self.LLM(response,prompt)
        print(LLM_parse, flush=True)
        
        
        sign = self.extract_answer(LLM_parse,answer,num=True)
        
        return LLM_parse,sign
    
    def number_approx(self,response,answer):
        
        prompt = """You will be provided with an response to a question that requires a numerical answer. It can be a int, float or a list. Your task is to indentify and parse the final numerical answer. You answer should only contain the answer of number(or number list) with no punctuations. Make sure your answer is in the correct format and is as accurate as possible. if you can not find the number or it is unclear, output 'unclear'
        Please read the following example. Then extract the answer from the model response and type it at the end of the prompt. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer.
        Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
        Question: Which number is missing?

        Model response: The number missing in the sequence is 14.

        Extracted answer: 14

        Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
        Question: What is the fraction of females facing the camera?

        Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.        

        Extracted answer: 0.6

        Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
        Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

        Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

        Extracted answer: 1.45

        Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.      
        Question: Between which two years does the line  graph saw its maximum peak?

        Model response: The line graph saw its maximum peak between 2007 and 2008.

        Extracted answer: [2007, 2008]
        """
        #print(response, type(response), flush=True)
        try:
            response = 'response:'+response+'Extracted answer:'
        except:
            response = "none"
            response = 'response:'+response+'Extracted answer:'
        
        
        LLM_parse = self.LLM(response,prompt)
        print(LLM_parse, flush=True)
        
        
        sign = self.extract_answer_approx(LLM_parse,answer,num=True)
        
        return LLM_parse,sign
    
    def extract_answer(self,LLM_parse,answer,num=False):
        
        if num:
            try:
                answer = int(answer)
                num_type = 'int'
            except ValueError:
                try:
                    answer = float(answer)
                    num_type = 'float'
                except ValueError:
                    try:
                        answer = ast.literal_eval(answer)
                        num_type = 'list'
                    except:
                        num_type = 'str'
                    
            if num_type == 'int':
                try:
                    LLM_parse = int(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'float':
                try:
                    LLM_parse = float(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'list':
                try:
                    LLM_parse = ast.literal_eval(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except:
                    return 0
            else:
                return 0
            
        else:
            
            LLM_parse = LLM_parse.lower()
            answer = answer.lower()
            
            char='.()[],:;!*#{}'
            for c in char:
                LLM_parse=LLM_parse.replace(c,' ')
                
            LLM_parse = re.sub(r'\s+', ' ', LLM_parse).strip()
                
            LLM_parse_list = LLM_parse.split(' ')
            
            
            if answer in LLM_parse_list:
                
                return 1
            
            else:
                return 0
    
    def extract_answer_approx(self,LLM_parse,answer,num=False):
        
        if num:
            try:
                answer = int(answer)
                num_type = 'int'
            except ValueError:
                try:
                    answer = float(answer)
                    num_type = 'float'
                except ValueError:
                    try:
                        answer = ast.literal_eval(answer)
                        num_type = 'list'
                    except:
                        num_type = 'str'
                    
            if num_type == 'int':
                try:
                    LLM_parse = int(LLM_parse)
                    approx_rate = min(int(LLM_parse)/int(answer),int(answer)/int(LLM_parse))
                    if LLM_parse == answer or approx_rate >= 0.95:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'float':
                try:
                    LLM_parse = float(LLM_parse)
                    approx_rate = min(float(LLM_parse)/float(answer),float(answer)/float(LLM_parse))
                    if LLM_parse == answer or approx_rate >= 0.95:
                        return 1
                    else:
                        return 0
                except ValueError:
                    return 0
                
            elif num_type == 'list':
                try:
                    LLM_parse = ast.literal_eval(LLM_parse)
                    if LLM_parse == answer:
                        return 1
                    else:
                        return 0
                except:
                    return 0
            else:
                return 0
            
        else:
            
            LLM_parse = LLM_parse.lower()
            answer = answer.lower()
            
            char='.()[],:;!*#{}'
            for c in char:
                LLM_parse=LLM_parse.replace(c,' ')
                
            LLM_parse = re.sub(r'\s+', ' ', LLM_parse).strip()
                
            LLM_parse_list = LLM_parse.split(' ')
            
            
            if answer in LLM_parse_list:
                
                return 1
            
            else:
                return 0
    
    def find_answer(self):
        # you must rewrite this function to find the response in your data and return it as a list
        response_column = [col for col in self.df.columns if 'extract_claims' in col and 'pred' not in col]
        return self.df[response_column].values.tolist()

    def find_response(self):
        answer_list = []
        response_column = [col for col in self.df.columns if 'extract_claims_pred' in col and 'intermediate' not in col]
        return self.df[response_column].values.tolist()

    
    def find_choice(self,under_eval_bench,index):
        choices = ast.literal_eval(under_eval_bench.df['choices'][index])
        options = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
        choice = ''
        for i in range(len(choices)):
            choice += options[i]+'. '+choices[i]+'\n'
        return choice
            
    
    
    def MIX(self, response, answer, under_eval_bench, index):
        
        if under_eval_bench.df['question_type'][index] == 'multi_choice':
            
            LLM_parse,sign = self.MCQ(response,answer,under_eval_bench,index)
        
        elif under_eval_bench.df['question_type'][index] == 'approx':
            
            LLM_parse,sign = self.number_approx(response,answer)
            
        else:
            LLM_parse,sign = self.number(response,answer)
            
        return LLM_parse,sign
    
    def extract_claims(self, input_string):
        pattern = re.compile(r'<claim>(.*?)</claim>', re.DOTALL)
        claims = pattern.findall(input_string)
        claims = [f"<claim>{c}</claim>" for c in claims]
        claims = "\n".join(claims)
        return claims

    def extract_steps(self, input_string):
        pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
        steps = pattern.findall(input_string)
        steps = [f"<step>{c}</step>" for c in steps]
        steps = "\n".join(steps)
        return steps
    
    def extract_pred_match(self, input_string):
        pattern = re.compile(r'pred_match(.*?)/pred_match', re.DOTALL)
        print(input_string)
        pred_match = pattern.findall(input_string)[0]
        print(pred_match) # >MATCH,REASONABLE,REASONABLE,REASONABLE,CONFLICT,MATCH<
        pattern1 = re.compile(r'\b(MATCH|REASONABLE|CONFLICT)\b')
        pred_match = pattern1.findall(pred_match)
        pred_match = [i.strip() for i in pred_match]
        # claims = [f"<claim>{c}</claim>" for c in claims]
        # claims = "\n".join(claims)
        return pred_match

    def extract_gt_match(self, input_string):
        pattern = re.compile(r'gt_match(.*?)/gt_match', re.DOTALL)
        print(input_string)
        gt_match = pattern.findall(input_string)[0]
        pattern1 = re.compile(r'\b(MATCH|CONFLICT)\b')
        gt_match = pattern1.findall(gt_match)
        gt_match = [i.strip() for i in gt_match]
        # claims = [f"<claim>{c}</claim>" for c in claims]
        # claims = "\n".join(claims)
        return gt_match

        
    def F1(self, response, answer, eval_benchmark, index):
        # first, compute claims F1 score
        # print((response), (answer))
        response, answer = response[0], answer[0]

        question_prompt = eval_benchmark.df['prompt'][index]
        try:
            question_type = eval_benchmark.df['question_type'][index]
            if isinstance(question_type, str) is False:
                question_type = 'free_form'
        except:
            question_type = 'free_form'
        # print(question_prompt)
        if question_type.startswith('multi'):
            question_prompt = "Here is a multiple choice question: " + question_prompt
        elif question_type.startswith('free'):
            question_prompt = "Here is a free-from question and requires the answer is exactly the same: " + question_prompt
        else:
            question_prompt = "Here is a free-from question and the answer can be approximately similar with 5\% error: " + question_prompt
        
        pred_claims = self.extract_claims(response)
        gt_claims = self.extract_claims(answer)
        match_prompt = judge_claim_f1_score_prompt.format(question=question_prompt, pred_claims=pred_claims, gt_claims=gt_claims)
        LLM_parse = self.LLM(match_prompt)
        # print(LLM_parse, flush=True)
        claim_precision, claim_recall, claim_f1 = self.extract_match_results(LLM_parse)

        # then, compute step F1 score
        pred_steps = self.extract_steps(response)
        gt_steps = self.extract_steps(answer)
        step_match_prompt = judge_step_f1_score_prompt.format(question=question_prompt, pred_steps=pred_steps, gt_steps=gt_steps)
        step_LLM_parse = self.LLM(step_match_prompt)
        # print(step_LLM_parse, flush=True)
        step_precision, step_recall, step_f1 = self.extract_match_results(step_LLM_parse)
        return LLM_parse, step_LLM_parse, claim_precision, claim_recall, claim_f1, step_precision, step_recall, step_f1
        # raise NotImplementedError
    def calculate_metrics(self, pred_matches, gt_matches):
        # Calculate True Positive matches
        tp_pred_scores = []
        for m in pred_matches:
            if m == 'MATCH':
                tp_pred_scores.append(1)
            elif m == 'REASONABLE':
                tp_pred_scores.append(1)
            else:
                tp_pred_scores.append(0)
        tp_pred = sum(tp_pred_scores)#sum(1 for m in pred_matches if m == "MATCH" or m == "REASONABLE")
        tp_gt = sum(1 for m in gt_matches if m == "MATCH")
        
        # Precision: TP_pred / (Total predicted MATCH)
        precision = tp_pred / len(pred_matches) if len(pred_matches) > 0 else 0.0
        
        # Recall: TP_gt / (Total ground-truth claims)
        recall = tp_gt / len(gt_matches) if len(gt_matches) > 0 else 0.0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1

    # Example usage:
    # pred_matches = ["MATCH", "CONFLICT"]
    # gt_matches = ["MATCH", "CONFLICT"]
    # precision, recall, f1 = calculate_metrics(pred_matches, gt_matches)
    def extract_match_results(self, LLM_parse):
        pred_match = self.extract_pred_match(LLM_parse)
        gt_match = self.extract_gt_match(LLM_parse)
        print(pred_match, "\n", gt_match)
        precision, recall, f1 = self.calculate_metrics(pred_matches=pred_match, gt_matches=gt_match)
        return precision, recall, f1


    # def LLM_match(self,merged_prompt):
        
    #     # prompt = """You will be provided with an response to a question that requires a numerical answer. It can be a int, float or a list. Your task is to indentify and parse the final numerical answer. You answer should only contain the answer of number(or number list) with no punctuations. Make sure your answer is in the correct format and is as accurate as possible. if you can not find the number or it is unclear, output 'unclear'
    #     # Please read the following example. Then extract the answer from the model response and type it at the end of the prompt. The answer usually exists in <answer>...</answer>, if not, please focus on the end of response to extract answer.
    #     # Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
    #     # Question: Which number is missing?

    #     # Model response: The number missing in the sequence is 14.

    #     # Extracted answer: 14

    #     # Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
    #     # Question: What is the fraction of females facing the camera?

    #     # Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.        

    #     # Extracted answer: 0.6

    #     # Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
    #     # Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

    #     # Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

    #     # Extracted answer: 1.45

    #     # Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.      
    #     # Question: Between which two years does the line  graph saw its maximum peak?

    #     # Model response: The line graph saw its maximum peak between 2007 and 2008.

    #     # Extracted answer: [2007, 2008]
    #     # """
    #     # #print(response, type(response), flush=True)
    #     # try:
    #     #     response = 'response:'+response+'Extracted answer:'
    #     # except:
    #     #     response = "none"
    #     #     response = 'response:'+response+'Extracted answer:'
        
        
    #     LLM_parse = self.LLM(merged_prompt)
    #     print(LLM_parse, flush=True)
        
        
    #     sign = self.extract_match_results(LLM_parse,answer,num=True)
        
    #     return LLM_parse,sign
    
    def score(self):
        # Load the evaluation results
        eval_results = import_file(self.result_path)
        candidate_classes = ['claim_precision', 'claim_recall', 'claim_f1', 'step_precision', 'step_recall', 'step_f1']
        # Initialize the score dictionaries
        mathvista_score = {'total':len(eval_results)}
        # mathvista_score['correct'] = eval_results['sign'].sum()
        # mathvista_score['accuracy'] = mathvista_score['correct']/mathvista_score['total']

        cache_results  = {'total':0, 
                            'claim_precision': 0.0, 
                            'claim_recall': 0.0,
                            'claim_f1': 0.0,
                            'step_precision': 0.0, 
                            'step_recall': 0.0,
                            'step_f1': 0.0,
                            }

        for i in range(len(eval_results)):
            # single_result = eval_results[i]
            # classification = eval_results['normalized_classification'][i]
            cache_results['total'] += 1
            cache_results['claim_precision'] += float(eval_results['claim_precision'][i])
            cache_results['claim_recall'] += float(eval_results['claim_recall'][i])
            cache_results['claim_f1'] += float(eval_results['claim_f1'][i])
            cache_results['step_precision'] += float(eval_results['step_precision'][i])
            cache_results['step_recall'] += float(eval_results['step_recall'][i])
            cache_results['step_f1'] += float(eval_results['step_f1'][i])
        
        for skill in candidate_classes:
            # df =eval_results[eval_results['metadata'].apply(lambda x: skill in ast.literal_eval(x)['skills'])]
            skill_score = cache_results[skill]
            skill_total = cache_results['total']#len(df)
            mathvista_score[skill] = skill_score/skill_total
        
        # Print and save the scores
        mathvista_score_df = pd.DataFrame([mathvista_score])
        mathvista_score_df.to_excel(self.score_name, index=False)