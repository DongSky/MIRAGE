import os
from tqdm import tqdm
from rich.console import Console
from ..LLM.LLM_state import get_api_client
from ..utils.file import import_file
from ..Vic.Vic_main import Vic, Vic_async
from ..utils.file import to_base64_image
from ..Vic.visual_inference_prompt import classify_prompt
from .extract_claim_prompt import extract_claims_prompt, extract_question_specific_hint_prompt, topic_specific_prompts, rephrase_prompt, judge_uncertainty_by_llms_prompt, judge_hallucination_few_shot_prompt
import re
from pprint import pprint
console = Console()

descript_image_prompt="""
You are a Multimodal Large Language Model (MLLM) tasked with analyzing scientific questions that include both textual descriptions and corresponding images. Your goal is to:
    a. Classify the question into one of the following categories: 1) Scientific 2) Arithmetic 3) Geometry 4) Logical 5) Statistical 6) Algebraic 7) Numeric Commonsense
    b. Provide a detailed description of the image, focusing on extracting visual details that are directly relevant to solving the question. Your description should help clarify the problem and provide necessary information for the next steps in solving it. 
    c. Note that you only need to finish the two task above, and don't need to solve this question directly.
To help you better understand the task, here are the characteristics of each category and examples of what their corresponding images might look like:
    1) Scientific:
        Characteristics: Questions in this category are related to natural sciences (e.g., physics, chemistry, biology) or applied sciences (e.g., engineering, environmental science). They often involve concepts like forces, energy, chemical reactions, or biological processes.
        Image Examples: Diagrams of experimental setups (e.g., a pulley system, a chemical reaction in a beaker). Graphs showing scientific data (e.g., temperature vs. time, force vs. distance). Images of natural phenomena (e.g., a plant cell under a microscope, a planet in space).
    2) Arithmetic:
        Characteristics: Questions in this category involve basic mathematical operations like addition, subtraction, multiplication, or division. They often require calculations with numbers.
        Image Examples: A table of numbers or a simple equation written on a chalkboard. A real-world scenario depicted visually (e.g., a group of apples with a question about how many are left after some are eaten). A calculator or abacus as a visual aid.
    3) Geometry:
        Characteristics: Questions in this category involve shapes, sizes, angles, and spatial relationships. They often require knowledge of geometric formulas or properties.
        Image Examples: Diagrams of geometric shapes (e.g., triangles, circles, rectangles) with labeled dimensions. 3D objects (e.g., cubes, spheres) with measurements. Coordinate planes with plotted points or lines.
    4) Logical:
        Characteristics: Questions in this category involve reasoning, patterns, or sequences. They often require deductive or inductive logic to solve.
        Image Examples: Puzzles or patterns (e.g., a sequence of shapes or numbers). Flowcharts or decision trees. Visual representations of logical statements (e.g., Venn diagrams).
    5) Statistical:
        Characteristics: Questions in this category involve data analysis, probability, or interpretation of graphs and charts. They often require understanding of mean, median, mode, or distributions.
        Image Examples: Bar charts, pie charts, or line graphs. Tables of data with rows and columns. Histograms or scatter plots.
    6) Algebraic:
        Characteristics: Questions in this category involve variables, equations, and algebraic expressions. They often require solving for unknowns or manipulating equations.
        Image Examples: Equations written on a whiteboard or paper (e.g., 2x + 3 = 7). Graphs of linear or quadratic functions. Visual representations of algebraic concepts (e.g., balancing scales for equations).
    7) Numeric Commonsense:
        Characteristics: Questions in this category involve everyday numerical reasoning or practical applications of numbers. They often require estimation or basic number sense.
        Image Examples: Real-world scenarios (e.g., a grocery store receipt, a clock showing time). Objects with quantities (e.g., a stack of books, a row of chairs). Measurements in daily life (e.g., a ruler next to an object, a thermometer).
Instructions:
    Input:
        A textual question related to science, mathematics, or logic.
        An accompanying image that provides context or additional information for the question.
    Output:
        Step 1: Classify the question into one of the 7 categories listed above. Justify your classification based on the textual question and the image.
        Step 2: Provide a detailed description of the image, focusing on extracting visual details that are directly relevant to solving the question. Your description should:
            Identify key elements in the image (e.g., shapes, graphs, diagrams, or objects) that are necessary for solving the problem.
            Explain how these elements relate to the textual question and the identified category.
            Highlight any visual cues or details (e.g., labels, measurements, annotations) that are critical for solving the problem.
            If the image contains text, symbols, or annotations, describe their meaning and relevance to the question.
            The goal is to provide enough detail so that the image description can directly support the next steps in solving the problem.
Example:
    Input:
        Question: "A car travels at a constant speed of 60 km/h. How far will it travel in 3 hours?"
        Image: An image showing A speedometer showing 60 km/h.
    Output:
        Classification: This question falls under the Arithmetic category because it involves calculating distance using speed and time.
        Image Description: The image shows a speedometer with the needle pointing to 60 km/h. The speedometer is a key element because it visually confirms the speed value provided in the question. The labeled scale on the speedometer helps verify the numerical value (60 km/h), which is essential for the calculation. Additionally, the image provides a real-world context, reinforcing the relationship between speed and distance traveled over time.
Here is the input:
    Question: {question}
    Image: <image>
Output:
        """

class Benchmarktest():
    
    def __init__(self,**kwargs) -> None:
        
        self.benchmark = kwargs.get("benchmark")
        self.indicator = kwargs.get("indicator")
        self.model = get_api_client()
        self.dirname = self.benchmark.dirname
        self.model_name = self.model.llm.model_name
        self.benchmark_name = self.benchmark.benchmark_name
        self.exchange_info_df = kwargs.get("exchange_info_df",None)
        
    @property
    def result_name(self):
        return os.path.join(self.dirname, f"{self.model_name}_{self.benchmark_name}_{self.indicator}_result.tsv")
    
    def factory(self):
        match self.indicator:
            case "vic":
                self.vic()
            case "vic_ready":
                self.vic_ready()
            case "vic_m":
                self.vic_m()
            case "switch_info":
                self.info_concate()
                self.switch_info()
            case "original":
                self.original()
            case "extract_claims":
                self.extract_claims()
            case 'cot':
                self.cot()
            case 'init_cot':
                self.init_cot()
            case 'init_cot_parallel':
                self.init_cot_parallel()
            case 'classify':
                self.classify()
            case 'extract_hint':
                self.extract_question_specific_hint()
            case 'rephrase_thinkings':
                self.rephrase_thinkings()
            case 'rephrase_thinkings_new':
                self.rephrase_thinkings_new()
            case 'extract_llm_judge_cot1':
                self.extract_llm_judge_cot1()
            case 'extract_llm_judge_cot2':
                self.extract_llm_judge_cot2()
            case 'extract_llm_judge_cot3':
                self.extract_llm_judge_cot3()
            case 'extract_hallu_detect':
                self.extract_hallu_detect()
            case _:
                raise ValueError("Indicator not supported")
            
    def checking_exiting_file(self):
        if os.path.exists(self.result_name):
            df = import_file(self.result_name)
            console.print(f"Loaded {len(df)} results from {self.result_name}",style="bold green")
            return len(df)
        else:
            console.print(f"Created {self.result_name}",style="bold green")
            return 0
        
    def classify(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = classify_prompt.format(question=row_info['prompt'])
            image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            match = re.search(r'<classification>(.*?)</classification>', response)
            if match:
                response = match.group(1)
            # return None
            row_df.loc[:,'classification'] = response
            self.store(row_df)
    
    def original(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            # prompt = descript_image_prompt.format(question=row_info['prompt'])
            image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            while response.startswith("Connection error.") is True:
                response = self.model.request(input_data)
            row_df.loc[:,f'{self.model_name.split("/")[-1]}_response'] = response
            self.store(row_df)
            import time
            time.sleep(1)
    def extract_claims(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            # response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col][0]
            prompt = extract_claims_prompt.format(cot=row_info['response'])
            # image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt}
            response = self.model.request(input_data)
            while response.startswith("Connection error.") is True:
                response = self.model.request(input_data)
            row_df.loc[:,'extract_claims_pred'] = response
            self.store(row_df)

    def extract_llm_judge_cot1(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            pprint(row_info)
            if isinstance(row_info['response'], float):
                row_info['response'] = "does not get answer."
            if len(str(row_info['thinking_1'])) < 5:
                row_info['thinking_1'] = row_info['thinking']
            row_df = row.left
            # response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col][0]
            prompt = """
### Task Requirement:

You are an expert reasoning chain evaluator. Your job is to score a model-generated Chain of Thought (CoT) based on six key dimensions that assess its correctness, coherence, and hallucination risk. You are provided with:

1. A problem statement.
2. A model-generated Chain of Thought (CoT).
3. One or more reference CoTs (these are valid, possibly diverse ground-truth reasoning paths).

You must evaluate the model's CoT based on the following 6 dimensions. For each dimension, assign a score between 1 and 5, where 5 is best. Also provide a concise explanation for each score.


### Evaluation Dimensions and Scoring Criteria:

**1. Factual Accuracy**  
- 5 = All stated facts, definitions, and formulas are correct.  
- 4 = Minor factual inaccuracies not critical to the reasoning.  
- 3 = At least one notable factual mistake, but the main conclusion still follows.  
- 2 = Several factual inaccuracies that impact the reasoning chain.  
- 1 = Major factual errors or invented knowledge that invalidate the reasoning.

**2. Logical Consistency**  
- 5 = Every step logically follows from the previous one; no contradictions.  
- 4 = Mostly consistent with a minor leap or imprecision.  
- 3 = Some unclear or unjustified transitions between steps.  
- 2 = Multiple reasoning steps are logically invalid.  
- 1 = Reasoning is incoherent or self-contradictory.

**3. Reasoning Completeness**  
- 5 = Fully complete; all key steps are included.  
- 4 = Minor omissions; conclusion still follows.  
- 3 = Missing one important step; slightly weakens the argument.  
- 2 = Missing multiple key steps; reasoning is hard to follow.  
- 1 = Severely incomplete; the reasoning fails to connect to the answer.

**4. Conceptual Reasoning Accuracy**  
- 5 = Correct use of all key mathematical, spatial, or logical concepts.  
- 4 = Small misunderstanding or imprecision.  
- 3 = Misuse of one core concept that weakens reasoning.  
- 2 = Multiple concept-level misapplications.  
- 1 = Misuse or hallucination of core principles or definitions.

**5. Strategy Appropriateness**  
- 5 = The chosen strategy is valid and optimal or close to optimal.  
- 4 = The strategy is valid but suboptimal.  
- 3 = Strategy is unconventional but valid; minor concerns.  
- 2 = The strategy is logically flawed or poorly suited.  
- 1 = The approach is fundamentally invalid for the problem.


### Output Format:
Return your evaluation as a valid JSON object with the following format:


{
  "Factual Accuracy": {
    "score": X,
    "explanation": "..."
  },
  "Logical Consistency": {
    "score": X,
    "explanation": "..."
  },
  "Reasoning Completeness": {
    "score": X,
    "explanation": "..."
  },
  "Conceptual Reasoning Accuracy": {
    "score": X,
    "explanation": "..."
  },
  "Strategy Appropriateness": {
    "score": X,
    "explanation": "..."
  }
}


### Few-shot Evaluation Example:

**Problem:** 
A triangle has angles of 40° and 60°. What is the third angle?  
**Model CoT:**  
"A triangle’s internal angles add up to 180°. So the third angle is 180 - 40 - 60 = 90°. Therefore, the answer is 90°."

**Reference CoT:**  
"Given two angles are 40° and 60°, and the triangle’s angles must sum to 180°, the third angle is 180 - (40 + 60) = 80°. Final answer: 80°."

**LLM Evaluation Output:**

{
  "Factual Accuracy": {
    "score": 4,
    "explanation": "The CoT correctly states that triangle angles sum to 180°, but miscalculates the subtraction."
  },
  "Logical Consistency": {
    "score": 4,
    "explanation": "The reasoning follows a valid logical structure, though it contains a numerical error."
  },
  "Reasoning Completeness": {
    "score": 5,
    "explanation": "All steps from premise to conclusion are present."
  },
  "Conceptual Reasoning Accuracy": {
    "score": 4,
    "explanation": "Correct geometric principle applied; numerical error affects final result."
  },
  "Strategy Appropriateness": {
    "score": 5,
    "explanation": "The subtraction strategy is standard and appropriate."
  }
}

**Problem:** 
If a car travels 120 miles in 2 hours, what is its average speed?  
**Model CoT:**  
"To find speed, we multiply distance by time. So 120 miles * 2 hours = 240 mph. The car's speed is 240 mph."

**Reference CoT:**  
"Average speed is calculated as distance divided by time. Therefore, 120 miles / 2 hours = 60 mph. The car's average speed is 60 mph."

**LLM Evaluation Output:**
{
  "Factual Accuracy": {
    "score": 1,
    "explanation": "Fundamental error in speed calculation formula (multiplied instead of divided)."
  },
  "Logical Consistency": {
    "score": 2,
    "explanation": "While the steps follow internally, they're based on a false premise about the speed formula."
  },
  "Reasoning Completeness": {
    "score": 3,
    "explanation": "Contains all structural steps but implements them incorrectly."
  },
  "Conceptual Reasoning Accuracy": {
    "score": 1,
    "explanation": "Completely misapplies the core concept of speed calculation."
  },
  "Strategy Appropriateness": {
    "score": 1,
    "explanation": "The chosen approach (multiplication) is fundamentally wrong for speed calculation."
  }
}

Now please evaluate the following CoT:

**Problem:** 
""" + row_info['prompt'] + """
**Model CoT:** 
""" + row_info['response'] + """
**Reference CoT:**  
""" + row_info['thinking'] + """

### Important Notes:
1. Be strict and honest during scoring - high scores should be reserved for flawless or near-flawless reasoning.
2. Pay special attention when final answers differ between model and reference CoTs.
3. Use the negative example as a reference for identifying serious flaws in reasoning.
4. DO NOT give high scores unless the CoT fully deserves it across all dimensions.

Return your scores and explanations in the JSON format as shown above.

**Output**

"""
            # prompt = judge_uncertainty_by_llms_prompt.format(prompt=row_info['prompt'], model_cot=row_info['response'], reference_cot=row_info['thinking'])
            # image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt}
            response = self.model.request(input_data)
            while response.startswith("Connection error.") is True:
                response = self.model.request(input_data)
            row_df.loc[:,'llm_hallu_score_json_1'] = response
            self.store(row_df)
    
    def extract_llm_judge_cot2(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            if isinstance(row_info['response'], float):
                row_info['response'] = "does not get answer."
            if len(str(row_info['thinking_1'])) < 5:
                row_info['thinking_1'] = row_info['thinking']
            row_df = row.left
            # response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col][0]
            prompt = """
### Task Requirement:

You are an expert reasoning chain evaluator. Your job is to score a model-generated Chain of Thought (CoT) based on six key dimensions that assess its correctness, coherence, and hallucination risk. You are provided with:

1. A problem statement.
2. A model-generated Chain of Thought (CoT).
3. One or more reference CoTs (these are valid, possibly diverse ground-truth reasoning paths).

You must evaluate the model's CoT based on the following 6 dimensions. For each dimension, assign a score between 1 and 5, where 5 is best. Also provide a concise explanation for each score.


### Evaluation Dimensions and Scoring Criteria:

**1. Factual Accuracy**  
- 5 = All stated facts, definitions, and formulas are correct.  
- 4 = Minor factual inaccuracies not critical to the reasoning.  
- 3 = At least one notable factual mistake, but the main conclusion still follows.  
- 2 = Several factual inaccuracies that impact the reasoning chain.  
- 1 = Major factual errors or invented knowledge that invalidate the reasoning.

**2. Logical Consistency**  
- 5 = Every step logically follows from the previous one; no contradictions.  
- 4 = Mostly consistent with a minor leap or imprecision.  
- 3 = Some unclear or unjustified transitions between steps.  
- 2 = Multiple reasoning steps are logically invalid.  
- 1 = Reasoning is incoherent or self-contradictory.

**3. Reasoning Completeness**  
- 5 = Fully complete; all key steps are included.  
- 4 = Minor omissions; conclusion still follows.  
- 3 = Missing one important step; slightly weakens the argument.  
- 2 = Missing multiple key steps; reasoning is hard to follow.  
- 1 = Severely incomplete; the reasoning fails to connect to the answer.

**4. Conceptual Reasoning Accuracy**  
- 5 = Correct use of all key mathematical, spatial, or logical concepts.  
- 4 = Small misunderstanding or imprecision.  
- 3 = Misuse of one core concept that weakens reasoning.  
- 2 = Multiple concept-level misapplications.  
- 1 = Misuse or hallucination of core principles or definitions.

**5. Strategy Appropriateness**  
- 5 = The chosen strategy is valid and optimal or close to optimal.  
- 4 = The strategy is valid but suboptimal.  
- 3 = Strategy is unconventional but valid; minor concerns.  
- 2 = The strategy is logically flawed or poorly suited.  
- 1 = The approach is fundamentally invalid for the problem.


### Output Format:
Return your evaluation as a valid JSON object with the following format:


{
  "Factual Accuracy": {
    "score": X,
    "explanation": "..."
  },
  "Logical Consistency": {
    "score": X,
    "explanation": "..."
  },
  "Reasoning Completeness": {
    "score": X,
    "explanation": "..."
  },
  "Conceptual Reasoning Accuracy": {
    "score": X,
    "explanation": "..."
  },
  "Strategy Appropriateness": {
    "score": X,
    "explanation": "..."
  }
}


### Few-shot Evaluation Example:

**Problem:** 
A triangle has angles of 40° and 60°. What is the third angle?  
**Model CoT:**  
"A triangle’s internal angles add up to 180°. So the third angle is 180 - 40 - 60 = 90°. Therefore, the answer is 90°."

**Reference CoT:**  
"Given two angles are 40° and 60°, and the triangle’s angles must sum to 180°, the third angle is 180 - (40 + 60) = 80°. Final answer: 80°."

**LLM Evaluation Output:**

{
  "Factual Accuracy": {
    "score": 4,
    "explanation": "The CoT correctly states that triangle angles sum to 180°, but miscalculates the subtraction."
  },
  "Logical Consistency": {
    "score": 4,
    "explanation": "The reasoning follows a valid logical structure, though it contains a numerical error."
  },
  "Reasoning Completeness": {
    "score": 5,
    "explanation": "All steps from premise to conclusion are present."
  },
  "Conceptual Reasoning Accuracy": {
    "score": 4,
    "explanation": "Correct geometric principle applied; numerical error affects final result."
  },
  "Strategy Appropriateness": {
    "score": 5,
    "explanation": "The subtraction strategy is standard and appropriate."
  }
}

**Problem:** 
If a car travels 120 miles in 2 hours, what is its average speed?  
**Model CoT:**  
"To find speed, we multiply distance by time. So 120 miles * 2 hours = 240 mph. The car's speed is 240 mph."

**Reference CoT:**  
"Average speed is calculated as distance divided by time. Therefore, 120 miles / 2 hours = 60 mph. The car's average speed is 60 mph."

**LLM Evaluation Output:**
{
  "Factual Accuracy": {
    "score": 1,
    "explanation": "Fundamental error in speed calculation formula (multiplied instead of divided)."
  },
  "Logical Consistency": {
    "score": 2,
    "explanation": "While the steps follow internally, they're based on a false premise about the speed formula."
  },
  "Reasoning Completeness": {
    "score": 3,
    "explanation": "Contains all structural steps but implements them incorrectly."
  },
  "Conceptual Reasoning Accuracy": {
    "score": 1,
    "explanation": "Completely misapplies the core concept of speed calculation."
  },
  "Strategy Appropriateness": {
    "score": 1,
    "explanation": "The chosen approach (multiplication) is fundamentally wrong for speed calculation."
  }
}

Now please evaluate the following CoT:

**Problem:** 
""" + row_info['prompt'] + """
**Model CoT:** 
""" + row_info['response'] + """
**Reference CoT:**  
""" + row_info['thinking_1'] + """

### Important Notes:
1. Be strict and honest during scoring - high scores should be reserved for flawless or near-flawless reasoning.
2. Pay special attention when final answers differ between model and reference CoTs.
3. Use the negative example as a reference for identifying serious flaws in reasoning.
4. DO NOT give high scores unless the CoT fully deserves it across all dimensions.

Return your scores and explanations in the JSON format as shown above.

**Output**

"""
            # image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt}
            response = self.model.request(input_data)
            while response.startswith("Connection error.") is True:
                response = self.model.request(input_data)
            row_df.loc[:,'llm_hallu_score_json_2'] = response
            self.store(row_df)
    
    def extract_llm_judge_cot3(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            # pprint(row_info)
            # print(type(row_info['response']))
            if isinstance(row_info['response'], float):
                row_info['response'] = "does not get answer."
            if isinstance(row_info['thinking_6'], float):
                row_info['thinking_6'] = row_info['thinking']
            row_df = row.left
            # response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col][0]
            prompt = """
### Task Requirement:

You are an expert reasoning chain evaluator. Your job is to score a model-generated Chain of Thought (CoT) based on six key dimensions that assess its correctness, coherence, and hallucination risk. You are provided with:

1. A problem statement.
2. A model-generated Chain of Thought (CoT).
3. One or more reference CoTs (these are valid, possibly diverse ground-truth reasoning paths).

You must evaluate the model's CoT based on the following 6 dimensions. For each dimension, assign a score between 1 and 5, where 5 is best. Also provide a concise explanation for each score.


### Evaluation Dimensions and Scoring Criteria:

**1. Factual Accuracy**  
- 5 = All stated facts, definitions, and formulas are correct.  
- 4 = Minor factual inaccuracies not critical to the reasoning.  
- 3 = At least one notable factual mistake, but the main conclusion still follows.  
- 2 = Several factual inaccuracies that impact the reasoning chain.  
- 1 = Major factual errors or invented knowledge that invalidate the reasoning.

**2. Logical Consistency**  
- 5 = Every step logically follows from the previous one; no contradictions.  
- 4 = Mostly consistent with a minor leap or imprecision.  
- 3 = Some unclear or unjustified transitions between steps.  
- 2 = Multiple reasoning steps are logically invalid.  
- 1 = Reasoning is incoherent or self-contradictory.

**3. Reasoning Completeness**  
- 5 = Fully complete; all key steps are included.  
- 4 = Minor omissions; conclusion still follows.  
- 3 = Missing one important step; slightly weakens the argument.  
- 2 = Missing multiple key steps; reasoning is hard to follow.  
- 1 = Severely incomplete; the reasoning fails to connect to the answer.

**4. Conceptual Reasoning Accuracy**  
- 5 = Correct use of all key mathematical, spatial, or logical concepts.  
- 4 = Small misunderstanding or imprecision.  
- 3 = Misuse of one core concept that weakens reasoning.  
- 2 = Multiple concept-level misapplications.  
- 1 = Misuse or hallucination of core principles or definitions.

**5. Strategy Appropriateness**  
- 5 = The chosen strategy is valid and optimal or close to optimal.  
- 4 = The strategy is valid but suboptimal.  
- 3 = Strategy is unconventional but valid; minor concerns.  
- 2 = The strategy is logically flawed or poorly suited.  
- 1 = The approach is fundamentally invalid for the problem.


### Output Format:
Return your evaluation as a valid JSON object with the following format:


{
  "Factual Accuracy": {
    "score": X,
    "explanation": "..."
  },
  "Logical Consistency": {
    "score": X,
    "explanation": "..."
  },
  "Reasoning Completeness": {
    "score": X,
    "explanation": "..."
  },
  "Conceptual Reasoning Accuracy": {
    "score": X,
    "explanation": "..."
  },
  "Strategy Appropriateness": {
    "score": X,
    "explanation": "..."
  }
}


### Few-shot Evaluation Example:

**Problem:** 
A triangle has angles of 40° and 60°. What is the third angle?  
**Model CoT:**  
"A triangle’s internal angles add up to 180°. So the third angle is 180 - 40 - 60 = 90°. Therefore, the answer is 90°."

**Reference CoT:**  
"Given two angles are 40° and 60°, and the triangle’s angles must sum to 180°, the third angle is 180 - (40 + 60) = 80°. Final answer: 80°."

**LLM Evaluation Output:**

{
  "Factual Accuracy": {
    "score": 4,
    "explanation": "The CoT correctly states that triangle angles sum to 180°, but miscalculates the subtraction."
  },
  "Logical Consistency": {
    "score": 4,
    "explanation": "The reasoning follows a valid logical structure, though it contains a numerical error."
  },
  "Reasoning Completeness": {
    "score": 5,
    "explanation": "All steps from premise to conclusion are present."
  },
  "Conceptual Reasoning Accuracy": {
    "score": 4,
    "explanation": "Correct geometric principle applied; numerical error affects final result."
  },
  "Strategy Appropriateness": {
    "score": 5,
    "explanation": "The subtraction strategy is standard and appropriate."
  }
}

**Problem:** 
If a car travels 120 miles in 2 hours, what is its average speed?  
**Model CoT:**  
"To find speed, we multiply distance by time. So 120 miles * 2 hours = 240 mph. The car's speed is 240 mph."

**Reference CoT:**  
"Average speed is calculated as distance divided by time. Therefore, 120 miles / 2 hours = 60 mph. The car's average speed is 60 mph."

**LLM Evaluation Output:**
{
  "Factual Accuracy": {
    "score": 1,
    "explanation": "Fundamental error in speed calculation formula (multiplied instead of divided)."
  },
  "Logical Consistency": {
    "score": 2,
    "explanation": "While the steps follow internally, they're based on a false premise about the speed formula."
  },
  "Reasoning Completeness": {
    "score": 3,
    "explanation": "Contains all structural steps but implements them incorrectly."
  },
  "Conceptual Reasoning Accuracy": {
    "score": 1,
    "explanation": "Completely misapplies the core concept of speed calculation."
  },
  "Strategy Appropriateness": {
    "score": 1,
    "explanation": "The chosen approach (multiplication) is fundamentally wrong for speed calculation."
  }
}

Now please evaluate the following CoT:

**Problem:** 
""" + row_info['prompt'] + """
**Model CoT:** 
""" + row_info['response'] + """
**Reference CoT:**  
""" + row_info['thinking_6'] + """

### Important Notes:
1. Be strict and honest during scoring - high scores should be reserved for flawless or near-flawless reasoning.
2. Pay special attention when final answers differ between model and reference CoTs.
3. Use the negative example as a reference for identifying serious flaws in reasoning.
4. DO NOT give high scores unless the CoT fully deserves it across all dimensions.

Return your scores and explanations in the JSON format as shown above.

**Output**

"""
            # image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt}
            response = self.model.request(input_data)
            while response.startswith("Connection error.") is True:
                response = self.model.request(input_data)
            row_df.loc[:,'llm_hallu_score_json_3'] = response
            self.store(row_df)

    def extract_hallu_detect(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            # pprint(row_info)
            if isinstance(row_info['response'], float):
                row_info['response'] = "does not get answer."
            if len(str(row_info['thinking_1'])) < 5:
                row_info['thinking_1'] = row_info['thinking']
            row_df = row.left
            # response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col][0]
            prompt = """
**Task Requirement:**
You are given the output of a multi-modal reasoning model, including a chain-of-thought (CoT) and an input consisting of a question and an accompanying image (represented by the placeholder <image>). You are also provided with a human-verified reference chain-of-thought (Ref-CoT) that correctly answers the question.

Your task is to analyze the predicted CoT and determine whether it contains any hallucinations based on comparison with the Ref-CoT and the input.

**Types of Hallucinations:**

1. **Spatial Hallucination**: Misunderstanding the visual structure, shape, angle, or spatial configuration based on the image.
2. **Logical Hallucination**: Flawed or missing reasoning steps, incorrect logical deductions, or invalid inferences.
3. **Factuality Hallucination**: Statements that contradict known math facts or the image data.
4. **Context Hallucination**: Inconsistencies between steps or contradictions with the predicted answer.
5. **Fabrication Hallucination**: Reasoning steps that are not derived from earlier steps, image data, or question context.

Please output a JSON result with the following fields, enclosed within `<result>` and `</result>` tags:
- `"result"`: Either `"CONFIRM"` or `"HALLUCINATION"`
- `"reason"`: A brief justification for your judgment

---

**Few-shot Examples for Reference:**

**Example 1 (Hallucination – Spatial + Factual):**
- **Image**: AB and CD cross at point O, angle AOE is 90°
- **Question**: As shown in the figure, angle BOD = 50.0, then angle COE = ()
- **Choices**: A:30° B:140° C:50° D:60°
- **Predicted CoT**: “In the given configuration, one finds (by ‘angle chasing’ around point O and using the right angle at O) that ∠BOD and ∠COE end up congruent. Since ∠BOD is 50°, ∠COE must also be 50°.”

**Output**
{
  "Overall": {"result": "HALLUCINATION", "reason": "Spatial relation was misinterpreted, leading to an incorrect angle identity."},
  "Spatial": {"result": "HALLUCINATION", "reason": "Incorrect assumption that ∠BOD ≅ ∠COE."},
  "Logical": {"result": "CONFIRM", "reason": "Logical flow is fine assuming the spatial assumption was correct."},
  "Factuality": {"result": "HALLUCINATION", "reason": "Factual error due to incorrect spatial configuration."},
  "Context": {"result": "CONFIRM", "reason": "Reasoning is self-consistent."},
  "Fabrication": {"result": "CONFIRM", "reason": "No fabricated steps; error comes from misinterpretation."}
}

---

**Example 2 (Hallucination – Logical):**
- **Question**: What is the area of the shaded L-shaped region?
- **Predicted CoT**: “Entire square has side length 5 (1+3+1), so area = 25. The three white squares have areas 1², 3², 1² = 11. So shaded area = 25 − 11 = 14.”

**Output**
{
  "Overall": {"result": "HALLUCINATION", "reason": "Incorrect assumption about which regions are shaded."},
  "Spatial": {"result": "CONFIRM", "reason": "Correct spatial decomposition of square."},
  "Logical": {"result": "HALLUCINATION", "reason": "Shaded region logic was flawed—should only include L-shape."},
  "Factuality": {"result": "CONFIRM", "reason": "No contradictions with known data."},
  "Context": {"result": "CONFIRM", "reason": "Internally consistent reasoning chain."},
  "Fabrication": {"result": "CONFIRM", "reason": "No invented facts or steps; just misapplied logic."}
}

---

**Example 3 (No Hallucination – Positive Example):**
- **Question**: What is \( \angle BAD \)?
- **Predicted CoT**:  
  "Since triangle ABC is isosceles with AB = AC and angle BAC = 40°, angles ABC and ACB must be equal. The triangle angle sum is 180°, so:  
  \( \angle ABC = \angle ACB = (180° - 40°) / 2 = 70° \).  
  Now, AD is the angle bisector of \( \angle BAC = 40° \), so it splits this angle into two equal parts:  
  \( \angle BAD = 40° / 2 = 20° \)."

**Output**
{
  "Overall": {"result": "CONFIRM", "reason": "All reasoning steps are logically and spatially valid and match the visual context."},
  "Spatial": {"result": "CONFIRM", "reason": "Correct understanding of the triangle's symmetry and angle positions."},
  "Logical": {"result": "CONFIRM", "reason": "Accurate use of triangle angle sum and angle bisector properties."},
  "Factuality": {"result": "CONFIRM", "reason": "No contradictions with mathematical facts or visual input."},
  "Context": {"result": "CONFIRM", "reason": "Each step supports the next; no inconsistencies found."},
  "Fabrication": {"result": "CONFIRM", "reason": "No steps are invented; all are well-supported by geometry and prior steps."}
}

---

**Now, for the following input, analyze the reasoning chain and provide your final evaluation according to the schema above.**
**Rather than original image input, you have reference ground-truth CoT as guidance to judge the reasoning hallucination**
"""+f"""
**Input:**
- **Question**: {row_info['prompt']}
- **Predicted CoT**: {row_info['response']}
- **Reference CoT**: {row_info['thinking']}
""" + """
**Output format:**
<result>
{
  "Overall": {
    "result": "CONFIRM" or "HALLUCINATION",
    "reason": "Overall explanation here..."
  },
  "Spatial": {
    "result": "CONFIRM" or "HALLUCINATION",
    "reason": "Detailed explanation for Spatial hallucination..."
  },
  "Logical": {
    "result": "CONFIRM" or "HALLUCINATION",
    "reason": "Detailed explanation for Logical hallucination..."
  },
  "Factuality": {
    "result": "CONFIRM" or "HALLUCINATION",
    "reason": "Detailed explanation for Factuality hallucination..."
  },
  "Context": {
    "result": "CONFIRM" or "HALLUCINATION",
    "reason": "Detailed explanation for Context hallucination..."
  },
  "Fabrication": {
    "result": "CONFIRM" or "HALLUCINATION",
    "reason": "Detailed explanation for Fabrication hallucination..."
  }
}
</result>

**Output**

"""
            # prompt = judge_uncertainty_by_llms_prompt.format(prompt=row_info['prompt'], model_cot=row_info['response'], reference_cot=row_info['thinking'])
            # image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt}
            response = self.model.request(input_data)
            while response.startswith("Connection error.") is True:
                response = self.model.request(input_data)
            row_df.loc[:,'llm_hallu_detect'] = response
            self.store(row_df)

    def rephrase_thinkings(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            # cot_prompt = 'Answer the following question step by step based on image. Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.' #You should also focus on the information from image. Explain each step of your reasoning process clearly and thoroughly before arriving at the final answer. Even if the problem seems simple, break it down into small, logical steps to ensure clarity and correctness.'
            prompt = rephrase_prompt.format(prompt=row_info['prompt'], thinking=row_info['thinking'], N=str(5))
            # classification = row_info['classification']
            # topic_specific_hint = topic_specific_prompts[classification]
            # prompt = prompt + f"\nHere we know this is a {classification} problem, hence we provide the topic specific hint as follows to help thinking: {topic_specific_hint}\n"
            # image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt}
            response = self.model.request(input_data)
            pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
            steps = pattern.findall(response)
            # steps = [f"<step>{c}</step>" for c in steps]
            for i in range(len(steps)):
                row_df.loc[:,f'rephrase_thinking_{i}'] = steps[i]
            self.store(row_df)
    
    def rephrase_thinkings_new(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            # cot_prompt = 'Answer the following question step by step based on image. Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.' #You should also focus on the information from image. Explain each step of your reasoning process clearly and thoroughly before arriving at the final answer. Even if the problem seems simple, break it down into small, logical steps to ensure clarity and correctness.'
            prompt = rephrase_prompt.format(prompt=row_info['prompt'], thinking=row_info['thinking'], N=str(5))
            # classification = row_info['classification']
            # topic_specific_hint = topic_specific_prompts[classification]
            # prompt = prompt + f"\nHere we know this is a {classification} problem, hence we provide the topic specific hint as follows to help thinking: {topic_specific_hint}\n"
            # image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt}
            response = self.model.request(input_data)
            row_df.loc[:,f'rephrase_thinking_raw'] = response
            # pattern = re.compile(r'<step>(.*?)</step>', re.DOTALL)
            # steps = pattern.findall(response)
            pattern = re.compile(r"\*\*Variant (\d+):\*\*\n(.*?)(?=\n\*\*Variant \d+:|\Z)", re.DOTALL)
            matches = pattern.findall(response)          
            steps = []
            for variant_id, content in matches:
                steps.append(content.strip())
            # steps = [f"<step>{c}</step>" for c in steps]
            # for i in range(len(steps)):
            #     row_df.loc[:,f'rephrase_thinking_{i}'] = steps[i]
            self.store(row_df)
    
    def cot(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            cot_prompt = 'Answer the following question step by step based on image. Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.' #You should also focus on the information from image. Explain each step of your reasoning process clearly and thoroughly before arriving at the final answer. Even if the problem seems simple, break it down into small, logical steps to ensure clarity and correctness.'
            prompt = cot_prompt+row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            row_df.loc[:,f'{self.model_name.split("/")[-1]}_response'] = response
            self.store(row_df)

    def reflection(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            cot_prompt = 'Answer the following question step by step based on image. Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags. After thinking the question, please re-evaulate the thinking process, and output the new answer in <answer> </answer> tags after the reflection procedure.' #You should also focus on the information from image. Explain each step of your reasoning process clearly and thoroughly before arriving at the final answer. Even if the problem seems simple, break it down into small, logical steps to ensure clarity and correctness.'
            prompt = cot_prompt+row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            row_df.loc[:,f'{self.model_name.split("/")[-1]}_response'] = response
            self.store(row_df)
    
    def cot_topic_hint(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            cot_prompt = 'Answer the following question step by step based on image. Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.' #You should also focus on the information from image. Explain each step of your reasoning process clearly and thoroughly before arriving at the final answer. Even if the problem seems simple, break it down into small, logical steps to ensure clarity and correctness.'
            prompt = cot_prompt+row_info['prompt']
            classification = row_info['classification']
            topic_specific_hint = topic_specific_prompts[classification]
            prompt = prompt + f"\nHere we know this is a {classification} problem, hence we provide the topic specific hint as follows to help thinking: {topic_specific_hint}\n"
            image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            row_df.loc[:,f'{self.model_name.split("/")[-1]}_response'] = response
            self.store(row_df)

    def cot_question_hint(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            cot_prompt = 'Answer the following question step by step based on image. Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.' #You should also focus on the information from image. Explain each step of your reasoning process clearly and thoroughly before arriving at the final answer. Even if the problem seems simple, break it down into small, logical steps to ensure clarity and correctness.'
            prompt = cot_prompt+row_info['prompt']
            question_specific_hint = row_info['question_hint']
            prompt = prompt + f"\nWe provide the topic specific hint as follows to help thinking: {question_specific_hint}\n"
            image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            row_df.loc[:,f'{self.model_name.split("/")[-1]}_response'] = response
            self.store(row_df)

    def cot_dual_hint(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            cot_prompt = 'Answer the following question step by step based on image. Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.' #You should also focus on the information from image. Explain each step of your reasoning process clearly and thoroughly before arriving at the final answer. Even if the problem seems simple, break it down into small, logical steps to ensure clarity and correctness.'
            prompt = cot_prompt+row_info['prompt']
            question_specific_hint = row_info['question_hint']
            classification = row_info['classification']
            topic_specific_hint = topic_specific_prompts[classification]
            prompt = prompt + f"\nHere we know this is a {classification} problem, hence we provide the topic specific hint as follows to help thinking: {topic_specific_hint}\n And for this question, we also provide question-specific hint to help thinking: {question_specific_hint}. \n Now, answer the question: {row_info['prompt']}"
            image = [to_base64_image(row_info['image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            row_df.loc[:,f'{self.model_name.split("/")[-1]}_response'] = response
            self.store(row_df)
    
    def init_cot(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            # print(row[0])
            row_info = row[0]
            row_df = row.left
            cot_prompt = 'You are a helpful assistant good at solving problems with step-by-step reasoning. You should first describe the input image as detailed as possible, then **step-by-step** thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$ with \\\\boxed{answer}.The reasoning process and answer are enclosed within <desp></desp> <think> </think> and <answer> </answer> tags, respectively, i.e.,<desp>This image shows a arithmetic question of 1+1=?.</desp> <think> Since $1+1=2$, so the answer is $2$. </think><answer> $\\\\boxed{2}$ </answer>, which means your output should start with <desp>, include <think> and end with </answer>. Question: ' #You should also focus on the information from image. Explain each step of your reasoning process clearly and thoroughly before arriving at the final answer. Even if the problem seems simple, break it down into small, logical steps to ensure clarity and correctness.'
            prompt = cot_prompt+row_info['prompt'].strip()
            try:
                image = [to_base64_image(row_info['image'])]
            except:
                image = [to_base64_image(row_info['decoded_image'])]
            input_data = {'query':prompt,'image':image}
            response = self.model.request(input_data)
            row_df.loc[:,f'{self.model_name}_init_thinking'] = response
            self.store(row_df)
    
    def extract_question_specific_hint(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            # response_column = [col for col in self.df.columns if 'response' in col and 'intermediate' not in col][0]
            prompt = extract_question_specific_hint_prompt.format(prompt=row_info['prompt'])
            image = [to_base64_image(row_info['image'])]
            classification = row_info['classification']
            topic_specific_hint = topic_specific_prompts[classification]
            input_data = {'query':prompt, 'image': image}
            response = self.model.request(input_data)
            while response.startswith("Connection error.") is True:
                response = self.model.request(input_data)
            row_df.loc[:, 'topic_specific_hint'] = topic_specific_hint
            row_df.loc[:,'question_specific_hint'] = response
            self.store(row_df)
            
    def init_cot_parallel(self, num_workers=12):
        """
        Parallel version of init_cot using ThreadPoolExecutor for concurrent processing
        while maintaining sequential output order.
        
        Args:
            num_workers (int): Number of parallel workers
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from threading import Lock
        import queue
        
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        # Create a lock for file writing
        file_lock = Lock()
        # Create a queue to store results in order
        result_queue = queue.Queue()
        # Current index for maintaining order
        current_idx = 0
        
        def process_row(idx, row):
            """Process a single row and return result with its index"""
            row_info = row[0]
            row_df = row.left
            cot_prompt = 'You are a helpful assistant good at solving problems with step-by-step reasoning. You should first describe the input image as detailed as possible, then **step-by-step** thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$ with \\boxed{answer}.The reasoning process and answer are enclosed within <desp></desp> <think> </think> and <answer> </answer> tags, respectively, i.e.,<desp>This image shows a arithmetic question of 1+1=?.</desp> <think> Since $1+1=2$, so the answer is $2$. </think><answer> $\\boxed{2}$ </answer>, which means your output should start with <desp>, include <think> and end with </answer>. Question: '
            prompt = cot_prompt + row_info['prompt'].strip()
            try:
                image = [to_base64_image(row_info['image'])]
            except:
                image = [to_base64_image(row_info['decoded_image'])]
            
            input_data = {'query': prompt, 'image': image}
            try:
                response = self.model.request(input_data)
                row_df.loc[:, f'{self.model_name}_init_thinking'] = response
                return idx, row_df
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                return idx, None
        
        def store_results():
            """Store results in order when they become available"""
            nonlocal current_idx
            while not result_queue.empty():
                idx, row_df = result_queue.get()
                if idx == current_idx and row_df is not None:
                    with file_lock:
                        self.store(row_df)
                    current_idx += 1
                    # Continue checking next items that might be ready
                    store_results()
        
        print(f"Processing {len(benchmark)} rows with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(process_row, idx, row): idx 
                for idx, row in enumerate(benchmark.iterrows())
            }
            
            # Process completed tasks
            for future in tqdm(as_completed(future_to_idx), total=len(benchmark)):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    result_queue.put(result)
                    # Try to store any results that are ready
                    store_results()
                except Exception as e:
                    print(f"Error processing future {idx}: {str(e)}")
        
        # Final check for any remaining results
        store_results()
        print("Parallel processing completed.")
    
    def vic(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            response = Vic(query=prompt,image=image)
            row_df.loc[:,f'visual_chain'] = str(response['visual_inference_chain'])
            row_df.loc[:,f'{self.model_name}_intermediate_response'] = response['intermediate_response']
            row_df.loc[:,f'{self.model_name}_response'] = response['response']
            self.store(row_df)
            
    def vic_ready(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            # print(type(row_info['image']))
            image = [to_base64_image(row_info['image'])]
            visual_chain = row.visual_chain.iloc[0]
            response = Vic(query=prompt,image=image,vic=visual_chain)
            row_df.loc[:,f'visual_chain'] = str(response['visual_inference_chain'])
            row_df.loc[:,f'{self.model_name}_intermediate_response'] = response['intermediate_response']
            row_df.loc[:,f'{self.model_name}_response'] = response['response']
            self.store(row_df)
            
    def vic_m(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            visual_chain = row.visual_chain.iloc[0]
            response = Vic(query=prompt,image=image,vic=visual_chain,vic_m=True)
            row_df.loc[:,f'visual_chain'] = str(response['visual_inference_chain'])
            row_df.loc[:,f'{self.model_name}_intermediate_response'] = response['intermediate_response']
            row_df.loc[:,f'{self.model_name}_response'] = response['response']
            self.store(row_df)
            
    def info_concate(self):
        if self.exchange_info_df is None:
            raise ValueError("exchange_info_df is required")
        exchange_info_column = [col for col in self.exchange_info_df.columns if 'intermediate_response' in col]
        self.benchmark.df['exchange_info'] = self.exchange_info_df[exchange_info_column].values.tolist()
        
    def switch_info(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            row_info = row[0]
            row_df = row.left
            prompt = row_info['prompt']
            image = [to_base64_image(row_info['image'])]
            visual_chain = row.visual_chain.iloc[0]
            exchange_info = row.df.exchange_info.iloc[0]
            response = Vic(query=prompt,image=image,vic=visual_chain,extract_info=exchange_info)
            row_df.loc[:,f'visual_chain'] = str(response['visual_inference_chain'])
            row_df.loc[:,f'{self.model_name}_intermediate_response'] = response['intermediate_response']
            row_df.loc[:,f'{self.model_name}_response'] = response['response']
            self.store(row_df)
            
    def store(self,row_df):
        if not os.path.exists(self.result_name):
            row_df.to_csv(self.result_name,sep='\t',index=False)
        else:
            row_df.to_csv(self.result_name,sep='\t',mode='a',index=False,header=False)
            
        console.print(f"Stored 1 result in {self.result_name}",style="bold green")
        
            
        
        