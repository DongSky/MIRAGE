import os
from tqdm import tqdm
from rich.console import Console
from ..LLM.LLM_state import get_api_client
from ..utils.file import import_file
from ..Vic.Vic_main import Vic, Vic_async
from ..utils.file import to_base64_image
from ..Vic.visual_inference_prompt import classify_prompt
import re
import pandas as pd
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
            case 'cot':
                self.cot()
            case 'init_cot':
                self.init_cot()
            case 'refine_cot_by_answer':
                self.refine_cot_by_answer()
            case 'init_cot_parallel':
                self.init_cot_parallel()
            case 'classify':
                self.classify()
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
            # 获取行索引信息和所有列数据
            idx, row_data = row
            row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            
            # 获取图像，支持多种可能的图像字段名
            image = None
            for img_field in ['image', 'decoded_image']:
                if img_field in row_data and pd.notna(row_data[img_field]):
                    image = [to_base64_image(row_data[img_field])]
                    break
            
            if prompt and image:
                formatted_prompt = classify_prompt.format(question=prompt)
                input_data = {'query': formatted_prompt, 'image': image}
                response = self.model.request(input_data)
                match = re.search(r'<classification>(.*?)</classification>', response)
                if match:
                    response = match.group(1)
                # 将结果添加到行数据中
                row_df.loc[idx, 'classification'] = response
                self.store(row_df)
            else:
                console.print(f"Skip row {idx}: missing prompt or image", style="bold red")
    
    def original(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            # 获取行索引信息和所有列数据
            idx, row_data = row
            row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            
            # 获取图像，支持多种可能的图像字段名
            image = None
            for img_field in ['image', 'decoded_image']:
                if img_field in row_data and pd.notna(row_data[img_field]):
                    image = [to_base64_image(row_data[img_field])]
                    break
            
            if prompt and image:
                formatted_prompt = descript_image_prompt.format(question=prompt)
                input_data = {'query': formatted_prompt, 'image': image}
                response = self.model.request(input_data)
                # 将结果添加到行数据中
                row_df.loc[idx, f'{self.model_name}_describe'] = response
                self.store(row_df)
            else:
                console.print(f"Skip row {idx}: missing prompt or image", style="bold red")
    
    def cot(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            # 获取行索引信息和所有列数据
            idx, row_data = row
            row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            
            # 获取图像，支持多种可能的图像字段名
            image = None
            for img_field in ['image', 'decoded_image']:
                if img_field in row_data and pd.notna(row_data[img_field]):
                    image = [to_base64_image(row_data[img_field])]
                    break
            
            if prompt and image:
                cot_prompt = 'Answer the following question step by step based on image. Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags.'
                formatted_prompt = cot_prompt + prompt
                input_data = {'query': formatted_prompt, 'image': image}
                response = self.model.request(input_data)
                # 将结果添加到行数据中
                row_df.loc[idx, f'{self.model_name}_response'] = response
                self.store(row_df)
            else:
                console.print(f"Skip row {idx}: missing prompt or image", style="bold red")
    
    def init_cot(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            # 获取行索引信息和所有列数据
            idx, row_data = row
            row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            if pd.isna(prompt) or prompt == '':
                console.print(f"Skip row {idx}: empty prompt", style="bold red")
                continue
                
            # 获取图像，支持多种可能的图像字段名
            image = None
            for img_field in ['image', 'decoded_image']:
                if img_field in row_data and pd.notna(row_data[img_field]):
                    try:
                        image = [to_base64_image(row_data[img_field])]
                        break
                    except Exception as e:
                        console.print(f"Error processing image in field {img_field}: {str(e)}", style="bold red")
            
            if not image:
                console.print(f"Skip row {idx}: missing image", style="bold red")
                continue
                
            # 继续处理有效的数据
            cot_prompt = 'You are a helpful assistant good at solving problems with step-by-step reasoning. You should first describe the input image as detailed as possible, then **step-by-step** thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$ with \\\\boxed{answer}.The reasoning process and answer are enclosed within <desp></desp> <think> </think> and <answer> </answer> tags, respectively, i.e.,<desp>This image shows a arithmetic question of 1+1=?.</desp> <think> Since $1+1=2$, so the answer is $2$. </think><answer> $\\\\boxed{2}$ </answer>, which means your output should start with <desp>, include <think> and end with </answer>. Question: '
            formatted_prompt = cot_prompt + prompt.strip()
            input_data = {'query': formatted_prompt, 'image': image}
            
            try:
                response = self.model.request(input_data)
                # 将结果添加到行数据中
                row_df.loc[idx, f'{self.model_name}_init_thinking'] = response
                self.store(row_df)
            except Exception as e:
                console.print(f"Error processing row {idx}: {str(e)}", style="bold red")
    
    def refine_cot_by_answer(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            # 获取行索引信息和所有列数据
            # print(row[0])
            row_data = row[0]
            row_df = row.left
            # row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            if pd.isna(prompt) or prompt == '':
                console.print(f"Skip row: empty prompt", style="bold red")
                continue
            
            answer = row_data.get('answer', '')
            if pd.isna(answer) or answer == '':
                console.print(f"Skip row: empty answer", style="bold red")
                continue
            
            # 获取图像，支持多种可能的图像字段名
            # image = None
            # for img_field in ['image', 'decoded_image']:
            #     if img_field in row_data and pd.notna(row_data[img_field]):
            #         try:
            #             image = [to_base64_image(row_data[img_field])]
            #             break
            #         except Exception as e:
            #             console.print(f"Error processing image in field {img_field}: {str(e)}", style="bold red")
            
            # if not image:
            #     console.print(f"Skip row {idx}: missing image", style="bold red")
            #     continue

            original_img_desp = row_data.get('desp', '')
            if pd.isna(original_img_desp) or original_img_desp == '':
                console.print(f"Skip row: missing original_img_desp", style="bold red")
                continue

            init_cot = row_data.get('init_thinking', '')
            if pd.isna(init_cot) or init_cot == '':
                console.print(f"Skip row: missing init_cot", style="bold red")
                continue
                
            # 继续处理有效的数据
            # cot_prompt = 'You are a helpful assistant good at solving problems with step-by-step reasoning. You should first describe the input image as detailed as possible, then **step-by-step** thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$ with \\\\boxed{answer}.The reasoning process and answer are enclosed within <desp></desp> <think> </think> and <answer> </answer> tags, respectively, i.e.,<desp>This image shows a arithmetic question of 1+1=?.</desp> <think> Since $1+1=2$, so the answer is $2$. </think><answer> $\\\\boxed{2}$ </answer>, which means your output should start with <desp>, include <think> and end with </answer>. Question: '

            cot_prompt = f"""
            ### TASK REWUUIREMENT ###

            You are an excellent AI assistant, skilled in reasoning and critical thinking, and particularly good at identifying and correcting hallucinations or other logical errors that may occur during the reasoning process. Since you cannot directly read image data, we will provide the following:
                0) The problem statement and its corresponding answer.

                1) A textual image description generated by another large model.

                2) An initial chain of thought and answer provided by another multimodal reasoning model.

            Your task is to output a corrected chain of thought and final answer. If the initial chain of thought contains no errors, it should be returned as-is. If there are any errors, they should be corrected, and the revised chain of thought and answer should be returned. The prompt format should be clear and highly readable.

            ### INPUT ###

            Question: {prompt}
            Answer: {answer}
            Original image description: {original_img_desp}
            Initial chain of thought and answer: {init_cot}
            
            ### OUTPUT ###


            """

            formatted_prompt = cot_prompt + prompt.strip()
            input_data = {'query': formatted_prompt}
            
            try:
                response = self.model.request(input_data)
                # 将结果添加到行数据中
                # row_df.loc[idx, f'refine_thinking'] = response
                # self.store(row_df)
                print(response)
                row_df.loc[:,f'refine_thinking'] = response
                self.store(row_df)
            except Exception as e:
                console.print(f"Error processing row: {str(e)}", style="bold red")
    
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
        
        def process_row(idx, row_tuple):
            """Process a single row and return result with its index"""
            # 获取行索引信息和所有列数据
            row_idx, row_data = row_tuple
            row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            if pd.isna(prompt) or prompt == '':
                print(f"Skip row {row_idx}: empty prompt")
                return idx, None
                
            # 获取图像，支持多种可能的图像字段名
            image = None
            for img_field in ['image', 'decoded_image']:
                if img_field in row_data and pd.notna(row_data[img_field]):
                    try:
                        image = [to_base64_image(row_data[img_field])]
                        break
                    except Exception as e:
                        print(f"Error processing image in field {img_field}: {str(e)}")
            
            if not image:
                print(f"Skip row {row_idx}: missing image")
                return idx, None
                
            # 继续处理有效的数据
            cot_prompt = 'You are a helpful assistant good at solving problems with step-by-step reasoning. You should first describe the input image as detailed as possible, then **step-by-step** thinks about the reasoning process in the mind and then provides the user with the answer. Your answer must be in latex format and wrapped in $...$ with \\boxed{answer}.The reasoning process and answer are enclosed within <desp></desp> <think> </think> and <answer> </answer> tags, respectively, i.e.,<desp>This image shows a arithmetic question of 1+1=?.</desp> <think> Since $1+1=2$, so the answer is $2$. </think><answer> $\\boxed{2}$ </answer>, which means your output should start with <desp>, include <think> and end with </answer>. Question: '
            formatted_prompt = cot_prompt + prompt.strip()
            input_data = {'query': formatted_prompt, 'image': image}
            
            try:
                response = self.model.request(input_data)
                # 将结果添加到行数据中
                row_df.loc[row_idx, f'{self.model_name}_init_thinking'] = response
                return idx, row_df
            except Exception as e:
                print(f"Error processing row {row_idx}: {str(e)}")
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
            # 获取行索引信息和所有列数据
            idx, row_data = row
            row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            
            # 获取图像，支持多种可能的图像字段名
            image = None
            for img_field in ['image', 'decoded_image']:
                if img_field in row_data and pd.notna(row_data[img_field]):
                    image = [to_base64_image(row_data[img_field])]
                    break
            
            if prompt and image:
                # 处理其他可能需要的字段
                additional_params = {}
                
                # 执行Vic函数
                response = Vic(query=prompt, image=image, **additional_params)
                
                # 将结果添加到行数据中
                row_df.loc[idx, 'visual_chain'] = str(response['visual_inference_chain'])
                row_df.loc[idx, f'{self.model_name}_intermediate_response'] = response['intermediate_response']
                row_df.loc[idx, f'{self.model_name}_response'] = response['response']
                self.store(row_df)
            else:
                console.print(f"Skip row {idx}: missing prompt or image", style="bold red")
            
    def vic_ready(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            # 获取行索引信息和所有列数据
            idx, row_data = row
            row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            visual_chain = row_data.get('visual_chain', None)
            
            # 获取图像，支持多种可能的图像字段名
            image = None
            for img_field in ['image', 'decoded_image']:
                if img_field in row_data and pd.notna(row_data[img_field]):
                    image = [to_base64_image(row_data[img_field])]
                    break
            
            if prompt and image and visual_chain:
                # 执行Vic函数
                response = Vic(query=prompt, image=image, vic=visual_chain)
                
                # 将结果添加到行数据中
                row_df.loc[idx, 'visual_chain'] = str(response['visual_inference_chain'])
                row_df.loc[idx, f'{self.model_name}_intermediate_response'] = response['intermediate_response']
                row_df.loc[idx, f'{self.model_name}_response'] = response['response']
                self.store(row_df)
            else:
                console.print(f"Skip row {idx}: missing prompt, image or visual_chain", style="bold red")
            
    def vic_m(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            # 获取行索引信息和所有列数据
            idx, row_data = row
            row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            visual_chain = row_data.get('visual_chain', None)
            
            # 获取图像，支持多种可能的图像字段名
            image = None
            for img_field in ['image', 'decoded_image']:
                if img_field in row_data and pd.notna(row_data[img_field]):
                    image = [to_base64_image(row_data[img_field])]
                    break
            
            if prompt and image and visual_chain:
                # 执行Vic函数
                response = Vic(query=prompt, image=image, vic=visual_chain, vic_m=True)
                
                # 将结果添加到行数据中
                row_df.loc[idx, 'visual_chain'] = str(response['visual_inference_chain'])
                row_df.loc[idx, f'{self.model_name}_intermediate_response'] = response['intermediate_response']
                row_df.loc[idx, f'{self.model_name}_response'] = response['response']
                self.store(row_df)
            else:
                console.print(f"Skip row {idx}: missing prompt, image or visual_chain", style="bold red")
            
    def info_concate(self):
        if self.exchange_info_df is None:
            raise ValueError("exchange_info_df is required")
        exchange_info_column = [col for col in self.exchange_info_df.columns if 'intermediate_response' in col]
        self.benchmark.df['exchange_info'] = self.exchange_info_df[exchange_info_column].values.tolist()
        
    def switch_info(self):
        existing_len = self.checking_exiting_file()
        benchmark = self.benchmark[existing_len:]
        
        for row in tqdm(benchmark.iterrows(),total=len(benchmark)):
            # 获取行索引信息和所有列数据
            idx, row_data = row
            row_df = row_data.to_frame().T
            
            # 获取必要字段，确保即使没有这些字段也能适当处理
            prompt = row_data.get('prompt', '')
            visual_chain = row_data.get('visual_chain', None)
            exchange_info = row_data.get('exchange_info', None)
            
            # 获取图像，支持多种可能的图像字段名
            image = None
            for img_field in ['image', 'decoded_image']:
                if img_field in row_data and pd.notna(row_data[img_field]):
                    image = [to_base64_image(row_data[img_field])]
                    break
            
            if prompt and image and visual_chain and exchange_info:
                # 执行Vic函数
                response = Vic(query=prompt, image=image, vic=visual_chain, extract_info=exchange_info)
                
                # 将结果添加到行数据中
                row_df.loc[idx, 'visual_chain'] = str(response['visual_inference_chain'])
                row_df.loc[idx, f'{self.model_name}_intermediate_response'] = response['intermediate_response']
                row_df.loc[idx, f'{self.model_name}_response'] = response['response']
                self.store(row_df)
            else:
                missing = []
                if not prompt: missing.append("prompt")
                if not image: missing.append("image")
                if not visual_chain: missing.append("visual_chain")
                if not exchange_info: missing.append("exchange_info")
                console.print(f"Skip row {idx}: missing {', '.join(missing)}", style="bold red")
            
    def store(self,row_df):
        if not os.path.exists(self.result_name):
            row_df.to_csv(self.result_name,sep='\t',index=False)
        else:
            row_df.to_csv(self.result_name,sep='\t',mode='a',index=False,header=False)
            
        console.print(f"Stored 1 result in {self.result_name}",style="bold green") 