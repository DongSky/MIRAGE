extract_claims_prompt = """
**Task**  
Given a reasoning chain, extract the key **steps** (critical reasoning or calculation steps) and **claims** (correct intermediate results or derived facts). Follow these guidelines for each problem type:  

1. **Geometry**: Angles, lengths, area/volume formulas, congruence/similarity, geometric theorems (e.g., Pythagorean theorem).  
2. **Arithmetic**: Critical numerical calculations (e.g., division, fractions) or multi-step operations.  
3. **Algebra**: Equations, inequalities, variable substitutions, function definitions, or computed variable values.  
4. **Spatial Reasoning**: Spatial transformations (rotation, reflection), 3D object properties, coordinate systems.  
5. **Logical Patterns (Rule Identification)**: Explicit rules (e.g., "add 3"), sequence relationships, or logical operators.  
6. **Science (Physics/Chemistry/Biology)**: Scientific laws, formulas (e.g., F=ma), experimental steps, chemical reactions, biological processes.  
7. **Statistics (Chart/Graph Problems)**: Data trends, statistical measures (mean, median), chart interpretations, probability rules.  

**If the reasoning chain is incomplete**, treat *"the answer is [answer in the input]"* as both a step and a claim.  

**If the reasoning chain is too long, output AT MOST 10 most important (i.e., critical to solve the problem) steps and AT MOST 10 most important (i.e., critical to solve the problem) claims in the chain. 

**Formatting:**  
- Wrap each step with `<step>...</step>`.  
- Wrap each claim with `<claim>...</claim>`.  

---  

**Examples:**  

**Example 1 (Geometry):**  
*Input:* "The triangle has sides 3, 4, 5. By Pythagoras, it’s right-angled. Area = (3×4)/2 = 6. The answer is 6."  
*Output:*  
`<step>Apply Pythagorean theorem to verify right angle</step>`  
`<claim>3² + 4² = 5² → Right-angled triangle</claim>`  
`<step>Calculate area: (3×4)/2</step>`  
`<claim>Area = 6</claim>`  
`<step>the answer is 6</step>`  
`<claim>the answer is 6</claim>`  

**Example 2 (Logical Pattern):**  
*Input:* "Sequence: 3, 6, 9, 12. Each term increases by 3. The answer is 15."  
*Output:*  
`<step>Identify pattern: increase by 3</step>`  
`<claim>Rule: Add 3 to previous term</claim>`  
`<step>the answer is 15</step>`  
`<claim>the answer is 15</claim>`  

**Example 3 (Statistics):**  
*Input:* "The bar chart shows sales peaking at 200 units in July. Mean = (100+150+200)/3 = 150. The answer is 150."  
*Output:*  
`<step>Identify peak sales from chart: 200 units in July</step>`  
`<claim>Peak value = 200</claim>`  
`<step>Calculate mean: (100+150+200)/3</step>`  
`<claim>Mean = 150</claim>`  
`<step>the answer is 150</step>`  
`<claim>the answer is 150</claim>`  

---  

Now, process the following input:  

*Input:* {cot}  
*Output:* 

"""


judge_f1_score_prompt = """
**Task:**

THINK STEP BY STEP and Analyze two sets of claims:  
1. **Model-generated claims** (between `<claims>` tags)  
2. **Ground-truth claims** (between `<claims>` tags)  
3. **Output the answer, then provide additional analysis to enhance the accuracy**

Process:  
1. Extract all model claims wrapped in `<claim>` tags from the first `<claims>` block.  
2. Extract all ground-truth (GT) claims wrapped in `<claim>` tags from the second `<claims>` block.  
3. Compare each model claim against all GT claims:  
   - Label as **MATCH** if it semantically aligns with **any** GT claim.  
   - Label as **CONFLICT** if no GT claim matches.  
4. Compare each GT claim against all model claims:  
   - Label as **MATCH** if it is covered by **any** model claim.  
   - Label as **CONFLICT** if no model claim matches.  
5. Calculate:  
   - **Precision**: `(True MATCH in model claims) / (Total model claims marked MATCH)`  
   - **Recall**: `(True MATCH in GT claims) / (Total GT claims)`  
   - **F1**: `2 * (Precision * Recall) / (Precision + Recall)`  

**Output Format:**  
```xml  
<pred_match>MATCH,CONFLICT,...</pred_match>  
<gt_match>MATCH,CONFLICT,...</gt_match>  
<precision>0.XX</precision>  
<recall>0.XX</recall>  
<f1>0.XX</f1>  
```  

**Input Example:** 
Prediction from Model:   
<claims>  
<claim>Earth orbits the Sun.</claim>  
<claim>Water boils at 90°C.</claim>  
</claims>  

Ground-Truth:
<claims>  
<claim>The Earth revolves around the Sun.</claim>  
<claim>Water boils at 100°C at sea level.</claim>  
</claims>  


**Output Example:**  
<pred_match>MATCH,CONFLICT</pred_match>  
<gt_match>MATCH,CONFLICT</gt_match>  
<precision>0.50</precision>  
<recall>0.50</recall>  
<f1>0.50</f1>  


**Input:**
Prediction from Model:   
<claims>  
[pred_model]  
</claims>  

Ground-Truth:
<claims>  
[gt_claims]
</claims> 

**Output:**

"""

extract_specific_hint_prompt = """
You are an expert instructor guiding learners through complex problem-solving processes. When presented with an image and a related [question/task description], follow this structured approach:

1. **Visual Context Interpretation**  
   - Briefly describe key elements in the image relevant to solving the problem  
   - Identify potential relationships between visual components  

2. **Problem Deconstruction**  
   - Break down the given question into 2-3 core sub-problems  
   - Highlight any implicit assumptions or constraints  

3. **Knowledge Domain Mapping**  
   - List 2-3 relevant academic/professional domains connected to the problem  
   - Specify which concepts/principles from these domains might apply  

4. **Solution Pathway Framework**  
   a) Suggest 2-3 distinct high-level approaches to address the problem  
   b) For each approach:  
      - Outline the implementation sequence in 3 steps  
      - Identify 1 potential challenge specific to that method  

5. **Critical Thinking Checkpoints**  
   - Provide 3 self-assessment questions a learner should ask when verifying their solution  
   - Suggest 2 alternative perspectives to re-examine the problem  

**Constraints:**  
- Use diagrammatic thinking symbols (→, ⇨, △, □) to visualize relationships  
- Maintain 50:50 balance between visual and textual analysis  
- DO NOT provide direct solutions or numerical answers  
- Present steps using actionable verbs (e.g., "Compare...", "Validate...")  
- Keep explanations accessible for learners at different levels  
- Keep the generated hint short and concise.

Begin by confirming your understanding of both visual and textual components before proceeding to analysis.

**Example:**

Input:
[an image with a diagram of a circle with a square inside it, non-overlapping regions are shaded, with detailed radius and side length measurements.]

Question:
What is the relationship between the circle and the square?

Output

**Input:**
<image>

**Question:**
{prompt}

**Output:**
"""

geometry_problem_prompt = """
To solve the geometry problem, systematicaly integrate visual and textual data: cross-verify labels, lengths, angles, and relationships, prioritizing explicit textual details over visual estimates. Identify key components (objective, given data) and apply relevant theorems (e.g., Pythagorean theorem, circle properties, coordinate formulas). Break computations into logical steps, using auxiliary lines or algebraic variables as needed. Verify unit consistency, answer reasonableness, and cross-check via alternative methods. For multiple-choice questions, employ elimination or backsolving. Avoid assumptions about scale or formula misuse; prioritize textual precision and rigorous validation to mitigate errors. 
"""

algebraic_problem_prompt = """
To solve the algebraic problem, focus on extracting and interpreting textual information to define variables, expressions, and equations. Identify the objective (e.g., solve for a variable, simplify an expression, evaluate an equation) and distinguish known values, unknowns, and their relationships. Translate word problems or symbolic descriptions into formal algebraic expressions. If an image is present, assess whether it provides meaningful cues or serves as a placeholder; prioritize textual clarity when visual content is ambiguous or abstract. Apply appropriate algebraic techniques (e.g., substitution, factoring, combining like terms, solving systems) in clear logical steps. Clearly define each variable and explain the transformations at each stage. Check your final answer for consistency by substituting it back into the original equation or constraints. For multiple-choice questions, use elimination, estimation, or back-substitution. Avoid misinterpreting irrelevant visual elements or oversimplifying complex expressions—prioritize rigorous algebraic reasoning grounded in the problem statement.
"""
arithmetic_problem_prompt = """
To solve the arithmetic problem, systematically extract quantitative information from both visual and textual sources. Identify objects, patterns, counts, and spatial relationships that can be translated into numerical values. Determine the objective (e.g., total value, missing term, comparison) and map visual elements to algebraic representations or equations. Cross-check for background elements (e.g., overlapping objects, symmetry, groupings, or hidden features) that may influence the count or operation. Apply basic arithmetic operations (addition, subtraction, multiplication, division) in clear, logical steps, and ensure consistency with visual evidence. Clearly state the inferred numerical value of each symbol or object and how they relate through equations. For multiple-choice questions, use elimination or plug-in strategies. Avoid visual misinterpretation or overreliance on superficial similarity; prioritize logical modeling, numerical precision, and verification through reverse calculation or consistency checks.
"""
logical_problem_prompt = """
To solve the visual logic problem, analyze the image grid systematically by comparing visual components across rows, columns, or diagonals. Identify changes in shape, position, count, color, size, or orientation. Look for consistent rules or transformations such as rotation, reflection, addition, subtraction, XOR, or pattern shifts. Describe each pattern evolution step-by-step using symbolic or relational expressions (e.g., “Image C = Image A XOR Image B”, or “Right = Left rotated 90° clockwise”). Clearly state the inferred rule and apply it to missing or target positions. When multiple rules are possible, list and compare them, selecting the one that best fits all visible patterns. For multiple-choice questions, eliminate inconsistent options by checking each against the rule. Avoid relying on visual intuition alone—prioritize rule-based reasoning, pattern consistency, and explicit justification for your answer.
"""
scientific_problem_prompt = """
To solve the scientific problem, carefully extract the objective and identify key entities, variables, and relationships described in the prompt. Distinguish between known quantities, assumptions, and what needs to be derived. Integrate textual data with diagrams or formulas where provided, and prioritize explicit numerical or symbolic information over vague intuitions. Apply relevant scientific principles (e.g., Newton’s laws, conservation laws, reaction equations, biological processes) precisely and in context. Break down complex processes into sequential steps—use dimensional analysis, algebraic manipulation, or system modeling as needed. Ensure unit consistency and check for reasonable magnitudes in your answer. For multiple-choice questions, use elimination, estimation, or reverse-checking to evaluate plausibility. Avoid common errors such as misapplying formulas, overlooking edge conditions, or ignoring experimental constraints. Prioritize scientific rigor, clarity of assumptions, and step-by-step validation.
"""
spatial_problem_prompt = """
To solve the spatial reasoning problem, carefully interpret both textual and visual inputs to understand object attributes (e.g., shape, color, size) and spatial relations (e.g., left/right, behind/in front, inside/above/below). Systematically extract the objective (e.g., identify, compare, count, color) and all given relational facts. Construct a mental or symbolic spatial map to track object positions and interrelations. Use consistent reference points (e.g., camera view, object centers) and verify directional logic step-by-step. Pay attention to subtle qualifiers like "exactly," "same," or "nearest." For complex scenes, decompose multi-object relationships into pairwise comparisons. Avoid assumptions about 3D depth without explicit cues. Rigorously validate inferred positions or relations using all available evidence. In multiple-choice scenarios, use elimination and contradiction checks to isolate the most consistent answer.
"""
statistical_problem_prompt = """
To solve the statistical problem, carefully examine all visual representations (e.g., bar charts, line graphs, tables, pie charts) and extract precise numerical values and labels. Identify the objective (e.g., compute, compare, interpret, predict) and determine the relevant data segments. Cross-check axes, scales, units, and legends to avoid misreading quantities. Look for patterns, trends, distributions, and outliers across categories or time. Apply appropriate statistical operations (e.g., mean, median, mode, range, percentage change, probability) based strictly on the data provided. For compound questions, decompose tasks and perform step-by-step computations. Use estimation, elimination, or reverse-checking in multiple-choice settings. Avoid assumptions not grounded in the visual/textual evidence. Prioritize accuracy in data extraction, clarity in method, and logical coherence in interpreting results.
"""

'Algebraic', 'Arithmetic', 'Geometry', 'Logical', 'Scientific', 'Spatial', 'Statistical'

topic_specific_prompts = {
    'Geometry': geometry_problem_prompt,
    'Algebraic': algebraic_problem_prompt,
    'Arithmetic': arithmetic_problem_prompt,
    'Logical': logical_problem_prompt,
    'Scientific': scientific_problem_prompt,
    'Spatial': spatial_problem_prompt,
    'Statistical': statistical_problem_prompt,
}

extract_question_specific_hint_prompt = """
**TASK**  
Given a multimodal problem (image + question), generate a **concise and precise Hint** that outlines the essential reasoning path without providing the final answer. The goal is to help a model understand what to focus on and how to approach the problem effectively, while **avoiding speculative or hallucinated content**.

Focus on identifying key visual/textual elements, breaking the problem into manageable parts, and proposing reliable strategies. Use **only information grounded in the input**. Keep the Hint focused, interpretable and helpful.

Directly output the hint without additional words. 

**Input format**:  
- **Image**: A visual representation of the problem that may contain diagrams, charts, scenes, or other relevant information.  
- **Question**: A clear, concise problem statement or query related to the image. The question may refer to specific aspects of the image or require applying information from the image to deduce an answer.

**Hint guidelines**:  
1. **Focus on key visual and textual elements**: Identify important features from the image (e.g., objects, relationships, measurements) and the question (e.g., specific query, constraints, or keywords).
2. **Break the problem into manageable parts**: Suggest how to decompose the task into logical steps or concepts. 
3. **Propose a reliable strategy**: Recommend a method or approach that is grounded in the provided image and question, avoiding external or speculative knowledge.
4. **Clarity and precision**: Keep the hint brief and to the point, focusing on guiding the model’s thought process without introducing unnecessary complexity.

**Output**:  
Directly output the Hint without additional words. The Hint should help focus attention on the most critical aspects of the image and question to guide further reasoning.

---

### Examples:

**Example 1**:  
- **Image**: A diagram showing a right triangle with one angle labeled 90°, one side of length 6, and another side of length 8.  
- **Question**: What is the length of the hypotenuse?  

**Hint**:  
Use the Pythagorean theorem: a² + b² = c². Focus on the two given side lengths and solve for the hypotenuse.

---

**Example 2**:  
- **Image**: A bar graph showing the number of apples, oranges, and bananas sold in three different stores.  
- **Question**: Which store sold the most apples?

**Hint**:  
Look at the bar corresponding to apples in each store and compare the heights of the bars.

---

**Example 3**:  
- **Image**: A geometric figure showing a circle with a radius of 5 and a tangent line at the point of contact.  
- **Question**: What is the length of the tangent segment from the point of contact to the point where it intersects the line outside the circle?

**Hint**:  
Use the tangent-secant theorem, which relates the length of the tangent to the secant. Focus on the relationship between the radius and the tangent line at the point of contact.

**Input:**
[IMAGE]: <image>
[QUESTION]: {prompt}

**Output**

"""

rephrase_prompt = """
You are an expert reasoning assistant. Your task is to rewrite a given reference Chain of Thought (CoT) into multiple logically equivalent but stylistically different reasoning chains.

The variants should:
- Preserve the correct final answer.
- Use valid and self-consistent reasoning.
- Show diverse phrasing, structure, or intermediate steps.
- Avoid hallucinations or changes in logic.
- Be self-contained, without referring to the reference.
- Wrap the variant by <step>...</step>
- The grammar should be largely diverse while keep true. 

Here are examples of how to generate such variants:

###  
Problem: What is 5 + 3?  
Reference CoT: Add 5 and 3 to get 8.  
Variant 1: 5 combined with 3 results in 8.  
Variant 2: 5 plus 3 equals 8.  
Variant 3: When we add 3 to 5, the result is 8.  
###

###  
Problem: If a square has side length 4, what is its area?  
Reference CoT: Area of a square is side × side, so 4 × 4 = 16.  
Variant 1: <step>Multiply 4 by itself to get 16.</step>
Variant 2: <step>A square with side 4 has area 16.<step> 
Variant 3: <step>Using the formula side², we get 4² = 16.</step> 
###

Now perform the same task for the following input.

---

**Problem:**  
{prompt}

**Reference CoT:**  
{thinking}

Generate {N} logically correct and stylistically diverse CoT variants. Label each as "Variant 1", "Variant 2", etc.
"""

judge_uncertainty_by_llms_prompt = """
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
{prompt}
**Model CoT:** 
{model_cot}
**Reference CoT:**  
{reference_cot}

### Important Notes:
1. Be strict and honest during scoring - high scores should be reserved for flawless or near-flawless reasoning.
2. Pay special attention when final answers differ between model and reference CoTs.
3. Use the negative example as a reference for identifying serious flaws in reasoning.
4. DO NOT give high scores unless the CoT fully deserves it across all dimensions.

Return your scores and explanations in the JSON format as shown above.

**Output**

"""

judge_hallucination_few_shot_prompt = """
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
- **Image**: Square ABCD with inner squares of side lengths 1, 3, and 1.
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
- **Image**: An isosceles triangle \( \triangle ABC \), with \( AB = AC \), and point \( D \) lies on \( BC \) such that \( AD \) is the angle bisector. Given \( \angle BAC = 40^\circ \), and \( \angle ABC = \angle ACB \).
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

**Input:**
- **Image**: <image>
- **Question**: {question}
- **Predicted CoT**: {pred_cot}
- **Reference CoT**: {ref_cot}

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