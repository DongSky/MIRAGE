judge_claim_f1_score_prompt = """
**Task:**

THINK STEP BY STEP and Analyze three components:  
1. **Question** (provide the basic judge background information)
2. **Model-generated claims** (between `<claims>` tags)  
3. **Ground-truth claims** (between `<claims>` tags)  

**Output the answer, then provide additional analysis to enhance the accuracy**  

**Important Notes:**
1. The number of decision in <pred_match>...</pred_match> should be the exactly same with claims in the prediction.
2. The number of decision in <gt_match>...</gt_match> should be the exactly same with claims in the ground-truth.
3. If the final answers of prediction and ground-truth are not same, do not easily predict REASONABLE for each claim. Be caution and critically think the consistency across claims and steps. 

---

### Process:  
1. **Extract**  
   - All model claims from the first `<claims>` block.  
   - All GT claims from the second `<claims>` block.  
   - All input options from the `<options>` block.  

2. **Compare model claims vs GT/options**  
   For each model claim:  
   - **MATCH**: Semantically aligns with **any GT claim**.  
   - **REASONABLE**: Not in GT but logically valid given the **input options** and existing claims.  
   - **CONFLICT**: Contradicts **any GT claim**.  

3. **Compare GT claims vs model claims**  
   For each GT claim:  
   - **MATCH**: Covered by **any model claim**.  
   - **CONFLICT**: No matching model claim.  

4. **Calculate Metrics**  
   - **Precision**: `(MATCH + REASONABLE) / Total model claims`  
   - **Recall**: `MATCH in GT claims / Total GT claims`  
   - **F1**: `2 * (Precision * Recall) / (Precision + Recall)`  

---

### Notes for Precision (Pred Matches)

# Step 1: Reformatting the Solution
Convert the unstructured solution into distinct reasoning steps while:
- Preserving all original content and order
- Not adding new interpretations
- Not omitting any steps

## Step Types
1. Logical Inference Steps
   - Contains exactly one logical deduction
   - Must produce a new derived conclusion
   - Cannot be just a summary or observation

2. Image Description Steps
   - Pure visual observations
   - Only includes directly visible elements
   - No inferences or assumptions

3. Background Information Steps
   - External knowledge or question context
   - No inference process involved

## Step Requirements
- Each step must be atomic (one conclusion per step)
- No content duplication across steps
- Initial analysis counts as background information
- Final answer determination counts as logical inference

# Step 2: Evaluating Correctness
Evaluate each step against:

## Ground Truth Matching
For image descriptions:
- Key elements must match ground truth descriptions

For logical inferences:
- Conclusion must EXACTLY match or be DIRECTLY entailed by ground truth

## Reasonableness Check (if no direct match)
Step must:
- Premises must not contradict any ground truth or correct answer
- Logic is valid
- Conclusion must not contradict any ground truth 
- Conclusion must support or be neutral to correct answer

### Notes for Recalls (GT Matches)

You need to match each ground truth middle step with the solution:

Match Criteria:
- The middle step should exactly match in the content or is directly entailed by a certain content in the solution
- All the details must be matched, including the specific value and content
- You should judge all the middle steps for whethere there is a match in the solution

---

### Output Format:  
```xml  
<pred_match>MATCH,REASONABLE,CONFLICT,...</pred_match>  
<gt_match>MATCH,CONFLICT,...</gt_match>  
<precision>0.XX</precision>  
<recall>0.XX</recall>  
<f1>0.XX</f1>  
```  

---

### Input Example:  
Question:
Water boiling temperature?

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

---

### Output Example:  
<pred_match>MATCH,REASONABLE</pred_match>  
<gt_match>MATCH,CONFLICT</gt_match>  
<precision>1.00</precision>  
<recall>0.50</recall>  
<f1>0.67</f1>  

**Explanation:**  
- Model claim "Water boils at 90°C" is **REASONABLE** (aligned with input options but not explicitly in GT).  
- Precision = (1 MATCH + 1 REASONABLE) / 2 claims = 1.00  
- Recall = 1 GT MATCH / 2 GT claims = 0.50  

---

### Input:  
Question:
{question}

Prediction from Model:  
<claims>  
{pred_claims}  
</claims>  

Ground-Truth:  
<claims>  
{gt_claims}  
</claims>  


**Output:**

"""


judge_step_f1_score_prompt = """
**Task:**

THINK STEP BY STEP and Analyze three components:  
1. **Question** (provide the basic judge background information)
2. **Model-generated steps** (between `<steps>` tags)  
3. **Ground-truth steps** (between `<steps>` tags)  

**Output the answer, then provide additional analysis to enhance the accuracy**  

**Important Notes:**
1. The number of decision in <pred_match>...</pred_match> should be the exactly same with steps in the prediction.
2. The number of decision in <gt_match>...</gt_match> should be the exactly same with steps in the ground-truth.


---

### Process:  
1. **Extract**  
   - All model steps from the first `<steps>` block.  
   - All GT steps from the second `<steps>` block.  
   - All input options from the `<options>` block.  

2. **Compare model steps vs GT/options**  
   For each model step:  
   - **MATCH**: Semantically aligns with **any GT step**.  
   - **REASONABLE**: Not in GT but logically valid given the **input options** and existing steps.  
   - **CONFLICT**: Contradicts **any GT step**.  

3. **Compare GT steps vs model steps**  
   For each GT step:  
   - **MATCH**: Covered by **any model step**.  
   - **CONFLICT**: No matching model step.  

4. **Calculate Metrics**  
   - **Precision**: `(MATCH + REASONABLE) / Total model steps`  
   - **Recall**: `MATCH in GT steps / Total GT steps`  
   - **F1**: `2 * (Precision * Recall) / (Precision + Recall)`  

---

### Notes for Precision (Pred Matches)

# Step 1: Reformatting the Solution
Convert the unstructured solution into distinct reasoning steps while:
- Preserving all original content and order
- Not adding new interpretations
- Not omitting any steps

## Step Types
1. Logical Inference Steps
   - Contains exactly one logical deduction
   - Must produce a new derived conclusion
   - Cannot be just a summary or observation

2. Image Description Steps
   - Pure visual observations
   - Only includes directly visible elements
   - No inferences or assumptions

3. Background Information Steps
   - External knowledge or question context
   - No inference process involved

## Step Requirements
- Each step must be atomic (one conclusion per step)
- No content duplication across steps
- Initial analysis counts as background information
- Final answer determination counts as logical inference

# Step 2: Evaluating Correctness
Evaluate each step against:

## Ground Truth Matching
For image descriptions:
- Key elements must match ground truth descriptions

For logical inferences:
- Conclusion must EXACTLY match or be DIRECTLY entailed by ground truth

## Reasonableness Check (if no direct match)
Step must:
- Premises must not contradict any ground truth or correct answer
- Logic is valid
- Conclusion must not contradict any ground truth 
- Conclusion must support or be neutral to correct answer

### Notes for Recalls (GT Matches)

You need to match each ground truth middle step with the solution:

Match Criteria:
- The middle step should exactly match in the content or is directly entailed by a certain content in the solution
- All the details must be matched, including the specific value and content
- You should judge all the middle steps for whethere there is a match in the solution

---

### Output Format:  
```xml  
<pred_match>MATCH,REASONABLE,CONFLICT,...</pred_match>  
<gt_match>MATCH,CONFLICT,...</gt_match>  
<precision>0.XX</precision>  
<recall>0.XX</recall>  
<f1>0.XX</f1>  
```  

---

### Input Example:  
Question:
Water boiling temperature?

Prediction from Model:  
<steps>  
<step>Earth orbits the Sun.</step>  
<step>Water boils at 90°C.</step>  
</steps>  

Ground-Truth:  
<steps>  
<step>The Earth revolves around the Sun.</step>  
<step>Water boils at 100°C at sea level.</step>  
</steps>  

---

### Output Example:  
<pred_match>MATCH,REASONABLE</pred_match>  
<gt_match>MATCH,CONFLICT</gt_match>  
<precision>1.00</precision>  
<recall>0.50</recall>  
<f1>0.67</f1>  

**Explanation:**  
- Model step "Water boils at 90°C" is **REASONABLE** (aligned with input options but not explicitly in GT).  
- Precision = (1 MATCH + 1 REASONABLE) / 2 steps = 1.00  
- Recall = 1 GT MATCH / 2 GT steps = 0.50  

---

### Input:  
Question:
{question}

Prediction from Model:  
<steps>  
{pred_steps}  
</steps>  

Ground-Truth:  
<steps>  
{gt_steps}  
</steps>  


**Output:**

"""

judge_hallucination_prompt = """
**Task Requirement:**
You are provided with a multi-modal reasoning model’s chain-of-thought (CoT). Your task is to analyze the chain-of-thought to detect any signs of hallucination across several categories. Please examine the chain-of-thought for the following hallucination types:

1. **Spatial Hallucination:** Check for errors in shape, position, and complex visual operations that do not match the input data.
2. **Logical Hallucination:** Look for inconsistencies or flaws in mathematical reasoning and logical problem solving.
3. **Factuality Hallucination:** Identify any scientific or statistical claims that contradict known facts or the given dataset.
4. **Context Hallucination:** Determine if there are inconsistencies or contradictions across different parts of the chain-of-thought or with the predicted answers.
5. **Fabrication Hallucination:** Detect if any parts of the mathematical or statistical reasoning appear to be fabricated or not supported by explicit intermediate steps.

For the overall analysis and for each hallucination type, provide:
- A **"result"** field that is either **"CONFIRM"** (if no hallucination is detected) or **"HALLUCINATION"** (if hallucination is present).
- A **"reason"** field that explains the detailed rationale behind your judgment.

Output your analysis in a JSON format wrapped inside `<result></result>` tags. The final JSON should have the following structure:

**Output format:**
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
**Note:**
Analyze the provided chain-of-thought carefully and provide your final evaluation according to the schema above. Ensure that your output is strictly in the required JSON format, enclosed within `<result>` and `</result>` tags, and that it accurately reflects your analysis of each hallucination type as well as the overall analysis.

**Input:**
{pred_cot}

**Output:**

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