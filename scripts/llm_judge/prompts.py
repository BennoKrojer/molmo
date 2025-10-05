IMAGE_PROMPT = """You are a visual interpretation expert specializing in connecting textual concepts to specific image regions. Your task is to analyze a list of candidate words and determine how strongly each one relates to the content within a specified bounding box in an image.

### **Inputs**
1.  **Image**: An image containing a red bounding box highlighting the region of interest.
2.  **Candidate Words**: A list of words to evaluate. Here are the candidate words:
    -   {candidate_words}

### **Instructions**
1.  **Analyze the Bounded Region**: First, carefully identify the primary objects, attributes (e.g., color, texture, shape), and any text located **inside** the red box.
2.  **Consider Context**: Use the immediate surroundings outside the box to better understand the context of the highlighted region.
3.  **Evaluate Each Word**: For each word in the candidate list, judge its relationship to the region based on the criteria below.
4.  **Generate JSON**: Compile your findings into a single JSON object.
5.  **Do not overthink or make implausible or vague connections to come up with a relation**.

### **Evaluation Guidelines**

* **Direct & Conceptual Links**:  
   - A word is **directly related** if it names an object, attribute (like color, shape, or texture), relation or text that is clearly visible inside the bounding box.  
   - A word is **indirectly related** if it connects conceptually or by function to an element in the box but is not literally present i.e. plausible indirect connections.  
* **Text-Specific Relations**: For regions with text, the connection can be direct (characters/words shown) or indirect (concepts implied by the text). 

### **Output Format**
Return a single JSON object. This object should contain the following fields:
{{
    "interpretable": true/false (true is one or more words are related to the region of interest, false otherwise),
    "directly_related_words": ["word1", "word2", ...] (list of words that are directly related to the region of interest)
    "indirectly_related_words": ["word1", "word2", ...] (list of words that are indirectly related to the region of interest)
    "reasoning": "reasoning for your answer"
}}
"""

"""
* **Direct & Conceptual Links**: A word can be related if it directly names an object, an attribute (like **color** or **shape**), or a concept related to the object's function (e.g., "traffic" for a stop sign) in the region of interest or any other property of the region of interest. Include plausible indirect connections.
* **Text-Specific Relations**: For regions with text, the connection might be direct or conceptual. The following examples are rough guidelines for the kinds of plausible associations you can make:
    * The word could be part of the text (e.g., "stop" for a "stop sign").
    * It might relate to a character or phrase (e.g., "L" for the word 'letter' in the region of interest).
    * It could be conceptually linked (e.g., "warning" for a sign that says "Caution" in the region of interest).
"""