IMAGE_PROMPT = """You are a visual interpretation expert specializing in connecting textual concepts to specific image regions. Your task is to analyze a list of candidate words and determine how strongly each one relates to the content of the image.

### **Inputs**
1.  **Image**: An image containing a red bounding box highlighting the region of interest.
2.  **Candidate Words**: A list of words to evaluate. Here are the candidate words:
    -   {candidate_words}

### **Evaluation Guidelines**
There are two types of relationships to consider between the candidate words and the region of interest:

* **Direct Links**: A word is **directly related** if it names an object, attribute (like color, shape, or texture), relation or text that is clearly visible **inside the red bounding box.**
* **Indrect or Conceptual Links**: These can come in two forms:
    - A word is indirectly related if it connects conceptually or by function to an element inside the box but is not literally present i.e. plausible indirect connections. Examples include: "technology" for a image showing a chip or a computer, "warning" for a caution sign etc.
    - A word is also be indirectly related if it describes some aspect of the image outside the box. For example, if the box highlights the ground near a tree, "tree" would be an indirectly related word.

**Important Note:** For regions with text, the connection can be direct (characters/words shown) or indirect (concepts implied by the text). The following examples are rough guidelines for the kinds of plausible associations you can make:
    * The word could be part of the text (e.g., "to" for "stop").
    * It might relate to a character or phrase (e.g., "L" for the word 'letter' in the region of interest).
    * It could be conceptually linked (e.g., "warning" for a sign that says "Caution" in the region of interest).

Carefully look at the highlighted region, its surrounding context and the image as a whole to make your determinations.

### **Output Format**
Return a single JSON object. This object should contain the following fields:
{{
    "interpretable": true/false (true if one or more words are related to the region of interest, false otherwise),
    "directly_related_words": ["word1", "word2", ...] (list of words that are directly related to the region of interest, empty list if none)
    "indirectly_related_words": ["word1", "word2", ...] (list of words that are indirectly related to the region of interest or the image as a whole, empty list if none)
    "reasoning": "reasoning for your answer"
}}
"""