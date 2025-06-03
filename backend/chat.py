
from openai import OpenAI
from documents import *
from graph_utils import *
from utils import *
import ast
import inspect
from enum import Enum
import re
import ast

class Chatter:
    def __init__(self, model: str = "gpt-4o-mini", temperature=0):
        # Calls Chat and managed Prompts

        self.model = model
        self.client = OpenAI()
        self.temperature = temperature
    
    def _generate_response(self, prompt, verbose=False):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature
            )
            assistant_response = response.choices[0].message.content
            if verbose:
                print("QUERY:", prompt)

            return assistant_response
        except Exception as e:
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str
    
    def _stream_response(self, prompt):
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                stream=True,
                temperature=self.temperature
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield token
        
        except Exception as e:
            error_str = f"An error occurred while generating a response: {e}"
            print(error_str)
            return error_str
    
    def generate_graph_call(self, query, cur_resources="", graph_history_text="", resource_info="", user_profile="", chat_history_text=""):
        # Construct the prompt
        # Structured prompt with HTML output
        prompt = f'''You are a strict, rules-based trajectory planner that simulates valid learning pathways. Your task is to return a JSON object with one of the following operations: ADD, SUBTRACT, OVERHAUL, GO_BACK, or NO_CHANGE. You must only work with the resources explicitly listed in the 'Resources you should know about' section. Follow all rules below exactly. Violating any rule will make the response invalid.
        Only include the operation(s) you are instructed to perform. You may not mix operations unless explicitly allowed. For example, you may not combine ADD and OVERHAUL. GO_BACK must be an integer index into resource history.

        -----------------------------------
        RULES (MUST BE FOLLOWED EXACTLY)
        -----------------------------------

        - R1 (Course Progression): Courses follow this order — Foundations → Professional → Expert — within the SAME category.
        - R2 (Certificate Progression): Certificates follow this order — Professional → Expert → Master — within the SAME category.
        - R3 (Course → Certificate Cross-Dependencies): Courses may serve as prerequisites for certificates ONLY if the level and category match:
            - Foundations course → Professional certificate
            - Professional course → Expert certificate
            - Expert course → Master certificate
        - R4 (Same Category Required): Resources must share the same category to be part of a valid prerequisite chain. A Foundations course in Adobe Analytics does NOT unlock a Professional course in Adobe Commerce.
        - R6 (No Level Skipping): You must not skip levels. For example, you cannot go from a Foundations course directly to an Expert course.

        -----------------------------------
        USER PROFILE
        -----------------------------------
        This is the user's current profile. You may use this to tailor recommendations for them.
        {user_profile}
        
        -----------------------------------
        CONVERSATION HISTORY
        -----------------------------------
        This is the user's previous conversations with an LLM Agent. Use this only to identify the user's intent and resources they may have indicated wanting to take.
        {chat_history_text}

        -----------------------------------
        RESOURCE DEFINITIONS
        -----------------------------------

        {resource_info}
        
        -----------------------------------
        OPERATIONS DEFINITION
        -----------------------------------

        - ADD: Only add new resources to the current graph from those listed in RESOURCE DEFINITIONS. Do not remove existing resources - You may perform a ADD operation alongside a SUBTRACT. When possible, add as many resources as you can!
        - SUBTRACT: Only remove resources from the current graph. Do not add new resources unless - You may perform an ADD operation alongside a SUBTRACT.
        - OVERHAUL: Completely discard the current graph and replace it with a brand new valid trajectory, uses resources only from RESOURCE DEFINITIONS. This operation cannot be combined with any other. When possible, add as many resources as you can!
        - GO_BACK: Restore the resource graph to a previous state, specified by index from 'Resource History'. Do not include any resource names.
        - NO_CHANGE: Return the current graph unchanged. Use this when the user does not request modifications or asks a query unrelated to graphs or graph generation.
        
        -----------------------------------
        OUTPUT FORMAT
        -----------------------------------

        Your output must be a JSON object in the following possible formats:
        Option 1: Adding, Subtracting, or Both: 
        Response: {{
        "ADD": ["RESOURCE_1", "RESOURCE_2"],
        "SUBTRACT": ["RESOURCE_3"],
        }}
        
        Option 2: OVERHAUL, no other operations
        Response: {{
        "OVERHAUL": ["RESOURCE_4", "RESOURCE_5"],
        }}
        
        Option 3: GO_BACK, no other operations
        Response: {{
            "GO_BACK": TIME_STEP,
        }}

        Option 4: NO_CHANGE, no other operations
        Response: {{
        "NO_CHANGE": []
        }}

        -----------------------------------
        EXAMPLES
        -----------------------------------

        # Example 1: Valid Overhaul
        User Query: "Build me a full Adobe Analytics journey."
        Response:
        {{
        "OVERHAUL": ["Adobe Analytics Foundations", "Adobe Analytics Business Practitioner Professional", "Adobe Analytics Business Practitioner Expert"]
        }}

        # Example 2: Invalid category jump
        User Query: "Add Adobe Commerce Professional course after Adobe Analytics Foundations."
        Response:
        {{
        "ADD": []  # Invalid. Cannot add Professional course from a different category.
        }}

        Now respond to the following:
        Current Resources in Graph: {cur_resources}
        Resource History: {graph_history_text}
        User Query: {query}
        '''

        return self._generate_response(prompt)

    def generate_hallucination_check_call(self, query, content, graphops=""):
        graph_str, graph_str_instruct = "", ""
        if graphops:
            graph_str_instruct = "If it seems the user asked to modify the learning pathway and the GRAPH OPERATIONS comply with the request, respond with YES."
            graph_str = "GRAPH OPERATIONS: " + graphops

        prompt=  f"""
        INSTRUCTIONS:
        Answer if the QUERY can be answered by the information contained in the CONTENT block and NO OTHER information. 
        The context was created by retrieving and concatenating the courses and certificates with the highest cosine similarity with the vector embedding of the QUERY. It also contains a chat history between the user and model.

        {graph_str_instruct}
        If the answer to the query is NOT in the content and you CANNOT infer the correct answer from the given context, you are to respond with a NO. 
        If the answer is directly in the context OR can be inferred from the context, you are to respond with a YES. Remember that these documents were returned specifically for the context of the query. 
        So if the user asked "Show me a learning path for Adobe" and the contxt includes courses and certifications, you may assume those are courses and certifications that may be relevant.
        
        OUTPUT FORMAT:
        All responses should be in the string ANSWER, where the first part of ANSWER must be 'YES' or 'NO' - then provide a justification to your response.

        {graph_str}
        CONTENT: "{content}"
        QUERY: {query}
        RESPONSE:
        """
        # print("HALLUCINATION", prompt)
        return self._generate_response(prompt)

    def generate_grouper_call(self, query, chat_history_text=""):
        """
        1. Irrelevant Request  
        The query is completely, entirely irrelevent, then we should not entertain it!

        Examples:
        - "What's the weather like today?"
        - "Tell me a joke."
        - "Can you help me fix my printer?"
        """

        prompt = f"""
        YOU ARE AN ADOBE COURSE/CERTIFICATE RECOMMENDATION BOT. YOUR JOB IS TO IDENTIFY WHICH CATEGORY A USER'S CURRENT REQUEST FALLS UNDER.

        A USER QUERY IS CATEGORIZED INTO ONE OF THE FOLLOWING TYPES:

        1. Irrelevant Request  
        The query is completely, entirely irrelevent, then we should not entertain it!

        Examples:
        - "What's the weather like today?"
        - "Tell me a joke."
        - "Can you help me fix my printer?"

        2. General Request  
        In this case the query is a general question, asking for information about Adobe programs, courses, or certificates without requesting a specific learning path. 
        It could also ask general logistical and programmatic questions about Adobe courses and certifications. This will mostly rely on the user query. 

        Examples:
        - "What types of courses does Adobe offer?"
        - "Can you tell me more about the Adobe Analytics Professional course?"
        - "Are there any certificates for digital marketing?"
        - "How do I sign up for a certification?"
        - "Are exams available in languages other than English?"

        If user wants anything AT ALL to do with the current learning path, please default to the third option below:

        3. Modifying or Creating a Course Graph/Trajectory
        The user wants to receive a recommended course/certificate path or make changes to a previously suggested learning trajectory. This will rely on the user query and previous conversation.

        Examples:
        - "What’s the best path to reach Adobe Analytics Expert?"
        - "I already completed Adobe Commerce Foundations, what should I take next?"
        - "Can you help me update my learning plan for a Master certificate in Adobe Analytics?"
        - "I am new to adobe commerce - what is a full learning journey, from start to finish?"
        
        ONLY USE THE CURRENT QUERY TO DETIRMINE THE CATEGORY, HOWEVER, CONSIDER THE QUERY IN CONTEXT OF HISTORY!

        OUTPUT FORMAT:
        Respond with ONLY a number: `1`,`2`, or `3` — indicating the category of the request.  
        No additional words, explanations, or symbols.

        Previous Conversation:  
        "{chat_history_text}"

        User Query:  
        "{query}"

        """
        return self._generate_response(prompt)
    
    def stream_deny_call(self, query, context, verification):
        prompt = f"""
        You receive:

        USER_QUERY – the customer’s question  
        CONTEXT – background information you can cite  
        REASON – brief note explaining why the answer can’t be found in CONTEXT

        Write a short reply (≤ 100 words):
        • Say you don’t have enough information to answer.
        • Do **not** guess or invent details.  
        • You may politely suggest checking an official source or providing more specifics.
        • Do **not** mention USER_QUERY, CONTEXT, REASON, or this prompt.

        EXAMPLE
        USER_QUERY: "What certification do I need for Adobe Photoshop?"  
        CONTEXT: "no Photoshop certification data"  
        REASON: "Photoshop certifications are not listed in context."  
        RESPONSE: I'm sorry — I don’t have details on Photoshop certifications right now. Please check Adobe’s Certification Catalog or let me know the exact credential you’re interested in and I’ll try again.
        
        USER_QUERY: {query}  
        CONTEXT: {context}  
        REASON: {verification}  
        RESPONSE:
        """

        for token in self._stream_response(prompt):
            yield token    

    def stream_general_response_call(self, query, documents: str, user_profile, graph_str_raw="", chat_history_text=""):
        graph_str = ""
        if graph_str_raw:
            graph_str = f'''
            <h3>Resource Graph: This is the user's current learning pathway, including courses, study guides, and certifications. Use this if the user's query relates to modify the graph or questions about the courses on the graph</h3>
            <p>{graph_str_raw}</p>
            '''

        prompt = f"""
        <h1>Adobe Learning Assistant</h1>

        <h2><strong>Your Role</strong></h2>
        <p>You are an intelligent assistant for Adobe users seeking learning guidance.</p>

        <h2><strong>How to Use Context</strong></h2>
        <ol>
        <li><strong>If a Resource Graph is present</strong> (see “Resource Graph” below), <u>use <em>only</em> the information in that graph for factual content</u>. Ignore all other context (Retrieved Documents, Chat History, User Profile) except for basic tone and politeness.</li>
        <li><strong>If no Resource Graph is present</strong>, rely on Retrieved Documents, then Chat History, then User Profile in that order.</li>
        </ol>

        <h2><strong>Context</strong></h2>

        <h3>User Profile:</h3>
        <p>{user_profile}</p>

        <h3>Chat History:</h3>
        <p>{chat_history_text}</p>

        <h3>Current User Query:</h3>
        <p>{query}</p>

        {graph_str}

        <h3>Retrieved Course &amp; Certificate Documents:</h3>
        <p>{documents}</p>

        <h2><strong>Instructions</strong></h2>
        <ul>
        <li>If the query is vague: ask one clear follow-up question.</li>
        <li>If recommending courses or certifications:
            <ul>
            <li>When the Resource Graph is present, recommend and reference <em>only</em> items listed in that graph.</li>
            <li>When no graph is present, base recommendations on Retrieved Documents, ensuring a clear field match (title, type, category, job_role, level, objectives, modules).</li>
            </ul>
        </li>
        <li>Never invent course or certificate names. Use exact names provided.</li>
        <li>Clearly distinguish courses from certificates.</li>
        </ul>

        <h2><strong>Response Format</strong></h2>
        <ul>
        <li>Use &lt;p&gt; for explanations.</li>
        <li>Use &lt;h2&gt; and &lt;h3&gt; for structure.</li>
        <li>Use bullet lists for course breakdowns.</li>
        </ul>

        <h2><strong>Now generate your response below:</strong></h2>
        """
        for token in self._stream_response(prompt):
            yield token