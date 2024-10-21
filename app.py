from flask import Flask, render_template, jsonify, request, send_from_directory,session,render_template_string
import random
import os
import re
import pickle
from pydantic import BaseModel,Field
import uuid
from langchain.chains import LLMChain
import os
from langchain_chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.prompts import PromptTemplate
from google.cloud.sql.connector import Connector, IPTypes
import sqlalchemy
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_community.llms import OpenAI
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableConfig, chain
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.utilities import PythonREPL
import threading
import time
import warnings
warnings.filterwarnings("ignore")


class Element(BaseModel):
    type: str
    text: str
    source_pdf: str  # Keep track of the source PDF file
    policy_name: str  # Add policy name to track policy

with open("pdf_elements.pkl", "rb") as f:
    data = pickle.load(f)

with open("pdf_elements_summarization.pkl", "rb") as f:
    data_summarization = pickle.load(f)

table_elements = data["table_elements"]
text_elements = data["text_elements"]
table_summaries =data_summarization['table_summaries']
text_summaries = data_summarization['text_summaries']

openai_embeddings = OpenAIEmbeddings()

# Initialize Chroma with persistence
vectorstore = Chroma(
    collection_name="LIC_summary",
    embedding_function=openai_embeddings
)

# Initialize in-memory docstore and retriever
store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key
)

summaries_with_source = [
    {"summary": summary, "source_pdf": elem.source_pdf,"policy_name":elem.policy_name}
    for summary, elem in zip(text_summaries, text_elements)
]

# Generating unique doc_ids
doc_ids = [str(uuid.uuid4()) for _ in summaries_with_source]

# Create Document objects with both summary and source_pdf in metadata
summary_texts = [
    Document(
        page_content=summary["summary"],
        metadata={id_key: doc_ids[i], "source_pdf": summary["source_pdf"],"policy_name":summary["policy_name"]}
    )
    for i, summary in enumerate(summaries_with_source)
]

# Add documents to the vector store
retriever.vectorstore.add_documents(summary_texts)

# Store the summaries and doc_ids with source_pdf in the docstore
retriever.docstore.mset(list(zip(doc_ids, [summary["summary"] for summary in summaries_with_source])))

# After getting the table_summaries, pair them with their source PDFs
table_summaries_with_source = [
    {"summary": summary, "source_pdf": elem.source_pdf,"policy_name":elem.policy_name}
    for summary, elem in zip(table_summaries, table_elements)
]

# Generating unique ids for each table summary
table_ids = [str(uuid.uuid4()) for _ in table_summaries_with_source]

# Create Document objects with both summary and source_pdf in metadata
summary_tables = [
    Document(
        page_content=table_summary["summary"],
        metadata={id_key: table_ids[i], "source_pdf": table_summary["source_pdf"],"policy_name":table_summary["policy_name"]}
    )
    for i, table_summary in enumerate(table_summaries_with_source)
]

# Add documents to the vector store
retriever.vectorstore.add_documents(summary_tables)

# Store the summaries and table_ids with source_pdf in the docstore
retriever.docstore.mset(list(zip(table_ids, [table_summary["summary"] for table_summary in table_summaries_with_source])))


### NLU based query execution


# Setup environment
service_account_path = 'tidal-mason-432703-m8-d2b1bea89627.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

INSTANCE_CONNECTION_NAME = 'tidal-mason-432703-m8:us-central1:mysqlserver0908'
DB_USER = 'admin'
DB_PASS = 'Asdo@123'
DB_NAME = 'recommend'

# Initialize the connector
connector = Connector()

def getconn():
    """Establish a connection to the Cloud SQL database."""
    conn = connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pymysql",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
        ip_type=IPTypes.PUBLIC
    )
    return conn

# Create SQLAlchemy engine with the Cloud SQL connector
pool = sqlalchemy.create_engine(
    "mysql+pymysql://",
    creator=getconn,
)

# Define the function
def recommend_policy(user_query):
    """
    Recommends a policy based on user input by querying the database
    and returning policy name and UIN based on conditions.

    Args:
        user_query (str): The query with user details and requirements.

    Returns:
        str: The recommended policies.
    """

    # Define LangChain's SQLDatabase using the existing pool
    db = SQLDatabase(engine=pool)

    # Setup the LLM model
    llm = ChatOpenAI(model_name="gpt-4")

    # Create toolkit for SQL agent
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Create the agent executor
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    # Define the prompt template with column mappings and user instructions
    policy_recommendation_template = """
    You are an expert in policy recommendations. Based on the user’s query, extract relevant details and map them to the database fields to recommend suitable policies by their name and UIN.

    ### Column Mappings:
    - `policy_minimum_age_entry`: Minimum age required for policy entry.
    - `policy_maximum_age_entry`: Maximum age allowed for policy entry.
    - `type_of_policy`: Type of insurance (e.g., Life, Savings, Pension, Retirement).
    - `minimum_sum_assured`: Minimum sum assured by the policy.
    - `maximum_sum_assured`: Maximum sum assured by the policy.
    - `payout`: Payout type (e.g., Lump sum, Installment).

    ### User Query:
    {query}

    ### Conditions:
    1. The user's age must be below the `policy_minimum_age_entry` or `policy_maximum_age_entry`.
    2. At least one of the following conditions must be satisfied:
       - The `minimum_sum_assured` is less than or equal to the user-specified sum assured.
       - The `maximum_sum_assured` is greater than or equal to the user-specified sum assured.
    3. Always apply distinct to the output
    4. Explain why you recommend  these polices with relevant points  like an insurance agent.

    ### Task:
    1. Parse the user query to extract values for each relevant field.
    2. Use these values to find suitable policies, returning only the `policy_name` and `policy_UIN_no` of those that match the criteria.
    3. Most provide 2 to 3 follow up questions from below to the users relevant to recommended policy like that.
        **follow-up questions** : 1. What is LIC jeevan umang?
                            2. what is LIC Jeevan Azad?
                            3. What is LIC new pension plus?
                            4. what is LIC jeevan Ustav?
                            5. What are benefits of LIC bhima Shree?
                            6. What are benefits of LIC jeevan Ustav?
                            7. what are the benefits of LIC new pension plus?
                            8. is there any Rebate for LIC bhima shree?
    Returns: response
             follow-up questions
    """

    # Create a PromptTemplate instance
    template = PromptTemplate(
        input_variables=["query"],
        template=policy_recommendation_template
    )

    # Format the template with the user query
    mapped_query = template.format(query=user_query)

    # Run the agent executor to get the response
    response = agent_executor.run(mapped_query)

    return response

### calculation_agent

python_repl = PythonREPL()

def insurance_calculator(query, llm=None, prompt_template=None):
    """
    Takes an insurance-related query, generates Python code to calculate the answer,
    and returns the result as an insurance agent with currency formatting.
    
    Parameters:
        query (str): A natural language query that describes an insurance calculation.
        llm: Optional. A ChatOpenAI LLM instance for generating Python code. If not provided, it will use a default instance.
        prompt_template: Optional. A PromptTemplate instance defining the code generation prompt. If not provided, a default prompt template will be used.
    
    Returns : str : An insurance agent kind of explanation.
    
    """
    
    # Define default LLM configuration if not provided
    if llm is None:
        llm = ChatOpenAI(temperature=0.4)
    
    # Define default prompt template if not provided
    if prompt_template is None:
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""You are an assistant that creates Python code to perform calculations for insurance-related queries.
            - Formulate a Python code snippet to answer the query.
            - If any information needed for the formula is missing, assume example values and document them.
            - Store the final calculation in a variable named `result`,
            - Only provide Python code for the calculation  you should print it out with `print(...)`."
            - make `result` in a indian currency format
            - An insurance agent 
            - most include 2 to 3 follow up questions with response. you can refer below follow up questions for that
                        **follow up questions** :   1. Calculate the monthly premium for a 200,000 term life insurance over 20 years with a 4.5% interest rate?
                                                2. What is the annual premium for a 50,000 whole life insurance policy with a 3% interest rate and a 30-year term?
                                                3. Find the monthly premium for a 500,000 disability insurance policy over a 15-year term, assuming an interest rate of 6%?
                                                4. Calculate the premium for a 250,000 critical illness insurance policy for 25 years, assuming a 5.5% interest rate ?
                                                5. What would be the annual premium for a 150,000 health insurance policy over 12 years with an interest rate of 4.2% ?
                                                6. Determine the monthly premium for a 300,000 mortgage insurance policy over 10 years with a 4% interest rate ?
                                                7. How much would the premium be for a 400,000 annuity over 20 years with a 3.8% interest rate ?
                                                8. Calculate the monthly premium for a 100,000 life insurance policy with a 5% interest rate and a 5-year term ?
                                                9. Find the annual premium for a 600,000 long-term care insurance policy over 30 years, assuming a 6.2% interest rate ?
                                                10. What would be the monthly premium for a 350,000 accidental death insurance policy

            Query: {query}
            
            Python Code:

            Return : result
                     follow-up questions
            """
        )
    
    # Set up the LangChain pipeline with the given LLM and prompt
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Run the chain and pass the query as input
    code_response = chain.run({"query": query})
    
    code = str(code_response)
    print(code)
    # Execute the generated Python code using the PythonREPL
    result = python_repl.run(code)
    print(result)
    return result


class recommend_requirment_checker(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="question are relevant to the insurance advice, 'yes' or 'no'"
    )


# LLM with function call

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(recommend_requirment_checker)

# Prompt
system = """
Policy Recommendation Checker

You are tasked with evaluating whether a user's question aligns with specific insurance policy types. Your goal is to determine if the question relates to life insurance, retirement plans, or other financial goals and considerations like age and sum assured.

1. **Criteria Matching**: The user's question must address each of the following aspects to be marked as relevant:
   - **Age**: Does the question mention or imply the user’s age, which may influence the type of policy recommended?
   - **Sum Assured**: Is the user's desired sum assured or coverage amount mentioned?
   - **Financial Goal**: Does the question clarify a financial objective, such as life insurance for protection, or a retirement plan for long-term savings?

2. **Relaxed Relevance**: The question does not need to perfectly specify the policy type. As long as the user's query hints at a financial goal (e.g., financial security, retirement, family protection, life insurance), mark it as relevant.

3. **Binary Grading**: Provide a binary score ('yes' or 'no') to indicate whether the user's question is relevant to determining a policy recommendation. To be graded as 'yes,' the question must address **age**, **sum assured**, and **financial goal** in a meaningful way.
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),

    ]
)

recom_checker = grade_prompt | structured_llm_grader


class missing_information(BaseModel):


    details: str = Field(
        description="Provide detailed information regarding any missing elements related to the question."
    )


# LLM with function call

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
structured_llm_grader = llm.with_structured_output(missing_information)

# Prompt
system = """
policy Recommendation Missing Information Checker

You are tasked with evaluating whether a user's question includes all the necessary information to align with specific insurance policy types. Your goal is to determine if the question relates to life insurance, retirement plans, or other financial goals, and whether critical information like age and sum assured is missing.


1. **Required Information**: The user's question must address each of the following aspects. If any of these are missing, indicate the missing details to the user:
   - **Age**: Does the question mention or imply the user’s age, which helps in recommending a policy suited to their stage in life?
   - **Sum Assured**: Has the user specified a desired sum assured or coverage amount that reflects the level of financial protection or savings they need?
   - **Financial Goal**: Does the question clarify the user's financial objective, such as life insurance for family protection, or a retirement plan for long-term savings?

2. **Identifying Missing Information**: If any of the three key elements (age, sum assured, financial goal) are missing or unclear, notify the user and provide guidance on what to include.

3. **Feedback**: Clearly indicate which details are missing (age, sum assured, financial goal) give 2 to 3 follow up questions from below.

                             **follow-up questions**  : 1. I am 42 years old and looking for a retirement policy with a sum assured of 150,000 to support my financial security in later years?

                                        2. I am 29 years old and interested in a life insurance policy with a sum assured of 75,000 to provide a safety net for my family?

                                        3. I am 50 years old and searching for a retirement plan with a sum assured of 200,000 to ensure financial stability in my retirement ?

                                         4.  I am 60 years old and would like a retirement policy with a sum assured of 120,000 to maintain my lifestyle post-retirement?

                                          5.  I am 25 years old and looking for a life insurance policy with a sum assured of 50,000 to cover future family expenses?

                                           6.  I am 38 years old and want a life insurance policy with a sum assured of 250,000 to protect my family from unexpected financial burdens?

                                            7.   I am 45 years old and seeking a retirement insurance policy with a sum assured of 180,000 to safeguard my post-retirement life?

                                            8. I am 33 years old and need a life insurance policy with a sum assured of 90,000 for long-term family security?

                                            9. I am 55 years old and looking into a retirement insurance policy with a sum assured of 300,000 for a comfortable retirement phase?

                                            10.I am 40 years old and would like a life insurance policy with a sum assured of 200,000 to ensure my family’s financial well-being?

4. Make response like a insurance agent way with polite manner with simple point with of Required information.
Returns: response
         follow-up question  
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),

    ]
)

missing_info = grade_prompt | structured_llm_grader

### Router


# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search","recommender_question","calculator"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to insurance brouchers of LIC.
Use the vectorstore for questions on these topics. Otherwise, use web-search related to insurance only.
use recommender_question for recommend or suggest a policy
use calculator to do the insurance calculations"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

### Retrieval Grader

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call

llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing the relevance of a retrieved document to a user's question. Your goal is to determine whether the document meaningfully addresses the user's query, even if the exact keywords are not present.

1. **Semantic Relevance**: If the document contains information that answers or relates to the user's question, either through keywords, synonyms, or overall meaning, consider it relevant. Focus on the intent behind the query and whether the document provides useful or contextually related information.

2. **Relaxed Match**: The test does not need to be overly strict. Even if the document does not contain exact terms from the question, as long as the meaning aligns and could be helpful or related, mark it as relevant.

3. **Binary Grading**: Provide a binary score ('yes' or 'no') to indicate whether the document is relevant to the question. If the document addresses the user's intent in any meaningful way, grade it as 'yes'."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),

    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
question = "what is LIC"
docs = retriever.invoke(question)
doc_txt = docs






# Pull the prompt from hub
base_prompt = hub.pull("rlm/rag-prompt")

# Define a prompt template (example) to further structure the pulled prompt
prompt_template = """
Using the following base prompt:
{base_prompt}

Context: {context}
Question: {question}

Based on the given context, answer the question and provide two or three relevant follow-up questions from below based on user {question} like that. Make response like a insurance agent way in a polite manner.

follow-up questions : 1. what are the GSV of LIC bhima shree 2. Where the GSV percentage of LIC bhima shree 3. What are the benefits of LIC bhima shree 4. what are the death benfits of LIC umang
Return : response
         follow-up questions
"""


# Use PromptTemplate to integrate the pulled prompt into the structure
template = PromptTemplate(
    input_variables=["base_prompt", "context", "question"], 
    template=prompt_template
)

# LLM
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.4)

# Chain with LLM and prompt template
rag_chain = LLMChain(llm=llm, prompt=template, output_parser=StrOutputParser())

# Run with your inputs, passing the base prompt as well
generation = rag_chain.invoke({"base_prompt": base_prompt, "context": docs, "question": question})
print(generation['text'])

### Hallucination Grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke({"documents": docs, "generation": generation})


### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader


### Question Re-writer for reteriver

# LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt
system = """You are a question re-writer that converts an input question into a more optimized version for vector store retrieval. Analyze the input question carefully, focusing on the underlying semantic intent and meaning related to insurance, specifically LIC (Life Insurance Corporation) policies.

When rewriting, ensure the question is clear, concise, and relevant to various aspects of LIC policies, such as coverage, premiums, benefits, claims, and types of insurance offered. Always rephrase the question to explicitly mention LIC policies where applicable, enhancing its specificity and relevance for better retrieval in a vector store."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()



# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Initialize the search tool
tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True
)

# Create a prompt template
prompt = ChatPromptTemplate(
    [
        ("system", """you are a LIC insurance web search to web search . give response in 3 points in a short form if need links provide that also. provide 2 to 3 relevant ***follow-up questions from below*** like that 
         
         follow-up question : 1. What is insurance 2. what is GSV in insurance 3. What is SSV in insurance 4. What are the available life insurance polices in LIC 5. What is maturity benefits?

         Make response like a human insurance agent way with polite manner.

         Return : response
                  follow-up questions
                
         
         
         """),
        ("human", "{question}")
    ]
)

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools([tool])
llm_chain = prompt | llm_with_tools

# Function to check if the query is insurance-related
def is_insurance_related(query: str) -> bool:
    insurance_keywords = ["insurance", "LIC", "policy", "premium", "coverage", "claim", "benefit"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in insurance_keywords)

@chain
def tool_web_search_chain(user_input: str):
    # Check if the user input is insurance-related
    if not is_insurance_related(user_input):
        return {"generation":"The query is not relevant to insurance topics or LIC. Please ask a question relevant to LIC."}

    input_ = {"question": user_input}
    ai_msg = llm_chain.invoke(input_)
    tool_msgs = tool.batch(ai_msg.tool_calls)
    return llm_chain.invoke({**input_, "messages": ai_msg.content})





class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

#Generate

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    base_prompt = hub.pull("rlm/rag-prompt")

    # RAG generation
    generation = rag_chain.invoke({"base_prompt": base_prompt, "context": docs, "question": question})
    return {"documents": documents, "question": question, "generation": generation['text']}

## Grade documents

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

## Transform

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def missing_information(state):
  print("--Missing information to recommend")
  question = state['question']
  result = missing_info.invoke({"question": question})
  output = ""
  for i in result:
    output += str(i[1])
  print(output)
  return {'generation':output}


def recommender_question(state):
  question=state['question']
  return {'question':question}

def policy_recommend_checker(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK question meets citeria or not---")
    question = state["question"]
    print(question)
    check_result = recom_checker.invoke({"question": question})
    print(check_result)
    grade = check_result.binary_score
    if grade == "yes":
        return "recommend_system"
    else:
        return "no"

def calculator(state):
    print("--- Doing Calcualtions ------")
    question = state["question"]
    result = insurance_calculator(question)
    print(result)
    return {"generation": result}


### Edges ###

## Routing 

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source.datasource == "recommender_question":
        print("---ROUTE QUESTION TO RECOMMEND---")
        return "recommender_question"
    elif source.datasource == "calculator":
        print("---ROUTE TO CACULATIONS-----")
        return "calculator"

## decide when to generate

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

## grade_generation_v_documents_and_question

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS")
        return "useful"

## Web search node

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = tool_web_search_chain.invoke(question)
    if "not relevant" in docs:
        return docs
    else: 
        web_results = []
        for respon in docs:
            if isinstance(respon[1], str):
                web_results.append(respon[1])
        return {"generation": web_results[0]}


def recommend_system(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains recommended documents
    """
    print("---Recommendation---")
    question = state["question"]

    # Retrieval
    documents = recommend_policy(question)
    return {"generation": documents}


## LangGraph

memory = MemorySaver()
workflow = StateGraph(GraphState)


# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query) # transform_query
workflow.add_node("recommend_system", recommend_system) # recommend_system
workflow.add_node("recommender_question",recommender_question)
workflow.add_node("missing_information",missing_information)
workflow.add_node("calculator",calculator)


# Build graph

workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "recommender_question": "recommender_question",
        "calculator":"calculator"
    },
)

workflow.add_conditional_edges(
    "recommender_question",
    policy_recommend_checker,
    {
        "recommend_system": "recommend_system",
        "no": "missing_information",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

workflow.add_edge("web_search", END)
workflow.add_edge("recommend_system", END)
workflow.add_edge("missing_information", END)
workflow.add_edge("calculator",END)

memory = MemorySaver()

# Compile
model_app = workflow.compile(checkpointer=memory)
# Run

from pprint import pprint
def get_insurance_response(query):
    config = {
        "configurable": {
            "thread_id": 1,
        }
    }

    inputs = {
        "question": query
    }
    
    results_dict = {}

    # Process input and get the output
    for output in model_app.stream(inputs, config, stream_mode="values"):
        for key, value in output.items():
            # Store the key-value pair in the dictionary
            results_dict[key] = value  # Update the dictionary with the current key-value pair

    # Return the generated response
    return results_dict.get("generation", "No response generated")


import os
import random
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Directory for PDF files
PDF_FOLDER = os.path.join(os.getcwd(), 'policies')

# API endpoint to list all PDFs in the policies folder
@app.route('/api/get_pdfs', methods=['GET'])
def get_pdfs():
    try:
        # List all files in the policies folder
        files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
        return jsonify({'pdfs': files})
    except Exception as e:
        return jsonify({'error': str(e)})

# Serve a specific PDF
@app.route('/api/pdf/<filename>', methods=['GET'])
def serve_pdf(filename):
    try:
        return send_from_directory(PDF_FOLDER, filename)
    except Exception as e:
        return jsonify({'error': str(e)})

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# API for the chatbot


def extract_follow_up_questions_and_text(text):
    # Regular expression to match numbered questions
    question_pattern = r'(\d+\.\s+.*?\?)'
    
    # Split the text into sections: follow-up questions and the rest
    split_text = re.split(question_pattern, text)
    
    # Initialize variables to store follow-up questions and other text
    follow_up_questions = []
    other_text = ""
    
    # Iterate through the split sections
    for i, part in enumerate(split_text):
        if re.match(r'\d+\.\s+.*?\?', part):  # If the part matches a question format
            follow_up_questions.append(part.strip())
        else:
            other_text += part.strip() + " "  # Add the non-question text
    
    return follow_up_questions, other_text.strip()

def format_follow_up_questions(questions):
    # Format each question into the required dictionary format
    formatted_questions = [{"question": q.lower().replace("?", "")} for q in questions]
    return formatted_questions


@app.route('/api/chat', methods=['POST'])
def chatbot():
    user_message = request.json.get('message', '').lower().strip()
    print(user_message)
    
    # Initialize reply as None
    reply = None
    
    if user_message == "seeking personalized policy advice tailored just for you":
        print(user_message)
        reply = """To provide personalized policy to you , can you please provide basic information like 1. Name 2. Age 3. Sum assured or expecting returns 4. Financial goal (looking for life insurance or retirement plan).
        
        Follow-up question examples:
        1. I am 42 years old and looking for a retirement policy with a sum assured of 150,000 to support my financial security in later years?
        2. I am 29 years old and interested in a life insurance policy with a sum assured of 75,000 to provide a safety net for my family?
        3. I am 50 years old and searching for a retirement plan with a sum assured of 200,000 to ensure financial stability in my retirement?
        """
    elif user_message == "interested in know more about insurance policies":
        reply = """
I have a below insurance plans available for you. How can i help with you this questions.

1. LIC Jeevan Ustav
2. LIC New Pension Plus
3. LIC Bima shree
4. LIC jeevan Akshay
5. LIC jeevan umang

Follow-up question examples:
1. What is LIC jeevan Ustav?
2. what is New Pension Plus?
3. What are the benfits of LIC BIMA and LIC jeevan umang?
"""
    else:
        # Default reply when no specific match
        reply = get_insurance_response(user_message)

    # Proceed only if reply is not None
    follow_up_questions, other_text = extract_follow_up_questions_and_text(reply)
    formatted_follow_up_questions = format_follow_up_questions(follow_up_questions)

    # Return other_text with preserved formatting for the UI
    return jsonify({
        'reply': render_template_string("<div class='response'>{{ text|safe }}</div>", text=other_text),
        'questions': formatted_follow_up_questions
    })


if __name__ == '__main__':
    app.run(debug=True)
