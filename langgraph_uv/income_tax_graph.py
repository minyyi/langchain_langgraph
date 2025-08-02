# %%
from dotenv import load_dotenv
load_dotenv()

# %%
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# %%
from langchain_chroma import Chroma

# vector_store = Chroma.from_documents(
#     documents=document_list,
#     embedding=embedding,
#     collection_name="chroma_tax",
#     persist_directory="./chroma_tax"
# )

vector_store = Chroma(
    collection_name="chroma_tax",
    embedding_function=embedding,
    persist_directory="./chroma_tax"
)

# %%
retriever = vector_store.as_retriever(search_kwargs={"k" : 3})
retriever

# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]     #retriever에 있는 문서에서 읽어오겠다
    answer : str
    
graph_builder = StateGraph(AgentState)

# %%
def retrieve(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다. 
    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state
        
    Returns:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.
    """
    
    query = state['query']
    docs = retriever.invoke(query)
    return {'context': docs}        #크로마디비에서 검색된 문서 


    

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o')

# %%
from langchain import hub

generate_prompt = hub.pull('rlm/rag-prompt')

def generate(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다.
    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state

    Returns:
        AgentState: 생성된 응답을 포함하는 state를 반환합니다.
    """
    
    context = state['context']
    query = state['query']
    
    rag_chain = generate_prompt | llm
    
    response = rag_chain.invoke({
        'question': query,
        'context' : context,
    })
    
    return {'answer': response}

# %%
from langchain import hub
from typing import Literal

#문서 관련성 체크 
doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")

#분기문. 특정 노드 언급이 제일 좋음. 
def check_doc_relevance(state: AgentState) -> Literal['generate', 'rewrite']:
    """
    주어진 state를 기반으로 문서의 관련성을 판단합니다. 
    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.
        
    Returns: 
        Literal['generate', 'rewrite']: 문서가 관련성이 높으면 'generate', 그렇지 않으면 'rewrite'를 반환합니다.
        
    
    """
    
    query = state['query']
    context = state['context']
    
    doc_relevance_chain = doc_relevance_prompt | llm 
    
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})
    
    if response['Score'] == 1:
        # print('관련성 : relevant')
        return 'relevant'
    # print('관련성 : irrelevant')
    return 'irrelevant'
    

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ["사람과 관련된 표현 -> 거주자"]

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
사전: {dictionary}
질문: {{query}}     
""")
#f-string사용해서 {}한번 더 해줘야함. 
def rewrite(state: AgentState) -> AgentState:
    """
    사용자의 질문을 사전을 고려하여 변경합니다.

    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트 현재 state.

    Returns:
        AgentState: 변경된 질문을 포함하는 state를 반환합니다. 
    """
    
    query = state['query']
    #chain
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()        
                                            #전체를 str로 반환. 이게 다시 질문이 되야하기 때문
    
    response = rewrite_chain.invoke({'query': query})
    
    return {'query': response}

# %%
from langchain_core.output_parsers import StrOutputParser

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}      
student_answer: {student_answer}      
""")
# documents: {documents}      #검색기로 가져온 문서 
# student_answer: {student_answer}        #generate가 만든 답변

#hallucination 검사할 때는 답이 딱 정해져야하기 때문에 llm 따로 해야함. 
hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']        #generate 노드에서 만들어준 결과값
    context = state['context']
    context = [doc.page_content for doc in context]     #내용만 뽑아내서 리스트 형식으로 
    
    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()

    response = hallucination_chain.invoke({'student_answer': answer, 'documents': context})
    # print('거짓말: ', response)
    return response     #'hallucinated' | 'not hallucinated'

# %%
# query = "연봉 5천만원 거주자의 소득세는 얼마인가요?"
# # query = "과세 기간은 어떻게 되나요?"
# context = retriever.invoke(query)
# generate_state = {'query': query, 'context': context}
# answer = generate(generate_state)
# print(f"answer: {answer}")

# hallucination_state = {'answer': answer, 'context': context}
# check_hallucination(hallucination_state)

# %%
from langchain import hub

helpfulness_prompt = hub.pull('langchain-ai/rag-answer-helpfulness')

def check_helpfulness_grader(state: AgentState) -> str:
    """
    사용자의 질문에 기반하여 생성된 답변의 유용성을 평가합니다. 
    Args:
        state (AgentState): 사용자의 질문과 생성도니 답변을 포함한 에이전트의 현재 state값 의미

    Returns:
        str: 답변이 유용하다고 판단되면 'helpful', 그렇지 않으면 'unhelpful'을 반환합니다. 
    """
    
    query = state['query']
    answer = state['answer']
    
    helpfulness_chain = helpfulness_prompt | llm
    response = helpfulness_chain.invoke({'question': query, 'student_answer': answer})
    
    if response['Score'] == 1:
        # print('유용성: helpful')
        return 'helpful'

    # print('유용성: unhelpful')
    return 'unhelpful'
    
#conditional_edge를 연속으로 사용하지 않고 노드로서 역할을 하기위해 추가함. 
def check_helpfulness(state: AgentState) -> AgentState:
    """
    유용성을 확인하는 자리 표시자 함수 입니다. 
    graph에서 conditional_edge를 연속으로 사용하지 않고 
    가독성을 높이기 위해 node를 추가해서 사용합니다. 
    Args:
        state (AgentState): 에이전트의 현재 state

    Returns:
        AgentState: 변경되지 않은 state 반환
    """
    
    return state

# # %%
# # query = "연봉 5천만원 거주자의 소득세는 얼마인가요?"
# query = "연봉 5천?"
# context = retriever.invoke(query)
# generate_state = {'query': query, 'context': context}
# answer = generate(generate_state)

# helpfulness_state = {'query': query, 'answer': answer}
# check_helpfulness_grader(helpfulness_state)

# %%
#node
graph_builder = StateGraph(AgentState)

graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)



# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'retrieve')
graph_builder.add_conditional_edges(
    'retrieve', 
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)
graph_builder.add_conditional_edges(
    'generate', 
    check_hallucination,
    {
        'not hallucinated' : 'check_helpfulness',       #노드라서 이동가능!
        'hallucinated' : 'generate',
    }
)
graph_builder.add_conditional_edges(
    'check_helpfulness', 
    check_helpfulness_grader,
    {
        'helpful' : END,     
        'unhelpful' : 'rewrite',
    }
)
graph_builder.add_edge('rewrite', 'retrieve')



# %%
# from IPython.display import Image, display
graph = graph_builder.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))

# %%
# initial_state = {'query': '연봉 5천만원 직장인의 소득세는?'}
# graph.invoke(initial_state)

# # %%
# initial_state = {'query': '점심 뭐먹을래?'}
# graph.invoke(initial_state)

# %%



