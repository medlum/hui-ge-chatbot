from streamlit_chat import message
import streamlit as st
from streamlit_extras.bottom_container import bottom
import streamlit_antd_components as sac
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from huggingface_hub.errors import OverloadedError
from langchain.agents import AgentExecutor, create_react_agent
from utils_agent_tools import *
from utils_prompt import *

# ---------set up page config -------------#
st.set_page_config(page_title="辉哥会聊天",
                   layout="wide", page_icon="👲")


# ---- set up creative chat history ----#
chat_msg = StreamlitChatMessageHistory(key="chat_key")
chat_history_size = 5

model = "Qwen/Qwen2.5-72B-Instruct"
# ---------set up for creative mode  -------------#
# Initialize LLM for creative mode
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task="text-generation",
    max_new_tokens=1000,
    do_sample=False,
    temperature=0.5,
    repetition_penalty=1.1,
    return_full_text=False,
    top_p=0.2,
    top_k=40,
    huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
)

# ---------set up general memory  -------------#
conversational_memory = ConversationBufferMemory(
    memory_key='chat_history',
    chat_memory=chat_msg,
    k=chat_history_size,
    return_messages=True
)

# ---------set up agent with tools  -------------#
react_agent = create_react_agent(llm, toolkit, prompt)


executor = AgentExecutor(
    agent=react_agent,
    tools=toolkit,
    memory=conversational_memory,
    max_iterations=10,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs=agent_kwargs,
)

for index, msg in enumerate(chat_msg.messages):

    if index == 0:  # skip none
        continue
    # bot's message is in even position as welcome message is added at initial
    if index % 2 == 0:
        message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", ""),
                is_user=True,
                key=f"user{index}",
                avatar_style="personas",
                seed="Robert")

    # user's message is in odd position
    else:
        message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", ""),
                is_user=False,
                key=f"bot{index}",
                avatar_style="personas",
                seed="Sophia",
                allow_html=True,
                is_table=True,)

if prompt := st.chat_input("Ask me a question..."):
    # show prompt message
    message(f'{prompt}',
            is_user=True,
            key=f"user",
            avatar_style="personas",
            seed="Robert",)

with st.spinner("Generating"):
    # st.write(response)
    response = executor.invoke(
        {'input': f'<|im_start|>{prompt}<|im_end|>'})

# response = response["text"]
message(response['output'].replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", ""),
        is_user=False,
        key=f"bot_2",
        avatar_style="personas",
        seed="Sophia",
        allow_html=True,
        is_table=True,)
