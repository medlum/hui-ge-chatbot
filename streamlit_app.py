from streamlit_chat import message
import streamlit as st
from utils_agent_tools import *
from utils_prompt import *
from utils_tts import *
from streamlit_extras.bottom_container import bottom
import streamlit_antd_components as sac
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from huggingface_hub.errors import OverloadedError
import re

# ---------set up page config -------------#
st.set_page_config(page_title="辉哥会聊天",
                   layout="wide", page_icon="👲")

# tab = select_tab()

# if tab == "chat":
# ---------set up toggle at the bottom -------------#
with bottom():
    mode_toggle = st.toggle(label="Creative Mode (创意模式)", value=False)

# ---- set up creative chat history ----#
chat_msg = StreamlitChatMessageHistory(key="chat_key")
chat_history_size = 5

# ---------set up LLM  -------------#
model = "Qwen/Qwen2.5-72B-Instruct"

# initialise LLM for agents and tools
llm_factual = HuggingFaceEndpoint(
    repo_id=model,
    max_new_tokens=1000,
    do_sample=False,
    temperature=0.1,
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
react_agent = create_react_agent(llm_factual, toolkit, prompt)

executor = AgentExecutor(
    agent=react_agent,
    tools=toolkit,
    memory=conversational_memory,
    max_iterations=10,
    handle_parsing_errors=True,
    verbose=True,
    agent_kwargs=agent_kwargs,
)

# ---------set up for creative mode  -------------#
# Initialize LLM for creative mode
llm_creative = HuggingFaceEndpoint(
    repo_id=model,
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

# ------ set up the llm chain -----#
chat_llm_chain = LLMChain(
    llm=llm_creative,
    prompt=chatPrompt,  # located at utils_prompt.py
    verbose=True,
    memory=conversational_memory,
)

# ------ initial welcome message -------#

# set up session state as a gate to display welcome message
if 'initial_msg' not in st.session_state:
    st.session_state.initial_msg = 0

# if 0, add welcome message to chat_msg
if st.session_state.initial_msg == 0:
    part_day = get_time_bucket()  # located at utils_tts.py
    # welcome_msg = f"{part_day} How about some news headlines to start your day?"
    chat_msg.add_ai_message(f"{part_day} {welcome_msg}")
# ------ set up message from chat history  -----#

for index, msg in enumerate(chat_msg.messages):

    # bot's message is in even position as welcome message is added at initial
    if index % 2 == 0:
        message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", "").replace("AI:", "").replace("Human:", ""),
                is_user=False,
                key=f"bot{index}",
                avatar_style="personas",
                seed="Sophia",
                allow_html=True,
                is_table=True,)

    # user's message is in odd position
    else:
        message(msg.content.replace("<|im_start|>", "").replace("<|im_end|>", ""),
                is_user=True,
                key=f"user{index}",
                avatar_style="personas",
                seed="Robert")

    # set initial_msg to 0 in first loop
    if index == 0:
        st.session_state.initial_msg = 1

# ---------set up for creative mode  -------------#
if mode_toggle:
    # initialize response type as creative
    response_type = "creative"
else:
    # initialize response type as agents
    response_type = "agents"

# ------ set up user input -----#

if prompt := st.chat_input("Ask me a question..."):
    # show prompt message
    message(f'{prompt}',
            is_user=True,
            key=f"user",
            avatar_style="personas",
            seed="Robert",)

    # ---- if response_type is agent -----#

    if response_type == "agents":

        with st.spinner("Generating...（请等一下）"):

            try:

                # use {'input': f'{prompt}<|eot_id|>'})
                response = executor.invoke(
                    {'input': f'<|im_start|>{prompt}<|im_end|>'})

                message(response['output'].replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|eot_id|>", "").replace("<|endoftext|>", ""),
                        is_user=False,
                        key=f"bot_2",
                        avatar_style="personas",
                        seed="Sophia",
                        allow_html=True,
                        is_table=True,)

            except OverloadedError as error:

                st.write(
                    "HuggingFace🤗 inference engine is overloaded now. Try toggling to the creative mode in the meantime.")

    # ---- if response_type is creative -----#

    elif response_type == "creative":

        with st.spinner("Generating...（请等一下）"):

            try:

                # use {'human_input': f'{prompt}<|eot_id|>'})
                response = chat_llm_chain.invoke(
                    {'human_input': f'<|im_start|>{prompt}<|im_end|>'})

                # remove prompt format for better display
                edited_response = response["text"].replace("AI:", "")
                human = re.search(r"Human:.*|human:.*", edited_response)
                if human is not None:
                    # exclude "Human:" located at end of string
                    edited_response = edited_response[:human.start()]

                # show message
                message(edited_response,
                        is_user=False,
                        key=f"bot_2",
                        avatar_style="personas",
                        seed="Sophia",
                        allow_html=True,
                        is_table=True,)

            except OverloadedError as error:
                st.write(
                    "HuggingFace🤗 inference engine is overloaded now. Try toggling to the creative mode in the meantime.")
