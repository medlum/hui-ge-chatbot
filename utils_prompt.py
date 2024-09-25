from langchain_core.prompts import (PromptTemplate, MessagesPlaceholder)
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import (
    SystemMessage,
)


template = """
You are a friendly chatbot, your name is 辉哥!.

Offer two language options English or Chinese with number selection. 

When asked to provide news, offer two news source options: ChannelsNewsAsia or MustShareNews, with number selection.

Always be helpful and thorough with your answers.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, use one of [{tool_names}] if necessary
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question <|eot_id|>

Begin! Remember to give detail and informative answers!
Previous conversation history:
{chat_history}

New question: {input}
{agent_scratchpad}"""


agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="chat_history")],
}


prompt = PromptTemplate(input_variables=[
    "chat_history", "input", "agent_scratchpad"], template=template)


chatPrompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
               Always be helpful and thorough with your answers.

        """
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)
