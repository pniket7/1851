import openai
import streamlit as st
import pickle
import time
import numpy as np
import pandas as pd
from typing import Optional, Union

def ErrorHandler(f, *args, **kwargs):
    def wrapper(*args, **kwargs):
        while True:
            try:
                f(*args, **kwargs)
                break
            # RateLimitError
            except openai.error.RateLimitError:
                print('Rate limit exceeded. I will be back shortly, please wait for a minute.')
                time.sleep(60)
            # AuthenticationError
            except openai.error.AuthenticationError as e:
                print(e)
                raise
    return wrapper

class ChatSession:
    completions = {
        1: dict(
            completion=openai.ChatCompletion,
            model="gpt-3.5-turbo",
            text='message.content',
            prompt='messages'
        ),
        0: dict(
            completion=openai.Completion,
            model="text-davinci-003",
            text='text',
            prompt='prompt'
        )
    }

    def __init__(self, gpt_name='GPT') -> None:
        # History of all messages in the chat.
        self.messages = []
        # History of completions by the model.
        self.history = []
        # The name of the model.
        self.gpt_name = gpt_name

    def chat(self, user_input: Optional[Union[dict, str]] = None, verbose=True, *args, **kwargs):
        """ Say something to the model and get a reply. """
        completion_index = 0 if kwargs.get('logprobs', False) or kwargs.get('model') == 'text-davinci-003' else 1
        completion = self.completions[completion_index]
        user_input = self.__get_input(user_input=user_input, log=True)
        user_input = self.messages if completion_index else self.messages[-1]['content']
        kwargs.update({completion['prompt']: user_input, 'model': completion['model']})
        if completion_index == 1:
            kwargs.update({'temperature': 0.5})
        self.__get_reply(completion=completion['completion'], log=True, *args, **kwargs)
        self.history[-1].update({'completion_index': completion_index})
        if verbose:
            self._call_(1)

    def display_probas(self, reply_index):
        """ Display probabilities of each word for the given reply by the model. """

        history = self.history[reply_index]
        assert not history.completion_index
        probas = history.logprobs.top_logprobs
        return pd.concat([
                pd.DataFrame(data=np.concatenate([[list(k.keys()), np.exp2(list(k.values())).round(2)]]).T,
                             columns=[str(i), f'{i}_proba'],
                             index=[f'candidate_{j}' for j in range(len(probas[0]))]
                            ) for i, k in enumerate(probas)], axis=1).T

    def inject(self, line, role):
        """ Inject lines into the chat. """

        self.__log(message={"role": role, "content": line})

    def clear(self, k=None):
        """ Clears session. If provided, last k messages are cleared. """
        if k:
            self.messages = self.messages[:-k]
            self.history = self.history[:-k]
        else:
            self._init_()

    def save(self, filename):
        """ Saves the session to a file. """

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename):
        """ Loads up the session. """

        with open(filename, 'rb') as f:
            temp = pickle.load(f)
            self.messages = temp.messages
            self.history = temp.history

    def merge(self, filename):
        """ Merges another session from a file with this one. """

        with open(filename, 'rb') as f:
            temp = pickle.load(f)
            self.messages += temp.messages
            self.history += temp.history

    def __get_input(self, user_input, log: bool = False):
        """ Converts user input to the desired format. """

        if user_input is None:
            user_input = input("> ")
        if not isinstance(user_input, dict):
            user_input = {"role": 'user', "content": user_input}
        if log:
            self.__log(user_input)
        return user_input

    @ErrorHandler
    def __get_reply(self, completion, log: bool = False, *args, **kwargs):
        """ Calls the model. """
        reply = completion.create(*args, **kwargs).choices[0]
        if log:
            if hasattr(reply, 'message'):
                self.__log(message=reply.message, history=reply)
            else:
                self.__log(message={"role": 'assistant', "content": reply.text}, history=reply)
        return reply

    def __log(self, message: dict, history=None):
        self.messages.append(message)
        if history is not None:
            assert isinstance(history, dict)
            self.history.append(history)

    def _call_(self, k: Optional[int] = None):
        """ Display the full chat log or the last k messages. """

        k = len(self.messages) if k is None else k
        for msg in self.messages[-k:]:
            message = msg['content']
            who = {'user': 'User: ', 'assistant': f'{self.gpt_name}: '}[msg['role']]
            print(who + message.strip() + '\n')

@ErrorHandler
def update_investor_profile(session, investor_profile: dict, questions: list[str], verbose: bool = False):
    ask_for_these = [i for i in investor_profile if not investor_profile[i]]
    n_limit = 20
    temp_reply = openai.ChatCompletion.create(messages=session.messages.copy(), model='gpt-3.5-turbo').choices[0].message.content
    for info_type in ask_for_these:
        choices = [*map(lambda x: x.message.content, openai.ChatCompletion.create(messages=
                                        session.messages +
                                        [{"role": "assistant", "content": temp_reply}] +
                                        [{"role": "user", "content": f'Do you know my {info_type} based on our conversation so far? Yes or no:'}],
                                        model='gpt-3.5-turbo', n=n_limit, max_tokens=30).choices)]
        if verbose:
            print('1:')
            print({i: round(choices.count(i) / len(choices), 2) for i in pd.unique(choices)})
        if np.any([*map(lambda x: 'yes' in x.lower(), choices)]):
            choices = [*map(lambda x: x.message.content, openai.ChatCompletion.create(messages=
                                        session.messages +
                                        [{"role": "assistant", "content": temp_reply}] +
                                        [{"role": "user", "content": questions[info_type]}],
                                        model='gpt-3.5-turbo', n=n_limit, max_tokens=50).choices)]
            if verbose:
                print('2:')
                print({i: round(choices.count(i) / len(choices), 2) for i in pd.unique(choices)})
            if np.any([*map(lambda x: 'yes' in x.lower(), choices)]):
                investor_profile[info_type] = 'yes'
            elif np.any([*map(lambda x: 'no' in x.lower(), choices)]):
                investor_profile[info_type] = 'no'

def initialize_sessionAdvisor():
    advisor = ChatSession(gpt_name='Advisor')
    advisor.inject(
        line="You are a financial advisor at a bank selling bank loans to the users.Your name is Ello bot. Your main goal is to converse with the user and guide him to buy a loan from the bank. If the user is convinced by you for taking a loan, tell the uer that you are handing over the conversation to a human representative of the bank for further steps and end the conversation. If the user is not convinced to take the loan ,try to convince him. If the user is still not convinced, thank the user and end the conversation. Start the conversation by inquiring whether the user is seeking loan for himself or on behalf of his client as a broker.Also ask this pre screening questions at the start of the conversation to understand the user's loan requirements.Here are the pre screening questions.1. Could you let us know which type of loan you're interested in, commercial or residential? 2. Are you planning to purchase a new property or refinance an existing one? 3. Can you tell us about your status,are you applying as an individual, a company, or as an investor? 4. Will your loan be primarily income based or asset based? 5. Do you have any experience in real estate, particularly in the commercial sector? 6. Could you provide some insights about the property you're interested in, like its condition, value, and intended use? 7. Where is the property located? Is it in an urban, suburban, or rural area? 8. For a complete understanding of your application, could you share your FICO score, if you know it? The questions should be asked one at a time, to not overwhelm the user. Be attentive to the user's needs and goals. Be brief in your responses. Access the user needs and gather information from the user about what type of loan he wants to take from the bank. Guide the user accordingly and if he is convinced, handover the user to the human representative and close the conversation, otherwise thank the user and close the conversation. Proceed with follow-up questions based solely on the user's immediate response, maintaining a strictly sequential flow. Ask one question at a time, waiting for and responding to each user input individually. Ensure that each response from the advisor contains only a single query or request for information, refraining from posing multiple questions or requests within the same reply. Keep your responses short. Use proper conversational language to create a personalized experience.Assess the user's needs and goals accurately by asking targeted questions and listening carefully to the user's responses. Be empathetic and understanding of the user's situation, acknowledging their concerns and providing reassurance when needed. Communicate clearly and concisely, avoiding complex language. Break down information into digestible chunks and provide explanations in a straightforward manner.At the end of the conversation before handing over to human representative,you should request the user's name and contact information. After that, you should summarize the responses provided by the user and indicate that you will connect them with a live agent.",
        role="user"
    )
    advisor.inject(line="Ok.", role="assistant")
    return advisor

if "my_text" not in st.session_state:
    st.session_state.my_text = ""

def submit():
    st.session_state.my_text = st.session_state.widget
    st.session_state.widget = ""
    
def main():
    st.title('ELLO-Financial Advisor Chatbot')

    # Load the OpenAI API key from Streamlit secrets
    openai.api_key = st.secrets["api_key"]

    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize sessionAdvisor if it doesn't exist or is set to None
    if "sessionAdvisor" not in st.session_state or st.session_state.sessionAdvisor is None:
        st.session_state.sessionAdvisor = initialize_sessionAdvisor()

    # Function to update chat history display
    # Function to update chat history display
    def update_chat_display(messages):
        chat_messages = ""
        if messages:
            for message in messages:
                role_color = "#0084ff" if message["role"] == "user" else "#9400D3"
                alignment = "right" if message["role"] == "user" else "left"
                content = message["content"]
                # If the message is from the bot and contains multiple sentences, split it into bullet points
                if message["role"] == "bot" and "." in content:
                    sentences = content.split(".")
                    bullet_points = "<ul>"
                    for sentence in sentences:
                        bullet_points += f"<li>{sentence.strip()}</li>"
                    bullet_points += "</ul>"
                    content = bullet_points
                chat_messages += f'<div style="text-align: {alignment}; margin-bottom: 10px;"><span style="background-color: {role_color}; color: white; padding: 8px 12px; border-radius: 20px; display: inline-block; max-width: 70%;">{content}</span></div>'
        return chat_messages



    # Display the chat history and bot thinking message together
    chat_container = st.empty()
    chat_and_thinking_display = update_chat_display(st.session_state.chat_history) + '<div id="thinking"></div>'
    chat_container.markdown(f'<div style="border: 1px solid black; padding: 10px; height: 400px; overflow-y: scroll; position: relative;">{chat_and_thinking_display}</div>', unsafe_allow_html=True)

    # Accept user input
    with st.form(key="my_form"):
        placeholder=st.empty()
        user_input = placeholder.text_input("Type your message here...",key="widget")
        # Create a button to send the user inputs
        if st.form_submit_button("Send",on_click=submit) and st.session_state.my_text:
            # Add the user's message to the chat history
            st.session_state.chat_history.append({"role": "user", "content": st.session_state.my_text})

            # Display "Bot is thinking..." message while bot generates response
            with st.spinner(text="Bot is thinking..."):
                # Update the chat session with the user's input
                st.session_state.sessionAdvisor.chat(user_input=st.session_state.my_text, verbose=False)

                # Get the chatbot's response from the last message in the history
                advisor_response = st.session_state.sessionAdvisor.messages[-1]['content'] if st.session_state.sessionAdvisor.messages else ""

                # Remove newlines and extra spaces from the response
                advisor_response = advisor_response.replace('\n', ' ').strip()

                # Add the bot's response to the chat history
                st.session_state.chat_history.append({"role": "bot", "content": advisor_response})

            # Display the updated chat history including new messages
            chat_and_thinking_display = update_chat_display(st.session_state.chat_history) + '<div id="thinking"></div>'
            chat_container.markdown(f'<div style="border: 1px solid black; padding: 10px; height: 400px; overflow-y: scroll; position: relative;">{chat_and_thinking_display}</div>', unsafe_allow_html=True)
        
    
    # Create a button to start a new conversation
    if st.button("New Chat"):
        # Clear the chat history to start a new conversation
        st.session_state.chat_history = []

        # Reinitialize sessionAdvisor for a new conversation
        st.session_state.sessionAdvisor = initialize_sessionAdvisor()

        # Clear the chat container for the new conversation
        chat_container.markdown("", unsafe_allow_html=True)
        st.markdown("New conversation started. You can now enter your query.")

    # Create a button to exit the current conversation
    if st.button("Exit Chat"):
        # Clear the chat history to exit the chat
        st.session_state.chat_history = []

        # Clear the chat container for the exited chat
        chat_container.markdown("", unsafe_allow_html=True)
        st.markdown("Chatbot session exited. You can start a new conversation by clicking the 'New Chat' button.")

if __name__ == "__main__":
    main()
