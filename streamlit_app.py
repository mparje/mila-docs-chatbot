import logging

import pandas as pd
import streamlit as st
from buster.busterbot import Buster
from gradio.utils import highlight_code
from markdown_it import MarkdownIt
from mdit_py_plugins.footnote.index import footnote_plugin

import cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# initialize buster with the config in cfg.py (adapt to your needs) ...
buster: Buster = Buster(cfg=cfg.buster_cfg, retriever=cfg.retriever)


def get_markdown_parser() -> MarkdownIt:
    """Modified method of https://github.com/gradio-app/gradio/blob/main/gradio/utils.py#L42

    Removes the dollarmath_plugin to render Latex equations.
    """
    md = (
        MarkdownIt(
            "js-default",
            {
                "linkify": True,
                "typographer": True,
                "html": True,
                "highlight": highlight_code,
            },
        )
        # .use(dollarmath_plugin, renderer=tex2svg, allow_digits=False)
        .use(footnote_plugin).enable("table")
    )

    # Add target="_blank" to all links. Taken from MarkdownIt docs: https://github.com/executablebooks/markdown-it-py/blob/master/docs/architecture.md
    def render_blank_link(self, tokens, idx, options, env):
        tokens[idx].attrSet("target", "_blank")
        return self.renderToken(tokens, idx, options, env)

    md.add_render_rule("link_open", render_blank_link)

    return md


def check_auth(username: str, password: str) -> bool:
    """Basic auth, only supports a single user."""
    # TODO: update to better auth
    is_auth = username == cfg.username and password == cfg.password
    logger.info(f"Log-in attempted. {is_auth=}")
    return is_auth


def format_sources(matched_documents: pd.DataFrame) -> str:
    if len(matched_documents) == 0:
        return ""

    sourced_answer_template: str = (
        """📝 Here are the sources I used to answer your question:<br>""" """{sources}<br><br>""" """{footnote}"""
    )
    source_template: str = """[🔗 {source.title}]({source.url}), relevance: {source.similarity:2.1f} %"""

    matched_documents.similarity = matched_documents.similarity * 100
    sources = "<br>".join([source_template.format(source=source) for _, source in matched_documents.iterrows()])
    footnote: str = "I'm a bot 🤖 and not always perfect."

    return sourced_answer_template.format(sources=sources, footnote=footnote)


def add_sources(history, response):
    documents_relevant = response.documents_relevant

    if documents_relevant:
        # add sources
        formatted_sources = format_sources(response.matched_documents)
        history.append([None, formatted_sources])

    return history


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def chat(history):
    user_input = history[-1][0]

    response = buster.process_input(user_input)

    history[-1][1] = ""

    for token in response.completion.completor:
        history[-1][1] += token

        yield history, response


st.set_page_config(page_title="Buster 🤖: A Question-Answering Bot for your documentation")

st.markdown("<h3><center>Buster 🤖: A Question-Answering Bot for your documentation</center></h3>", unsafe_allow_html=True)

chatbot = st.empty()

st.markdown("This application uses GPT to search the docs for relevant info and answer questions.")

question = st.text_input(label="What's your question?", help="Ask a question to AI stackoverflow here...")
submit = st.button("Send")

examples = [
    "How can I run a job with 2 GPUs?",
    "What kind of GPUs are available on the cluster?",
    "What is the $SCRATCH drive for?",
]

if st.checkbox("Show examples"):
    st.markdown("#### Examples")
    st.code(examples)

history = []

if submit:
    history = user(question, history)
    history, response = next(chat(history))
    history = add_sources(history, response)

for user_input, bot_response in history:
    if user_input:
        st.text_area("User", value=user_input, key=user_input, height=100)
    if bot_response:
        st.text_area("Buster", value=bot_response, key=bot_response, height=100)


