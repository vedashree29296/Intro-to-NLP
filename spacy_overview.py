import streamlit as st
import spacy
from spacy import displacy
import pandas as pd


DEFAULT_TEXT = "Mark Zuckerberg is the CEO of Facebook."
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


st.title("Basics of NLP")
st.image("NLPBASICS.png", width=800)

spacy_model = "en_core_web_md"
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model("en_core_web_md")
model_load_state.empty()
text = st.text_area("Text to analyze", DEFAULT_TEXT)
doc = process_text(spacy_model, text)

if "ner" in nlp.pipe_names:
    st.header("Named Entities")
    st.sidebar.header("Named Entities")
    default_labels = ["PERSON", "ORG", "GPE", "LOC"]
    labels = st.sidebar.multiselect(
        "Entity labels", nlp.get_pipe("ner").labels, default_labels
    )
    html = displacy.render(doc, style="ent", options={"ents": labels})
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
    if "entity_linker" in nlp.pipe_names:
        attrs.append("kb_id_")
    data = [
        [str(getattr(ent, attr)) for attr in attrs]
        for ent in doc.ents
        if ent.label_ in labels
    ]
    df = pd.DataFrame(data, columns=attrs)
    st.dataframe(df)

if "textcat" in nlp.pipe_names:
    st.header("Text Classification")
    st.markdown(f"> {text}")
    df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
    st.dataframe(df)

st.header("Token attributes")

attrs = [
    "idx",
    "text",
    "lemma_",
    "pos_",
    "tag_",
    "dep_",
    "head",
    "ent_type_",
    "ent_iob_",
    "shape_",
    "is_alpha",
    "is_ascii",
    "is_digit",
    "is_punct",
    "like_num",
]
data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
df = pd.DataFrame(data, columns=attrs)
st.dataframe(df)

if "parser" in nlp.pipe_names:
    st.header("Dependency Parse & Part-of-speech tags")
    st.sidebar.header("Dependency Parse")
    split_sents = st.sidebar.checkbox("Split sentences", value=True)
    collapse_punct = st.sidebar.checkbox("Collapse punctuation", value=True)
    collapse_phrases = st.sidebar.checkbox("Collapse phrases")
    compact = st.sidebar.checkbox("Compact mode")
    options = {
        "collapse_punct": collapse_punct,
        "collapse_phrases": collapse_phrases,
        "compact": compact,
    }
    docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
    for sent in docs:
        html = displacy.render(sent, options=options)
        # Double newlines seem to mess with the rendering
        html = html.replace("\n\n", "\n")
        if split_sents and len(docs) > 1:
            st.markdown(f"> {sent.text}")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

st.header("Libraries and Tools:")
st.image("nlp_libs.png", width=800)


st.header("Applications:")
vector_size = nlp.meta.get("vectors", {}).get("width", 0)
if vector_size:
    st.header("Similarity")
    text1 = st.text_input("Text or word 1", "apple")
    text2 = st.text_input("Text or word 2", "orange")
    doc1 = process_text(spacy_model, text1)
    doc2 = process_text(spacy_model, text2)
    similarity = doc1.similarity(doc2)
    if similarity > 0.5:
        st.success(f"{round(similarity * 100,2)} %")
    else:
        st.error(f"{round(similarity * 100,2)} %")

st.header("Co-reference")
html = "<a href='https://huggingface.co/coref/'>Coref Resolution Demo</a>"
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

st.header("Sentiment Analysis")
html = "<a href='https://monkeylearn.com/sentiment-analysis-online/'>Sentiment Analysis Demo</a>"
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

st.header("Text Generation")
html = "<a href='https://transformer.huggingface.co/doc/distil-gpt2'>Text Generation Demo</a>"
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

st.header("Question Answering")
html = "<a href='https://demo.deeppavlov.ai/#/en/textqa'>Question Answering Demo</a>"
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


st.header("Basics of SpaCY")
st.subheader("Installing SpaCY")
st.markdown("```pip install spacy```")

st.subheader("Installing SpaCY Model")
st.markdown("```python -m spacy download en_core_web_md```")

st.subheader("Loading SpaCY Model")
st.markdown("```import spacy ```")
st.markdown("```nlp=spacy.load('en_core_web_md')```")

st.subheader("Using SpaCY Model")
st.write(f"Sample Text: {text}")
with st.echo():
    doc = nlp(text)
    for token in doc:
        st.write("token: ", token.text, "POS: ", token.pos_, "LEMMA: ", token.lemma_)

st.write("Entities")
with st.echo():
    for ent in doc.ents:
        st.write("entity: ", ent.text, "label: ", ent.label_)
