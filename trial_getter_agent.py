# %%
# !pip install faiss-cpu
# !pip install langchain
# !pip install openai
# !pip install tiktoken

# %%
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader

# %%
import glob
trial_paths = glob.glob('/Users/ramonprieto/src/oncology_poc/raw/*/*.json')

# %%
import json
trial_documents = []
documents_to_load = 100
for trial_path in trial_paths[:documents_to_load]:
    trial = json.load(open(trial_path))
    nct_id = trial['FullStudy']['Study']['ProtocolSection']['IdentificationModule']['NCTId']
    trial_documents.append(Document(page_content=json.dumps(trial['FullStudy']), metadata={'nct_id': nct_id}))

# %%
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(trial_documents, embeddings)

# %%
query = "Radiotherapy for prostate cancer"
docs = db.similarity_search(query, k=1)

# %%
json.loads(docs[0].page_content)['Study']['ProtocolSection']['DescriptionModule']


