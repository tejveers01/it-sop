from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
from io import BytesIO
from os.path import basename
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import ibm_boto3
from ibm_botocore.client import Config
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

load_dotenv()

app = FastAPI(title="SOP Chatbot API", version="1.0.0")

COS_ENDPOINT = os.getenv("COS_ENDPOINT")
COS_API_KEY_ID = os.getenv("COS_API_KEY_ID")
COS_INSTANCE_CRN = os.getenv("COS_INSTANCE_CRN")
COS_BUCKET_NAME = os.getenv("COS_BUCKET_NAME")

cos = ibm_boto3.client(
    "s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_INSTANCE_CRN,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT,
)

MILVUS_USERNAME = os.getenv("MILVUS_USERNAME")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_GRPC_HOST = os.getenv("MILVUS_GRPC_HOST", "4c5f2692-40e6-4b8b-8cb9-84cc5aea4e78.cvbhm81d0dmnvl5rjek0.lakehouse.appdomain.cloud")
MILVUS_GRPC_PORT = os.getenv("MILVUS_GRPC_PORT", "30319")
MILVUS_URI = f"grpc://{MILVUS_GRPC_HOST}:{MILVUS_GRPC_PORT}"

try:
    connections.connect(
        alias="default",
        uri=MILVUS_URI,
        user=MILVUS_USERNAME,
        password=MILVUS_PASSWORD,
        secure=True
    )
    logging.info("✅ Connected to Milvus")
except Exception as e:
    logging.error(f"❌ Milvus connection failed: {e}")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
collection_name = "sop_collection"

if utility.has_collection(collection_name):
    collection = Collection(name=collection_name)
    collection.load()
    logging.info(f"✅ Loaded collection with {collection.num_entities} entities")
else:
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="sop_text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="sop_no", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="task", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="approvals", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="responsibility", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="timelines", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="remarks", dtype=DataType.VARCHAR, max_length=1000),
    ]
    schema = CollectionSchema(fields, description="SOP chatbot embeddings")
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    )
    collection.load()
    logging.info("✅ Created new Milvus collection")

full_df = None

class QueryRequest(BaseModel):
    query: str

class SOPResponse(BaseModel):
    sop_no: str
    task: str
    steps: str
    approvals: str
    responsibility: str
    timelines: str
    remarks: str
    summary: str
    match_type: str

@app.on_event("startup")
async def startup_event():
    global full_df
    logging.info("Loading SOP data on startup...")
    full_df = load_sop_data_from_cos()
    if full_df is not None and not full_df.empty:
        store_embeddings_to_milvus(full_df)
        logging.info(f"✅ Loaded {len(full_df)} SOP records")
    else:
        logging.warning("⚠️ No SOP data loaded")

def list_cos_files():
    try:
        response = cos.list_objects_v2(Bucket=COS_BUCKET_NAME)
        return [item["Key"] for item in response.get("Contents", [])]
    except Exception as e:
        logging.error(f"Error listing COS files: {e}")
        return []

def preprocess_excel(file_path):
    if isinstance(file_path, (bytes, BytesIO)):
        if isinstance(file_path, BytesIO):
            file_path.seek(0)
            df = pd.read_excel(file_path, header=None)
        else:
            df = pd.read_excel(BytesIO(file_path), header=None)
    else:
        df = pd.read_excel(file_path, header=None)

    header_row = None
    for i in range(min(5, len(df))):
        vals = df.iloc[i].astype(str).str.lower().str.strip().tolist()
        if any("task" in v for v in vals) or any("step" in v for v in vals):
            header_row = i
            break
    if header_row is None:
        header_row = 0

    if isinstance(file_path, (bytes, BytesIO)):
        if isinstance(file_path, BytesIO):
            file_path.seek(0)
            df = pd.read_excel(file_path, header=header_row)
        else:
            df = pd.read_excel(BytesIO(file_path), header=header_row)
    else:
        df = pd.read_excel(file_path, header=header_row)

    df.columns = (
        df.columns.map(str)
        .str.strip()
        .str.lower()
        .str.replace("\n", " ")
        .str.replace(r"\s+", " ", regex=True)
    )

    sop_no_col = next((c for c in df.columns if "bpf no" in c or "sop no" in c), None)
    task_col = next((c for c in df.columns if "task" in c), None)
    step_col = next((c for c in df.columns if "step" in c and "task" in c), None)

    if not step_col:
        step_col = next((c for c in df.columns if "step" in c), None)

    if not sop_no_col:
        df["sop_no"] = ""
    else:
        df.rename(columns={sop_no_col: "sop_no"}, inplace=True)
    
    if not task_col:
        raise KeyError("Missing 'Task' column")
    df.rename(columns={task_col: "task"}, inplace=True)
    
    if not step_col:
        raise KeyError("Missing 'Steps involved in this task' column")
    df.rename(columns={step_col: "steps involved in this task"}, inplace=True)

    step_idx = df.columns.get_loc("steps involved in this task")
    next_col = None
    if step_idx + 1 < len(df.columns):
        next_col = df.columns[step_idx + 1]

    if next_col:
        df["steps involved in this task"] = df.apply(
            lambda row: (
                f"{row['steps involved in this task']} {row[next_col]}".strip()
                if re.match(r"^[\da-zA-Z\.]+$", str(row['steps involved in this task']).strip())
                and str(row[next_col]).strip() not in ["", "nan"]
                else str(row['steps involved in this task']).strip()
            ),
            axis=1,
        )

    time_col = next((c for c in df.columns if "timeline" in c), None)
    if time_col:
        df.rename(columns={time_col: "timelines"}, inplace=True)
    else:
        df["timelines"] = ""

    keep_cols = [
        "sop_no", "task", "steps involved in this task",
        "approvals", "remarks", "timelines", "responsibility"
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    df = df.fillna("")
    for col in df.columns:
        df[col] = df[col].astype(str).replace(["nan", "None", "NaT"], "").str.strip()

    df["sop_no"] = df["sop_no"].replace("", None).ffill().fillna("")
    df["task"] = df["task"].replace("", None).ffill().fillna("")
    df = df[df["steps involved in this task"] != ""].copy()

    df["task"] = df["task"].apply(lambda x: re.sub(r"^\d+(\.\d+)*\s*", "", x).strip())
    df["sop_full_text"] = (df["sop_no"] + " " + df["task"]).str.lower().str.strip()

    return df

def load_sop_data_from_cos():
    logging.info("Loading SOP data from COS...")
    all_data = []
    cos_sop_keys = list_cos_files()
    
    sop_files = {
        os.path.splitext(basename(key))[0].replace("_", " ").title(): key
        for key in cos_sop_keys
        if isinstance(key, str) and key.lower().endswith(".xlsx") and not key.startswith("~$")
    }
    
    for fname, key in sop_files.items():
        try:
            response = cos.get_object(Bucket=COS_BUCKET_NAME, Key=key)
            content = response["Body"].read()
            df = preprocess_excel(content)
            df["source_file"] = fname.lower()
            all_data.append(df)
            logging.info(f"Loaded {fname}: {len(df)} rows")
        except Exception as e:
            logging.error(f"Failed to process {fname}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data, ignore_index=True)

def store_embeddings_to_milvus(df):
    if collection.num_entities > 0:
        logging.info(f"Milvus already has {collection.num_entities} embeddings. Skipping.")
        return
    
    step_col = next((c for c in ["steps involved in this task", "steps involved in the task"] if c in df.columns), None)
    if not step_col:
        return

    valid_rows = df.dropna(subset=[step_col]).copy().reset_index(drop=True)
    if len(valid_rows) == 0:
        return

    combined_texts = (valid_rows["task"].fillna("") + ". " + valid_rows[step_col].fillna("")).tolist()
    embeddings = embedder.encode(combined_texts, show_progress_bar=False)

    data = [
        embeddings,
        valid_rows[step_col].astype(str).tolist(),
        valid_rows["source_file"].astype(str).tolist() if "source_file" in valid_rows else [""] * len(valid_rows),
        valid_rows["sop_no"].astype(str).tolist(),
        valid_rows["task"].astype(str).tolist(),
        valid_rows["approvals"].astype(str).tolist(),
        valid_rows["responsibility"].astype(str).tolist(),
        valid_rows["timelines"].astype(str).tolist(),
        valid_rows["remarks"].astype(str).tolist(),
    ]

    collection.insert(data)
    collection.flush()
    logging.info(f"Inserted {len(valid_rows)} rows to Milvus")

def search_sop(query_text, top_k=3, threshold=0.35):
    if full_df is None or full_df.empty:
        return None

    query_lower = query_text.lower().strip()

    exact_match = full_df[
        (full_df["task"].astype(str).str.lower().str.strip() == query_lower)
        | (full_df["sop_no"].astype(str).str.lower().str.strip() == query_lower)
    ]

    if not exact_match.empty:
        row = exact_match.iloc[0]
        step_col = next((c for c in ["steps involved in this task", "steps involved in the task"] if c in full_df.columns), None)
        return {
            "match": row.to_dict() if step_col else None,
            "type": "exact"
        }

    partial = full_df[
        (full_df["task"].astype(str).str.lower().str.contains(query_lower, na=False))
        | (full_df["sop_full_text"].astype(str).str.contains(query_lower, na=False))
    ]

    if not partial.empty:
        row = partial.iloc[0]
        return {
            "match": row.to_dict(),
            "type": "partial"
        }

    if collection.num_entities == 0:
        return None
    
    try:
        vector = embedder.encode([query_text])[0]
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["sop_text", "source_file", "sop_no", "task", "approvals", "responsibility", "timelines", "remarks"]
        )
        
        if results and results[0] and float(results[0][0].score) >= threshold:
            return {
                "match": results[0][0].entity,
                "type": "semantic",
                "score": float(results[0][0].score)
            }
    except Exception as e:
        logging.error(f"Milvus search failed: {e}")
    
    return None

def get_summary_from_watsonx(context):
    try:
        creds = {"url": os.getenv("WATSONX_URL"), "apikey": os.getenv("WATSONX_API_KEY")}
        project_id = os.getenv("WATSONX_PROJECT_ID")
        model_id = os.getenv("WATSONX_MODEL_ID", "meta-llama/llama-2-70b-chat")
        model = Model(model_id=model_id, credentials=creds, project_id=project_id)
        gen_params = {GenParams.DECODING_METHOD: "greedy", GenParams.MAX_NEW_TOKENS: 100}
        
        prompt = f"Summarize this in 2-3 sentences:\n{context}"
        result = model.generate(prompt, gen_params)
        return result.get("results", [{}])[0].get("generated_text", "No summary available")
    except Exception as e:
        logging.error(f"WatsonX error: {e}")
        return "Summary generation failed"

@app.get("/")
async def root():
    return {"message": "SOP Chatbot API", "status": "running"}

@app.get("/health")
async def health():
    milvus_status = collection.num_entities > 0
    data_loaded = full_df is not None and not full_df.empty
    return {
        "status": "healthy",
        "milvus": milvus_status,
        "data_loaded": data_loaded,
        "embeddings": collection.num_entities if milvus_status else 0,
        "records": len(full_df) if data_loaded else 0
    }

@app.post("/query", response_model=SOPResponse)
async def query_sop(request: QueryRequest):
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = search_sop(request.query)
    
    if not result:
        raise HTTPException(status_code=404, detail="No matching SOP found")
    
    match = result["match"]
    match_type = result["type"]
    
    step_col = next((c for c in ["steps involved in this task", "steps involved in the task"] if c in match if isinstance(match, dict)), None)
    steps_text = match.get(step_col, "") if isinstance(match, dict) else ""
    
    summary = get_summary_from_watsonx(steps_text) if steps_text else "No summary available"
    
    if isinstance(match, dict):
        return SOPResponse(
            sop_no=match.get("sop_no", ""),
            task=match.get("task", ""),
            steps=steps_text,
            approvals=match.get("approvals", ""),
            responsibility=match.get("responsibility", ""),
            timelines=match.get("timelines", ""),
            remarks=match.get("remarks", ""),
            summary=summary,
            match_type=match_type
        )
    else:
        raise HTTPException(status_code=500, detail="Invalid match result")

@app.get("/status")
async def status():
    return {
        "milvus_entities": collection.num_entities,
        "sop_records": len(full_df) if full_df is not None else 0,
        "cos_files": len(list_cos_files())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)