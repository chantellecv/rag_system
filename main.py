from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import requests
from PyPDF2 import PdfReader
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
import openai
import io
import numpy as np

app = FastAPI()


@app.post("/set_credentials/", tags=["Settings"])
async def set_credentials(openai_key: str = Form(...), database_name: str = Form(...), mongodb_uri: str = Form(...)):
    global user_openai_key, user_mongodb_uri

    # Store the user's OpenAI key and MongoDB URI
    user_openai_key = openai_key
    user_mongodb_uri = mongodb_uri

    # Set the OpenAI API key globally
    openai.api_key = user_openai_key

    # Connect to MongoDB using the user's URI
    client = AsyncIOMotorClient(user_mongodb_uri)
    db = client[database_name]
    
    # Set up MongoDB collections based on user-provided URI
    fs = AsyncIOMotorGridFSBucket(db)
    global files_collection, tags_collection
    files_collection = db["files"]
    tags_collection = db["tags"]

    return {"message": "Credentials set successfully!"}

def generate_embedding(text: str):
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def tag(text: str, tags: list[str]):
    tags_str = ", ".join(tags)
    
    # Make the request
    url = "https://fourth-ir-tagging-agent-8dwd.onrender.com/Tag"
    headers = {
        'accept': 'application/json'
    }
    params = {
        'Text': text,
        'Tags': tags_str,
        'Threshold': '0.8'
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json()["tags"]
    

@app.post("/upload/", tags=["Rag System"])
async def upload_file(file: UploadFile = File(...)):
   
    extracted_text = ""
    
    if file.content_type == "application/pdf":
        # Read the file into memory
        contents = await file.read()
        # Use PdfReader to extract text
        pdf_reader = PdfReader(io.BytesIO(contents))
        for page in pdf_reader.pages:
            extracted_text += page.extract_text() + "\n"
    
        print("text extracted successfully: ")      
     
    else:
        # Handle unsupported file types
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    try:
        db_tags = []
        async for tag_doc in tags_collection.find({}, {"_id": 0, "tag": 1}):
            db_tags.append(tag_doc["tag"])
    except Exception:
        raise HTTPException(status_code=404, detail="Database not found. Kindly set credentials above.") 
    
        
    identified_tags = tag(extracted_text, db_tags)
    print("tags successfully identified")    
    

    try:
        embedding = generate_embedding(extracted_text)
        print("embeddings generated")
    except Exception as e:
        error_message = str(e).split(" - ")[1].split("'error': ")[1].split(", 'type':")[0].strip("{}").split("'message': ")[1].strip("'")
        raise HTTPException(status_code=e.status_code, detail=error_message) 
    
    try:
        
        file_doc = {
            "filename": file.filename,
            "text": extracted_text,
            "tags": identified_tags,
            "embedding": embedding
        }
        
        result = await files_collection.insert_one(file_doc)
        print("embeddings inserted into db")
        return {"file_id": str(result.inserted_id), "filename": file.filename, "tags": identified_tags}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/query/", tags=["Rag System"])
async def query_database(query: str):
    try:
        db_tags = [tag_doc["tag"] async for tag_doc in tags_collection.find({}, {"_id": 0, "tag": 1})]
        query_tags = tag(query, db_tags)
    except Exception:
        raise HTTPException(status_code=404, detail="Database not found. Kindly set credentials above.") 
    
    print("query has been tagged", query_tags )
    
    # Generate OpenAI embedding for query
    try:
        query_embedding = generate_embedding(query)
        print("query has been embedded")
    except Exception as e:
        error_message = str(e).split(" - ")[1].split("'error': ")[1].split(", 'type':")[0].strip("{}").split("'message': ")[1].strip("'")
        raise HTTPException(status_code=e.status_code, detail=error_message) 
    
    # Retrieve matching embeddings based on tags
    matching_docs = [doc async for doc in files_collection.find({"tags": {"$in": query_tags}}, {"_id": 0, "embedding": 1, "text": 1})]

    # print("matching docs: ", matching_docs)
    
    if not matching_docs:
        return {"response": "No relevant documents found."}

    # Compute similarity (dot product as cosine similarity is normalized)
    similarities = [
        (doc["text"], np.dot(query_embedding, doc["embedding"]))
        for doc in matching_docs
    ]

    # Sort by highest similarity
    most_relevant_text = sorted(similarities, key=lambda x: x[1], reverse=True)[0][0]

    # Use ChatGPT to generate response
    chat_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an assistant that answers questions based on retrieved documents."},
                  {"role": "user", "content": f"Answer based on this document:\n{most_relevant_text}\n\nQuery: {query}"}]
    )

    return {"response": chat_response.choices[0].message.content}

@app.post("/add_tag/", tags=["Tags"])
async def add_tag(tag: str):
    try:
        # Find the file in MongoDB
        if await tags_collection.find_one({"tag": tag}):
            raise HTTPException(status_code=400, detail="Tag already exists")
        # Insert the new tag into the tags collection
        await tags_collection.insert_one({"tag": tag})
    except Exception:
        raise HTTPException(status_code=404, detail=str("Database not found. Kindly set credentials above.")) 
    return {"message": f"Tag '{tag}' added successfully"}

# Endpoint to view all tags for a specific file
@app.get("/get_tags/", tags=["Tags"])
async def view_tags():
    # Retrieve all tags from the tags collection
    tags = []
    try:
        async for tag_doc in tags_collection.find({}, {"_id": 0, "tag": 1}):
            tags.append(tag_doc["tag"])
    except Exception:
        raise HTTPException(status_code=404, detail=str("Database not found. Kindly set credentials above.")) 
    
    return {"tags": tags}

# Endpoint to delete a tag from a file
@app.delete("/delete_tag/", tags=["Tags"])
async def delete_tag(tag_name: str):
    # Find the file in MongoDB
    try: 
        if not await tags_collection.find_one({"tag": tag_name}):
            raise HTTPException(status_code=404, detail="Tag not found")
        # Delete the tag from the tags collection
        await tags_collection.delete_one({"tag": tag_name})
        # Remove the tag from all files that have it
        await files_collection.update_many({"tags": tag_name}, {"$pull": {"tags": tag_name}})
    except HTTPException:
        # Re-raise the HTTPException if it was already raised (e.g., "Tag not found")
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Database not found. Kindly set credentials above.") 
    return {"message": f"Tag '{tag_name}' deleted successfully"}