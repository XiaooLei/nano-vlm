# -*- coding: utf-8 -*-
import os
import io
import base64
import uuid
import time
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List, Dict, Any, Union
import json
import traceback
import requests
import uvicorn
from pydantic import BaseModel, Field
from model import VLMModel

app = FastAPI(
    title="VLM API",
    description="OpenAI-compatible Vision Language Model API",
    version="1.0.0"
)

MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "qwen2.5-vlm-0.5b"
VISION_NAME = "google/siglip2-base-patch16-224"
LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECTOR_PATH = "checkpoints/projector_final_qwen2.5-0.5b-instruct_siglip2-base-patch16-224.pt"
LORA_PATH = ""


class MessageContentText(BaseModel):
    type: str = "text"
    text: str


class MessageContentImage(BaseModel):
    type: str = "image_url"
    image_url: dict


class Message(BaseModel):
    role: str
    content: Union[str, List[Union[MessageContentText, MessageContentImage, dict]]]


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[Message]
    max_tokens: int = 256
    temperature: float = 1.0
    stream: bool = False


def load_model():
    global MODEL
    #print(f"Initializing model, device: {DEVICE}...")
    
    MODEL = VLMModel(llm_name=LLM_NAME, vision_name=VISION_NAME)
    
    if os.path.exists(PROJECTOR_PATH):
        print(f"Loading Projector weights: {PROJECTOR_PATH}")
        projector_state = torch.load(PROJECTOR_PATH, map_location=DEVICE)
        MODEL.projector.load_state_dict(projector_state)
    else:
        print(f"Projector weights not found: {PROJECTOR_PATH}, using random initialization")
    
    MODEL.to(DEVICE)
    MODEL.eval()
    print("Model loaded successfully!")


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "local",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None
            }
        ]
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    if model_id != MODEL_NAME:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "id": model_id,
        "object": "model",
        "created": 1700000000,
        "owned_by": "local",
        "permission": [],
        "root": model_id,
        "parent": None
    }


@app.get("/")
def root():
    return {
        "message": "VLM API is running",
        "model_device": DEVICE,
        "version": "OpenAI-compatible",
        "endpoints": {
            "/v1/models": "GET - List models",
            "/v1/chat/completions": "POST - Chat completions (OpenAI-compatible)",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": DEVICE
    }


@app.post("/v1/chat/completions", response_model=Dict)
async def chat_completions(request: ChatCompletionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        messages = request.messages
        model = request.model
        max_tokens = request.max_tokens
        temperature = request.temperature
        stream = request.stream
        
        last_message = messages[-1] if messages else None
        if not last_message or last_message.role != "user":
            raise HTTPException(status_code=400, detail="Last message must be from user")
        
        content = last_message.content
        
        img = None
        prompt_text = ""
        
        if isinstance(content, list):
            for item in content:
                item_dict = item if isinstance(item, dict) else item.model_dump() if hasattr(item, 'model_dump') else {}
                if item_dict.get("type") == "image_url":
                    image_url = item_dict.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                    if url.startswith("data:"):
                        base64_data = url.split(",", 1)[1]
                        image_data = base64.b64decode(base64_data)
                        img = Image.open(io.BytesIO(image_data)).convert("RGB")
                    elif url.startswith("http"):
                        response = requests.get(url)
                        if response.status_code != 200:
                            raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {response.status_code}")
                        image_data = response.content
                        img = Image.open(io.BytesIO(image_data)).convert("RGB")
                elif item_dict.get("type") == "text":
                    prompt_text += item_dict.get("text", "")
        elif isinstance(content, str):
            prompt_text = content
        
        if not prompt_text:
            raise HTTPException(status_code=400, detail="No prompt text provided")
        
        if img is None:
            img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        
        prompt_template = f"<|im_start|>user\n<image>\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        
        response = MODEL.answer(
            image=img,
            prompt=prompt_template,
            max_new_tokens=max_tokens
        )
        
        if stream:
            return StreamingResponse(
                generate_stream(response, model),
                media_type="text/event-stream"
            )
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_text),
                "completion_tokens": len(response),
                "total_tokens": len(prompt_text) + len(response)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def generate_stream(response: str, model: str):
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())
    
    for i, char in enumerate(response):
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": char},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    final_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)