from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Depends
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
import httpx
import logging
from typing import List, Optional

from app.config import get_settings
from app.database import supabase
from app.ai_engine import process_image, load_models
from app.security import encrypt_token

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniVision")

settings = get_settings()
app = FastAPI(title="OmniVision API")

# Startup event to load models
@app.on_event("startup")
async def startup_event():
    # load_models() # No longer needed for API-based engine
    pass

# Models for Facebook Webhook
class WebhookEntry(BaseModel):
    id: str
    time: int
    messaging: List[dict]

class WebhookEvent(BaseModel):
    object: str
    entry: List[WebhookEntry]

# --- Helper Functions ---

async def send_facebook_message(recipient_id: str, message_text: str, page_access_token: str):
    """Sends a text message to a user via Facebook Graph API."""
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={page_access_token}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to send message: {e.response.text}")

async def send_facebook_image(recipient_id: str, image_url: str, page_access_token: str):
    """Sends an image to a user via Facebook Graph API."""
    url = f"https://graph.facebook.com/v18.0/me/messages?access_token={page_access_token}"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {
            "attachment": {
                "type": "image",
                "payload": {
                    "url": image_url, 
                    "is_reusable": True
                }
            }
        }
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to send image: {e.response.text}")

async def get_shop_config(page_id: str):
    """Fetches shop configuration from Supabase."""
    try:
        response = supabase.table("shops").select("*").eq("page_id", page_id).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Database error: {e}")
        return None

async def process_incoming_message(event: dict):
    """Background task to process the incoming message."""
    sender_id = event.get("sender", {}).get("id")
    recipient_id = event.get("recipient", {}).get("id") # This is the Page ID
    message = event.get("message", {})
    
    if not sender_id or not recipient_id:
        return

    # Fetch shop config to get access token
    shop_config = await get_shop_config(recipient_id)
    if not shop_config:
        logger.warning(f"Shop not found for Page ID: {recipient_id}")
        return
    
    # Check Credits
    owner_id = shop_config.get("owner_id")
    credits = 0
    if owner_id:
        try:
            user_resp = supabase.table("users").select("credits").eq("facebook_user_id", owner_id).execute()
            if user_resp.data:
                credits = user_resp.data[0]["credits"]
                if credits <= 0:
                    logger.warning(f"Insufficient credits for User {owner_id}")
                    # Optional: Send message to user saying "Out of credits"
                    return
            else:
                logger.warning(f"User {owner_id} not found in users table.")
                # If user not found, maybe allow or block? Let's block to be safe.
                return
        except Exception as e:
            logger.error(f"Error checking credits: {e}")
            return

    # Decrypt token
    from app.security import decrypt_token
    try:
        page_access_token = decrypt_token(shop_config["encrypted_access_token"])
    except Exception as e:
        logger.error(f"Token decryption failed: {e}")
        return

    if "attachments" in message:
        for attachment in message["attachments"]:
            if attachment["type"] == "image":
                image_url = attachment["payload"]["url"]
                
                # Deduct Credit
                if owner_id:
                    try:
                        supabase.table("users").update({"credits": credits - 1}).eq("facebook_user_id", owner_id).execute()
                        logger.info(f"Deducted 1 credit from user {owner_id}")
                    except Exception as e:
                        logger.error(f"Failed to deduct credit: {e}")
                
                await handle_image_search(sender_id, image_url, recipient_id, shop_config, page_access_token)
                return # Only process first image for now

    # If text message, maybe send a greeting or help text
    # await send_facebook_message(sender_id, "Send me an image to find products!", page_access_token)

async def handle_image_search(user_id: str, image_url: str, page_id: str, shop_config: dict, token: str):
    """Downloads image, runs AI, finds matches, and replies."""
    try:
        # Download image
        async with httpx.AsyncClient() as client:
            resp = await client.get(image_url)
            if resp.status_code != 200:
                await send_facebook_message(user_id, "Failed to download image.", token)
                return
            image_bytes = resp.content

        # Run AI Pipeline
        # This is synchronous CPU intensive work, ideally run in a threadpool or separate worker
        # FastAPI BackgroundTasks run in the same event loop if async, or threadpool if sync def.
        # process_image is sync, so it blocks. We should run it in a thread.
        import asyncio
        from functools import partial
        
        loop = asyncio.get_running_loop()
        ai_result = await loop.run_in_executor(None, process_image, image_bytes)
        
        siglip_emb = ai_result["siglip_embedding"]
        dino_emb = ai_result["dino_embedding"]

        # Search in Database
        # match_products(query_siglip, query_dino, match_threshold, filter_page_id)
        params = {
            "query_siglip": siglip_emb,
            "query_dino": dino_emb,
            "match_threshold": 0.70,
            "filter_page_id": int(page_id)
        }
        
        # Supabase rpc call
        logger.info(f"Calling match_products with params: {params}")
        rpc_response = supabase.rpc("match_products", params).execute()
        matches = rpc_response.data
        logger.info(f"Matches found: {matches}")

        if matches:
            top_match = matches[0]
            # --- Visual Verification Step ---
            # Download candidate image
            candidate_url = top_match["image_url"]
            verification_score = 0
            try:
                async with httpx.AsyncClient() as client:
                    c_resp = await client.get(candidate_url)
                    if c_resp.status_code == 200:
                        candidate_bytes = c_resp.content
                        
                        # Run Verification
                        from app.ai_engine import verify_visual_match
                        verification_score = await loop.run_in_executor(None, verify_visual_match, image_bytes, candidate_bytes)
            except Exception as e:
                logger.error(f"Verification download failed: {e}")

            logger.info(f"Final Verification Score: {verification_score}")

            # Filter based on score
            if verification_score < 65:
                logger.info("Match rejected due to low visual score.")
                msg_not_found = shop_config.get("msg_not_found", "Sorry, we could not find a match for that item.")
                await send_facebook_message(user_id, msg_not_found, token)
                return

            # Handle "Soft Match" (65-85%)
            if 65 <= verification_score < 85:
                await send_facebook_message(user_id, "We couldn't find an exact match, but this is the closest we found:", token)

            # Format message
            msg_template = shop_config.get("msg_found", "Found {name} for {price}. Confidence: {confidence}%")
            
            # Use verification score as confidence
            confidence_pct = verification_score
            
            # Safe format to ignore missing keys in template
            try:
                reply_text = msg_template.format(
                    name=top_match.get("name", "Unknown"), 
                    price=top_match.get("price", "N/A"),
                    confidence=confidence_pct
                )
            except Exception as e:
                logger.error(f"Template format error: {e}")
                reply_text = f"Found {top_match.get('name')}."
            
            await send_facebook_message(user_id, reply_text, token)
            
            # Send Product Image if enabled
            if shop_config.get("send_image", False):
                await send_facebook_image(user_id, top_match["image_url"], token)
        else:
            msg_not_found = shop_config.get("msg_not_found", "No match found.")
            await send_facebook_message(user_id, msg_not_found, token)

    except Exception as e:
        logger.error(f"Error in search pipeline: {repr(e)}") # Use repr to see full error
        import traceback
        logger.error(traceback.format_exc())
        await send_facebook_message(user_id, "An error occurred while processing your image.", token)

# --- Routes ---

@app.get("/")
def health_check():
    return {"status": "ok", "service": "OmniVision"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Facebook Webhook verification."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == settings.FACEBOOK_VERIFY_TOKEN:
            logger.info("Webhook verified successfully.")
            return PlainTextResponse(content=challenge, status_code=200)
        else:
            logger.warning(f"Webhook verification failed. Token: {token}, Expected: {settings.FACEBOOK_VERIFY_TOKEN}")
            raise HTTPException(status_code=403, detail="Verification failed")
    return HTTPException(status_code=400, detail="Missing parameters")

@app.post("/webhook")
async def webhook_handler(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming messages from Facebook."""
    try:
        body = await request.json()
        if body.get("object") == "page":
            for entry in body.get("entry", []):
                for messaging_event in entry.get("messaging", []):
                    # Add processing to background task to respond quickly to FB
                    background_tasks.add_task(process_incoming_message, messaging_event)
            return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)
        else:
            raise HTTPException(status_code=404, detail="Not a page event")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        # Return 200 to prevent FB from retrying indefinitely on bad logic
        return PlainTextResponse(content="EVENT_RECEIVED", status_code=200)

# --- OAuth & Shop Management (Simplified) ---

class ShopOnboard(BaseModel):
    page_id: int
    access_token: str
    name: str
    owner_id: Optional[str] = None # Added owner_id

@app.post("/onboard")
async def onboard_shop(shop: ShopOnboard):
    """Endpoint to register a new shop/page."""
    # Encrypt token
    encrypted_token = encrypt_token(shop.access_token)
    
    data = {
        "page_id": shop.page_id,
        "encrypted_access_token": encrypted_token,
        # shop_api_key is auto-generated
    }
    
    if shop.owner_id:
        data["owner_id"] = shop.owner_id
    
    try:
        # 1. Save to DB
        response = supabase.table("shops").upsert(data).execute()
        
        # 2. Subscribe App to Page Webhooks
        # This ensures we receive messages for this page
        subscribe_url = f"https://graph.facebook.com/v18.0/{shop.page_id}/subscribed_apps"
        subscribe_params = {
            "access_token": shop.access_token,
            "subscribed_fields": "messages,messaging_postbacks"
        }
        async with httpx.AsyncClient() as client:
            sub_resp = await client.post(subscribe_url, params=subscribe_params)
            if sub_resp.status_code != 200:
                logger.error(f"Failed to subscribe app to page {shop.page_id}: {sub_resp.text}")
                # We don't raise error here to allow onboarding to complete, but we log it.
                # In production, we might want to return a warning.

        return {"status": "success", "data": response.data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from fastapi import UploadFile, File, Form

@app.post("/products")
async def add_product(
    shop_id: int = Form(...),
    name: str = Form(...),
    price: float = Form(...),
    file: UploadFile = File(...)
):
    """Add a product with image upload."""
    
    # 1. Read file content
    image_bytes = await file.read()
    
    # 2. Upload to Supabase Storage
    import uuid
    filename = f"{shop_id}/{uuid.uuid4()}.jpg"
    try:
        # Upload
        supabase.storage.from_("products").upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        # Get Public URL
        image_url = supabase.storage.from_("products").get_public_url(filename)
    except Exception as e:
        logger.error(f"Storage upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload image")

    # 3. Generate Embeddings
    # Run sync function in executor
    import asyncio
    loop = asyncio.get_running_loop()
    ai_result = await loop.run_in_executor(None, process_image, image_bytes)
    
    # 4. Save to DB
    data = {
        "shop_id": shop_id,
        "name": name,
        "price": price,
        "image_url": image_url,
        "siglip_embedding": ai_result["siglip_embedding"],
        "dino_embedding": ai_result["dino_embedding"]
    }
    
    try:
        response = supabase.table("products").insert(data).execute()
        return {"status": "created", "product": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/shops/{page_id}/products")
async def get_shop_products(page_id: int):
    """Get all products for a specific shop."""
    try:
        response = supabase.table("products").select("*").eq("shop_id", page_id).order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- User & Shop Management API (For PHP Frontend) ---

class UserCreate(BaseModel):
    facebook_user_id: str
    name: str
    email: Optional[str] = None

@app.post("/users")
async def create_or_update_user(user: UserCreate):
    """Upsert user from Facebook Login."""
    data = {
        "facebook_user_id": user.facebook_user_id,
        "name": user.name,
        "email": user.email
    }
    try:
        # Upsert: Insert or Update on conflict
        response = supabase.table("users").upsert(data).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    """Get user details (credits)."""
    try:
        response = supabase.table("users").select("*").eq("facebook_user_id", user_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="User not found")
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/shops")
async def get_user_shops(user_id: str):
    """Get shops owned by user."""
    try:
        response = supabase.table("shops").select("page_id").eq("owner_id", user_id).execute()
        # We only return IDs here. Frontend fetches names from FB Graph.
        # Or we could store names in DB. For now, just IDs.
        return [row['page_id'] for row in response.data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/shops/{page_id}")
async def get_shop_details(page_id: int):
    """Get shop config."""
    try:
        response = supabase.table("shops").select("*").eq("page_id", page_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Shop not found")
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ShopUpdate(BaseModel):
    msg_found: str
    msg_not_found: str
    send_image: bool
    # include_confidence: bool # Deprecated
    service_image: bool
    service_chat: bool
    chat_context: Optional[str] = ""

@app.put("/shops/{page_id}")
async def update_shop_details(page_id: int, shop: ShopUpdate):
    """Update shop messages and settings."""
    try:
        data = {
            "msg_found": shop.msg_found,
            "msg_not_found": shop.msg_not_found,
            "send_image": shop.send_image,
            # "include_confidence": shop.include_confidence, # Deprecated
            "service_image": shop.service_image,
            "service_chat": shop.service_chat,
            "chat_context": shop.chat_context
        }
        response = supabase.table("shops").update(data).eq("page_id", page_id).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/shops/{page_id}")
async def delete_shop(page_id: int):
    """Disconnects a shop: Unsubscribes from Webhook and deletes from DB."""
    try:
        # 1. Get Access Token to Unsubscribe
        shop_config = await get_shop_config(str(page_id))
        if shop_config:
            from app.security import decrypt_token
            try:
                page_access_token = decrypt_token(shop_config["encrypted_access_token"])
                
                # 2. Unsubscribe from Facebook Webhooks
                unsubscribe_url = f"https://graph.facebook.com/v18.0/{page_id}/subscribed_apps"
                async with httpx.AsyncClient() as client:
                    await client.delete(unsubscribe_url, params={"access_token": page_access_token})
                    logger.info(f"Unsubscribed app from page {page_id}")
            except Exception as e:
                logger.error(f"Failed to unsubscribe page {page_id}: {e}")
                # Continue to delete from DB even if unsubscribe fails

        # 3. Delete from Database
        response = supabase.table("shops").delete().eq("page_id", page_id).execute()
        return {"status": "success", "message": f"Shop {page_id} disconnected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
