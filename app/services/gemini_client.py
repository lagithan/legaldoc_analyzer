# backend/app/services/gemini_client.py

import httpx
import logging
from ..config.settings import GEMINI_API_URL, GEMINI_HEADERS

logger = logging.getLogger(__name__)

class EnhancedGeminiClient:
    def __init__(self):
        self.api_url = GEMINI_API_URL
        self.headers = GEMINI_HEADERS

    async def generate_content(self, prompt: str, max_tokens: int = 1200) -> str:
        """Enhanced Gemini API call with better error handling"""

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.1,  # Lower for more consistent legal analysis
                "topP": 0.8,
                "topK": 10
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload
                )

                if response.status_code != 200:
                    logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                    raise Exception(f"Gemini API error: {response.status_code}")

                result = response.json()

                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return content.strip()
                else:
                    raise Exception("No content in Gemini response")

            except httpx.TimeoutException:
                raise Exception("Gemini API timeout")
            except Exception as e:
                logger.error(f"Gemini API call failed: {str(e)}")
                raise Exception(f"Gemini API error: {str(e)}")