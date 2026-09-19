"""LLM client for RAG generation."""

import requests
import time


class LLMClient:
    """OpenAI-compatible LLM client."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initialize the LLM client.
        
        Args:
            api_key: API key.
            base_url: Base URL for the API.
            model: Model name.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
    
    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send.
            temperature: Sampling temperature.
            
        Returns:
            Generated response text.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
                
            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except requests.exceptions.ConnectionError:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts")


def demo_answer(query: str, docs: list[str]) -> str:
    """Generate a demo answer without an LLM (for testing).
    
    Args:
        query: User question.
        docs: Retrieved documents.
        
    Returns:
        Demo answer string.
    """
    if not docs:
        return "No relevant documents found."
    
    return f"[DEMO MODE] Based on the retrieved context: {docs[0][:200]}..."
