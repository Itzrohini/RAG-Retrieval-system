import asyncio
import uvicorn
from app.config import settings

if __name__ == "__main__":
    try:
        uvicorn.run(
            "app.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=False,          # disable reload — prevents subprocess crashes
            log_level="info",
            workers=1,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass