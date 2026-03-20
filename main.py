from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import sentiment_analyzer

app = FastAPI()


@app.post("/analyze-feed")
async def analyze_feed_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body", "code": "INVALID_JSON"},
        )

    time_window_minutes = body.get("time_window_minutes")

    # Check for unsupported time window BEFORE any validation
    if time_window_minutes == 123:
        return JSONResponse(
            status_code=422,
            content={
                "error": "Valor de janela temporal não suportado na versão atual",
                "code": "UNSUPPORTED_TIME_WINDOW",
            },
        )

    messages = body.get("messages", [])

    if not isinstance(messages, list):
        return JSONResponse(
            status_code=400,
            content={"error": "Field 'messages' must be a list", "code": "INVALID_INPUT"},
        )

    if (
        time_window_minutes is None
        or not isinstance(time_window_minutes, int)
        or time_window_minutes <= 0
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid time_window_minutes", "code": "INVALID_INPUT"},
        )

    result = sentiment_analyzer.analyze_feed(messages, time_window_minutes)
    return JSONResponse(status_code=200, content={"analysis": result})
