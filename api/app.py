from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

from tempfile import NamedTemporaryFile

from api.inference import predict

from prometheus_client import generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi import Response
from fastapi.responses import FileResponse
from api.inference import predict_with_image
from api.inference import predict_video
from fastapi import HTTPException

app = FastAPI(
    title="PerceptionOps API"
)

@app.get("/")
def health():

    return {
        "status": "healthy",
        "model": "YOLOv8n"
    }


@app.post("/predict")
async def run_prediction(
    file: UploadFile = File(...)
):

    with NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp:

        temp.write(
            await file.read()
        )

        temp_path = temp.name

    detections = predict(
        temp_path
    )
    result = predict(temp_path)
    return result

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/predict-image")
async def predict_image_visual(
    file: UploadFile = File(...)
):

    with NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp:

        temp.write(await file.read())
        temp_path = temp.name

    _, annotated_path = predict_with_image(
        temp_path
    )

    return FileResponse(
        annotated_path,
        media_type="image/jpeg"
    )


@app.post("/predict-video")
async def predict_video_endpoint(
    file: UploadFile = File(...)
):
    try:

        with NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp:

            temp.write(
                await file.read()
            )

            temp_path = temp.name

        output_video = predict_video(
            temp_path
        )

        return FileResponse(
            output_video,
            media_type="video/mp4",
            filename="prediction.mp4"
        )

    except Exception as e:

        print("VIDEO ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )