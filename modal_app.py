import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("app", remote_path="/root/app")
)

app = modal.App("pare-hurip-api", image=image)

@app.function(
    image=image,
    memory=2048,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def fastapi_app():
    from app.main import app as web_app

    return web_app
