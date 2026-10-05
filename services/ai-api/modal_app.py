"""Optional Modal deployment entry point; deploy from the repository root."""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements("services/ai-api/requirements.txt").add_local_dir("services/ai-api/app", "/root/app")
modal_app = modal.App("truthlens-ai-api", image=image)


@modal_app.function(timeout=300, min_containers=0)
@modal.asgi_app()
def fastapi_app():
    import sys
    sys.path.insert(0, "/root")
    from app.main import app
    return app
