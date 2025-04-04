import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("index_service.app:app", host="0.0.0.0", port=8003, reload=True)
