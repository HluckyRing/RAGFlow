import os
import sys
import uvicorn


def main():
    import src.server
    uvicorn.run(src.server.app, host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    main()
