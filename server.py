import http.server
import json
import numpy as np
from functools import partial

HOST_NAME = "localhost"
PORT_NUMBER = 8888
HIDDEN_NODE_COUNT = 15


class JSONHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        response_code = 200
        response = ""
        content_len = int(self.headers.get("Content-Length", 0))
        content = self.rfile.read(content_len)
        payload = json.loads(content)
        if self.path == "/train":
            response_code = 200
        elif self.path == "/predict":
            response_code = 200
            response = {"type": "predict", "result": 9}
        else:
            response_code = 404
        self.send_response(response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response).encode("utf-8"))

def main():
    print(f"Serving HTTP on {HOST_NAME} port {PORT_NUMBER}")
    httpd = http.server.HTTPServer((HOST_NAME, PORT_NUMBER), partial(JSONHandler, directory="."))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    else:
        print("Unexpected server exception occurred.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    main()
