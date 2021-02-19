import http.server
import json
import numpy as np
from functools import partial

import nn

HOST_NAME = "localhost"
PORT_NUMBER = 8888
INPUT_NODE_COUNT = 784
HIDDEN_NODE_COUNT = 40
OUTPUT_NODE_COUNT = 10

neural_network = nn.OCRNeuralNetwork(
    INPUT_NODE_COUNT,
    HIDDEN_NODE_COUNT,
    OUTPUT_NODE_COUNT,
    "neural_network.json"
)


class JSONHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        response_code = 200
        response = ""
        content_len = int(self.headers.get("Content-Length", 0))
        content = self.rfile.read(content_len)
        payload = json.loads(content)
        if self.path == "/train":
            neural_network.back_propagate(np.array(payload["image"]), int(payload["label"]))
            response_code = 200
        elif self.path == "/predict":
            response_code = 200
            predictions = neural_network.predict(np.array(payload["image"]))
            print(predictions)
            prediction = max(predictions)
            response = {"type": "predict", "result": predictions.tolist().index(prediction)}
        elif self.path == "/initialize":
            response_code = 200
            neural_network.initialize()
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
