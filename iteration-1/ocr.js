let ocrDemo = (function() {
    const GRID_STROKE_COLOR = "blue";
    const BACKGROUND_COLOR = "white";
    const STROKE_COLOR = "black";
    const PIXEL_WIDTH = 50
    const GRID_WIDTH = 3;
    const CANVAS_WIDTH = PIXEL_WIDTH * GRID_WIDTH;
    let pixelData = [[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]];
    const GRID_STROKE_COLOR = "blue";
    const BACKGROUND_COLOR = "black";
    const STROKE_COLOR = "white";
    const CANVAS_WIDTH = 200;
    const GRID_WIDTH = 20;
    // We'll draw on a 20 x 20 grid.
    // CANVAS_WIDTH divided by 20 GRID_WIDTH equals 10 pixels per grid block.
    const SCALED_PIXEL_WIDTH = CANVAS_WIDTH / GRID_WIDTH;
    const HOST = "localhost";
    const PORT = "8888";
    let pixelData = [];
    function drawGrid(ctx) {
        ctx.fillStyle = BACKGROUND_COLOR;
        ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_WIDTH);
        ctx.strokeStyle = GRID_STROKE_COLOR;
        for (
            let x = PIXEL_WIDTH, y = PIXEL_WIDTH;
            x < CANVAS_WIDTH;
            x += PIXEL_WIDTH, y += PIXEL_WIDTH
        ) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, CANVAS_WIDTH);
            ctx.moveTo(0, y);
            ctx.lineTo(CANVAS_WIDTH, y);
            ctx.stroke();
        }
    };
    function fillSquare(mouseEvent, context, canvasElement) {
        let boundingRect = canvasElement.getBoundingClientRect();
        let x = mouseEvent.clientX - boundingRect.x;
        let y = mouseEvent.clientY - boundingRect.y;
        let xPixel = Math.floor(x / SCALED_PIXEL_WIDTH);
        let yPixel = Math.floor(y / SCALED_PIXEL_WIDTH);
        context.fillStyle = STROKE_COLOR;
        context.fillRect(
            xPixel * SCALED_PIXEL_WIDTH,
            yPixel * SCALED_PIXEL_WIDTH,
            SCALED_PIXEL_WIDTH,
            SCALED_PIXEL_WIDTH
        );
        // Along with coloring the square in the grid,
        // we also want to store the information that we colored
        // a particular pixel in our pixelData that we'll later
        // send to a server to either train our model or make
        // a prediction.
        let pixelIndex = yPixel * GRID_WIDTH + xPixel;
        pixelData[pixelIndex] = 1;
    }
    function onMouseDown(event, context, canvas) {
        canvas.isDrawing = true;
        fillSquare(event, context, canvas);
    }
    
    function onMouseUp(canvas) {
        canvas.isDrawing = false;
    }
    
    function onMouseMove(event, context, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        fillSquare(event, context, canvas);
    }
    function onLoadFunction() {
        resetCanvas();
        let canvasEl = document.getElementById("canvas");
        let context = canvasEl.getContext("2d");
        canvasEl.onmousemove = function(event) { onMouseMove(event, context, canvasEl); };
        canvasEl.onmousedown = function(event) { onMouseDown(event, context, canvasEl); };
        canvasEl.onmouseup = function(_) { onMouseUp(canvasEl); };
    }
    function resetCanvas() {
        let canvasEl = document.getElementById("canvas");
        let context = canvasEl.getContext("2d");
        let gridSize = Math.pow((CANVAS_WIDTH / SCALED_PIXEL_WIDTH), 2);
        pixelData = [];
        while (gridSize--) pixelData.push(0);
        console.log(pixelData);
        drawGrid(context);
    }
    function sendData(json) {
        let xhr = new XMLHttpRequest();
        xhr.open("POST", `http://${HOST}:${PORT}`);
        xhr.onload = function() {
            if (xhr.status == 200) {
                let responseJSON = JSON.parse(xhr.responseText);
                if (responseJSON && responseJSON.type == "predict") {
                    alert(`The neural network predicts you wrote a '${responseJSON.result}'`)
                }
            } else {
                alert(`Server returned status ${xhr.status}.`);
            }
        };
        xhr.onerror = function() {
            alert(`Error occured while connecting to server: ${xhr.target.statusText}`);
        };
        let msg = JSON.stringify(json);
        xhr.setRequestHeader("Content-Length", msg.length);
        xhr.setRequestHeader("Connection", "close");
        xhr.send(msg);
    }
    function train() {
        let digitValue = document.getElementById("digit").value;
        if (!digitValue.match(/^\d/)) {
            alert("Please type and draw a digit in order to train the network.");
            return;
        }
        let json = {
            image: pixelData,
            label: digitValue
        };
        sendData(json);
    }
    function predict() {
        if (pixelData.indexOf(1) < 0) {
            alert("Please draw a digit in order to use prediction.");
        } else {
            let json = {
                image: pixelData,
                predict: true
            };
            sendData(json);
        }
    }
    return {
        onLoadFunction,
        train,
        predict,
        resetCanvas
    };
})();
