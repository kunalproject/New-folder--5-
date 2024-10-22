
const express = require('express');
const { spawn } = require('child_process');
const app = express();
const port = 5000;
const cores = require('cors');
app.use(cores());

// Define the `/predict` route

app.get('/predict', (req, res) => {
    console.log("request made ")
    const inputData = req.query.data;

    // Check if input data is provided
    if (!inputData) {
        return res.status(400).json({ error: 'No input data provided' });
    }

    // Pass the input data to the Python script as arguments
    const inputArgs = inputData.split(',');

    // Ensure 8 inputs are provided
    if (inputArgs.length !== 8) {
        return res.status(400).json({ error: 'Input data must contain 8 values.' });
    }

    // Spawn a Python process to run the prediction script
    const pythonProcess = spawn('python', ['predict_diabetes.py', ...inputArgs]);

    let scriptOutput = '';

    // Capture the script output (stdout)
    pythonProcess.stdout.on('data', (data) => {
        scriptOutput += data.toString();
    });


    // Handle the script completion
    pythonProcess.on('close', (code) => {
        if (code === 0) {
            if (!res.headersSent) {
                res.json({ result: scriptOutput.trim() });
            }
        } else {
            console.error(`Python script exited with code ${code}`);
            if (!res.headersSent) {
                res.status(500).json({ error: 'Failed to complete prediction.' });
            }
        }
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
