@echo off
echo Starting Helmet Guard AI System...

start cmd /k "cd backend && npm start"
start cmd /k "cd frontend && npm run dev"
start cmd /k "cd ai && \"C:\Users\Banu Prakash\AppData\Local\Python\bin\python.exe\" detect.py"

echo System components are launching!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo AI: Running Computer Vision...
