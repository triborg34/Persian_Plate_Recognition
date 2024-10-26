import asyncio
import websockets
import base64

async def handle_connection(websocket, path):
    print("Client connected")
    try:
        async for message in websocket:
            # Decode the received base64-encoded image data
            image_data = base64.b64decode(message)
            print("Received an image frame")  # For debugging
            # Here, you could process or save the image data as needed
    except websockets.ConnectionClosed:
        print("Client disconnected")

async def start_server():
    # Start the WebSocket server on localhost:8765
    server = await websockets.serve(handle_connection, "localhost", 8765)
    print("WebSocket server is running on ws://localhost:8765")
    await server.wait_closed()

# Run the WebSocket server
asyncio.run(start_server())
