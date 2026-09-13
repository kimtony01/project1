import asyncio
import websockets
import json
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("AISSTREAM_API_KEY")
print("API KEY:", API_KEY)

BOUNDING_BOX = [[1.0, 100.0], [6.0, 104.0]]

async def collect_and_save():
    ships_data = {}

    async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
        subscribe_message = {
            "APIKey": API_KEY,
            "BoundingBoxes": [BOUNDING_BOX],
            "FilterMessageTypes": ["PositionReport", "ShipStaticData"]
        }
        print("전송하는 구독 메시지:", json.dumps(subscribe_message))
        await websocket.send(json.dumps(subscribe_message))

        try:
            async with asyncio.timeout(60):
                async for message_json in websocket:
                    message = json.loads(message_json)

                    if "error" in message:
                        print("에러 발생:", message)
                        continue

                    if "MetaData" not in message:
                        print("MetaData 없는 메시지:", message)
                        continue

                    msg_type = message["MessageType"]
                    mmsi = message["MetaData"]["MMSI"]

                    if mmsi not in ships_data:
                        ships_data[mmsi] = {}

                    if msg_type == "ShipStaticData":
                        data = message["Message"]["ShipStaticData"]
                        ships_data[mmsi]["name"] = data.get("Name", "").strip()
                        ships_data[mmsi]["destination"] = data.get("Destination", "").strip()

                    if msg_type == "PositionReport":
                        data = message["Message"]["PositionReport"]
                        ships_data[mmsi]["lat"] = data.get("Latitude")
                        ships_data[mmsi]["lon"] = data.get("Longitude")
                        ships_data[mmsi]["speed"] = data.get("Sog")

        except asyncio.TimeoutError:
            pass

    result = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ships": ships_data
    }

    with open("ship_data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {len(ships_data)}척, {result['updated_at']}")

if __name__ == "__main__":
    asyncio.run(collect_and_save())