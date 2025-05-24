import asyncio
import aiohttp
from imouapi.api import ImouAPIClient
from imouapi.device import ImouDiscoverService, ImouDevice
import cv2
from imouapi.exceptions import APIError

async def get_rtsp_url(app_id: str, app_secret: str) -> str:
    # 1. Create an HTTP session
    async with aiohttp.ClientSession() as session:
        # 2. Instantiate & authenticate the client
        client = ImouAPIClient(app_id, app_secret, session)  
        await client.async_connect()  
        
        # 3. Discover your cameras
        disco = ImouDiscoverService(client)
        devices = await disco.async_discover_devices()
        if not devices:
            raise RuntimeError("No Imou devices found on this account")
        
        # 4. Pick the camera you want (e.g. the first one)
        device_info = devices['Ranger 2C 3MP-DDDC']
        device = ImouDevice(client, device_info._device_id)
        await device.async_initialize()
        
        # 5. Bind a live session and retrieve its token
        try:
            bind = await client.async_api_bindDeviceLive(device_info._device_id, "SD")
            # play_token = bind.get("playToken") or bind.get("liveToken")
        except APIError as e:
            pass
        stream_info = await client.async_api_getLiveStreamInfo(device_info._device_id)

        hls_url = next(
            (s["hls"] for s in stream_info["streams"] 
            if s["streamId"] == 0 and s["hls"].startswith("http://")),
            next(
            (s["hls"] for s in stream_info["streams"]  if s["streamId"] == 0),
            None
            )
        )

        if hls_url is None:
            raise RuntimeError("No streamId=0 entry found in response.")

        # print("Playing from:", hls_url)

        # 3. open it in OpenCV (FFmpeg backend must be built with HLS support)
        cap = cv2.VideoCapture(hls_url)

        if not cap.isOpened():
            raise RuntimeError("Failed to open HLS stream")

        # drop the internal buffer down to 1 frame (if your OpenCV build honors it)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except:
            pass

        while True:
            # optionally grab and discard old frames:
            for _ in range(3):
                cap.grab()

            ret, frame = cap.retrieve()
            if not ret:
                continue

            cv2.imshow("Live RTSP", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        # return rtsp_url

if __name__ == "__main__":
    APP_ID = "lc438c434135624f1c"
    APP_SECRET = "6a7ccf4e95a640c6ad2f3f5126128e"
    url = asyncio.run(get_rtsp_url(APP_ID, APP_SECRET))
    print("RTSP URL:", url)
# C34A2AMPBV0DDDCf51643ce-f7a2-47ae-8043-04e8701eb985