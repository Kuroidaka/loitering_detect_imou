import asyncio
import aiohttp
from imouapi.api import ImouAPIClient
from imouapi.device import ImouDiscoverService
from imouapi.exceptions import APIError

class ImouStreamClient:
    def __init__(self, app_id: str, app_secret: str, camera_alias: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.camera_alias = camera_alias

        self._session = None
        self._client = None
        self._device_info = None

    async def _ensure_connected(self):
        if self._client is None:
            self._session = aiohttp.ClientSession()
            self._client = ImouAPIClient(self.app_id, self.app_secret, self._session)
        await self._client.async_connect()

    async def _ensure_device(self):
        if self._device_info is None:
            disco = ImouDiscoverService(self._client)
            devices = await disco.async_discover_devices()
            for d in devices:
                if d == self.camera_alias:
                    self._device_info = devices[d]
                    return
            raise RuntimeError(f"Camera '{self.camera_alias}' not found")

    async def get_hls_url(self) -> str:
        # 1) make sure we’re connected and have device info
        await self._ensure_connected()
        await self._ensure_device()

        # 2) try to bind (ignore “already exists”)
        try:
            await self._client.async_api_bindDeviceLive(self._device_info._device_id, "HD")
        except APIError as e:
            # LV1001 = live session already exists; we can ignore
            if "LV1001" not in str(e):
                raise

        # 3) fetch the live-stream info (by device ID)
        stream_info = await self._client.async_api_getLiveStreamInfo(
            self._device_info._device_id
        )

        # 4) pick the HLS URL for streamId=0, preferring HTTPS
        hls_url = next(
            (s["hls"] for s in stream_info["streams"]
             if s["streamId"] == 0 and s["hls"].startswith("https://")),
            next(
                (s["hls"] for s in stream_info["streams"]
                 if s["streamId"] == 0),
                None
            )
        )

        if not hls_url:
            raise RuntimeError("No streamId=0 entry found in response.")

        return hls_url

    async def close(self):
        if self._session:
            await self._session.close()
