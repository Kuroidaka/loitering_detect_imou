from zlapi import ZaloAPI
from zlapi.models import ThreadType, Message
from config import settings

zalo_client = ZaloAPI(phone=settings.ZALO_NUMBER, password=settings.ZALO_PASSWROD, imei=settings.ZALO_IMEI, cookies=settings.ZALO_COOKIES)


groups = zalo_client.fetchAllGroups()

# print(groups.gridVerMap)

# group_ids = [group.keys() for group in groups.gridVerMap]
keys = list(vars(groups.gridVerMap).keys())

[print(zalo_client.fetchGroupInfo(key)) for key in keys]