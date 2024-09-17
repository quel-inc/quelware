from ipaddress import IPv4Address
from typing import Any

from pydantic import BaseModel, Field


class _SyncInterface(BaseModel):
    ipaddr: IPv4Address = Field()
    port: int = Field(default=16385, ge=0, le=65535)

    def __hash__(self) -> int:
        return (int(self.ipaddr), self.port).__hash__()

    @classmethod
    def create(cls, ipaddrs: dict[str, str], settings: dict[str, Any]) -> "_SyncInterface":
        return cls(ipaddr=IPv4Address(ipaddrs[settings["nic"]]), port=int(settings["port"]))
