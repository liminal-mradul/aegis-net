"""
Network protocol message definitions
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
from enum import Enum
import json

class MessageType(Enum):
    # Network management
    PEER_DISCOVERY = "peer_discovery"
    PEER_RESPONSE = "peer_response"
    HEARTBEAT = "heartbeat"
    
    # Consensus
    PROPOSE = "propose"
    VOTE = "vote"
    COMMIT = "commit"
    
    # Secure aggregation
    KEY_EXCHANGE = "key_exchange"
    MASKED_CONTRIBUTION = "masked_contribution"
    AGGREGATION_RESULT = "aggregation_result"
    
    # Audit
    AUDIT_QUERY = "audit_query"
    AUDIT_RESPONSE = "audit_response"

@dataclass
class Message:
    msg_type: str
    sender_id: str
    data: Dict[str, Any]
    timestamp: float
    signature: Optional[str] = None
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        return cls(**json.loads(json_str))
