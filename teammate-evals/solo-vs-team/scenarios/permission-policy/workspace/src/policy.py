from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class Rule:
    effect: str
    action: str
    resource: str


@dataclass
class Role:
    name: str
    rules: list[Rule] = field(default_factory=list)
    inherits: tuple[str, ...] = ()


class PolicyEngine:
    def __init__(self, roles: Mapping[str, Role]) -> None:
        self.roles = dict(roles)

    def is_allowed(
        self,
        role_names: list[str],
        action: str,
        resource: str,
        context: Mapping[str, str] | None = None,
    ) -> bool:
        for role_name in role_names:
            role = self.roles.get(role_name)
            if role is None:
                continue
            for rule in role.rules:
                if rule.effect == "allow" and rule.action == action and rule.resource == resource:
                    return True
        return False
