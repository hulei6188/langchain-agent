from enum import Enum


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"

    def __str__(self) -> str:
        return self.value


ROLE_ALIASES = {
    "owner": Role.ADMIN,
    "admin": Role.ADMIN,
    "member": Role.USER,
    "user": Role.USER,
}


def normalize_role(role: str | None) -> str:
    return str(ROLE_ALIASES.get(str(role or "").lower(), Role.USER))


def can_manage(role: str | None) -> bool:
    return normalize_role(role) == Role.ADMIN
